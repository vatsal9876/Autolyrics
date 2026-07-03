import os
import gc
import sys
from datetime import datetime
from pathlib import Path
import torch
import jiwer
from transformers import (
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    WhisperProcessor,
    TrainerCallback,
    BitsAndBytesConfig,
)
from transformers.models.whisper.modeling_whisper import WhisperModel
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# Guardrail to protect model forward pass across re-runs
if not getattr(WhisperModel, "_is_patched_safely", False):
    _original_forward = WhisperModel.forward
    def _patched_forward(self, *args, **kwargs):
        if kwargs.get('input_ids', None) is None:
            kwargs.pop('input_ids', None)
        if kwargs.get('inputs_embeds', None) is None:
            kwargs.pop('inputs_embeds', None)
        return _original_forward(self, *args, **kwargs)
    WhisperModel.forward = _patched_forward
    WhisperModel._is_patched_safely = True

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_eval.dataset import prepare_hf_dataset

class GarbageCollectionCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    def on_evaluate(self, args, state, control, **kwargs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class TextEvaluationLoggerCallback(TrainerCallback):
    def __init__(self, run_id):
        self.log_dir = str(REPO_ROOT / "backend" / "eval_checkpoint")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, f"eval_history_{run_id}.txt")
        with open(self.log_path, "w") as f:
            f.write("="*95 + "\n")
            f.write(f"📊 COLAB RUN EVALUATION HISTORY | ID: {run_id}\n")
            f.write("="*95 + "\n")
            f.write(f"{'Step':<10}{'Epoch':<10}{'Train Loss':<12}{'Eval Loss':<12}{'WER %':<10}{'Norm WER %':<12}{'CER %':<10}{'Norm CER %':<12}\n")
            f.write("-"*95 + "\n")

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        step = state.global_step
        epoch = f"{state.epoch:.3f}"
        train_loss = "N/A"
        if len(state.log_history) > 0:
            for log in reversed(state.log_history):
                if "loss" in log and log.get("step") == step:
                    train_loss = f"{log['loss']:.4f}"
                    break
        eval_loss = f"{metrics.get('eval_loss', 0.0):.4f}"
        raw_wer = f"{metrics.get('eval_wer', 0.0):.2f}%" if 'eval_wer' in metrics else "N/A"
        norm_wer = f"{metrics.get('eval_wer_normalized', 0.0):.2f}%" if 'eval_wer_normalized' in metrics else "N/A"
        raw_cer = f"{metrics.get('eval_cer', 0.0):.2f}%" if 'eval_cer' in metrics else "N/A"
        norm_cer = f"{metrics.get('eval_cer_normalized', 0.0):.2f}%" if 'eval_cer_normalized' in metrics else "N/A"

        with open(self.log_path, "a") as f:
            f.write(f"{step:<10}{epoch:<10}{train_loss:<12}{eval_loss:<12}{raw_wer:<10}{norm_wer:<12}{raw_cer:<10}{norm_cer:<12}\n")
        print(f"📝 Evaluation metrics saved for step {step}")

def main():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_id = "openai/whisper-medium"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"whisper_lyrics_400steps_{timestamp}"
    output_dir = str(REPO_ROOT / "backend" / "models" / run_folder_name)

    manifest_path = str(REPO_ROOT / "data" / "chunks_metadata.jsonl")
    separated_vocals_dir = str(REPO_ROOT / "data" / "chunked_vocals")
    data_dir = str(REPO_ROOT / "data")
    train_val_split_file = os.path.join(data_dir, "train_val_chunks.txt")

    processor = WhisperProcessor.from_pretrained(model_id, language="english", task="transcribe")
    dataset = prepare_hf_dataset(manifest_path, processor, separated_vocals_dir, train_val_split_file)

    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"].select(range(min(1000, len(split["train"]))))
    eval_dataset = split["test"].select(range(min(24, len(split["test"]))))
    data_collator = SpeechDataCollatorWithPadding(processor=processor)

    jiwer_cleaner = jiwer.Compose([
        jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()
    ])

    def compute_metrics(eval_pred):
        predictions, label_ids = eval_pred
        decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
        labels = [[t if t != -100 else processor.tokenizer.pad_token_id for t in l] for l in label_ids]
        decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

        raw_truth, raw_pred = " ".join(decoded_labels), " ".join(decoded_preds)
        raw_wer = jiwer.wer(raw_truth, raw_pred) * 100
        raw_cer = jiwer.cer(raw_truth, raw_pred) * 100

        clean_preds = [jiwer_cleaner(t) for t in decoded_preds]
        clean_labels = [jiwer_cleaner(t) for t in decoded_labels]
        filtered_pairs = [(p, r) for p, r in zip(clean_preds, clean_labels) if r.strip()]

        if filtered_pairs:
            filtered_preds, filtered_labels = zip(*filtered_pairs)
            norm_truth, norm_pred = " ".join(filtered_labels), " ".join(filtered_preds)
            norm_wer = jiwer.wer(norm_truth, norm_pred) * 100
            norm_cer = jiwer.cer(norm_truth, norm_pred) * 100
        else:
            norm_wer, norm_cer = raw_wer, raw_cer

        return {"wer": raw_wer, "cer": raw_cer, "wer_normalized": norm_wer, "cer_normalized": norm_cer}

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", quantization_config=quantization_config, low_cpu_mem_usage=True
    )
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    model.config.suppress_tokens = []
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=32, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=0.15, bias="none", task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, lora_config)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=3e-5,
        warmup_steps=20,                    # Scaled for 400 steps
        max_steps=400,                      # 🚨 SET TO 400 STEPS
        gradient_checkpointing=True,
        fp16=True,
        predict_with_generate=True,
        prediction_loss_only=False,

        eval_strategy="steps",
        eval_steps=25,                      # 🚨 EVALUATE EVERY 25 STEPS
        per_device_eval_batch_size=1,

        logging_steps=25,
        save_strategy="steps",
        save_steps=25,                      # Save checkpoint files every 50 steps
        save_total_limit=None,
        report_to="none",
        load_best_model_at_end=False,
        optim="adamw_8bit",
        weight_decay=0.01,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        args=training_args, model=model, train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=data_collator, processing_class=processor, compute_metrics=compute_metrics,
        callbacks=[GarbageCollectionCallback(), TextEvaluationLoggerCallback(run_id=run_folder_name)]
    )

    print("\n🚀 Kicking off the full 400-step optimization run...")
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"\n🎉 Finished! Definitive run weights stored inside: {output_dir}")

if __name__ == "__main__":
    main()