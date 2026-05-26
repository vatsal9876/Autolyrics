import os
import gc
import torch
import evaluate
from transformers import (
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    WhisperProcessor,
    TrainerCallback
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# Import your native Hugging Face dataset structures
from dataset import prepare_hf_dataset, SpeechDataCollatorWithPadding

class GarbageCollectionCallback(TrainerCallback):
    """🛠️ MEMORY GUARD: Clears residual audio tensor allocations 
    at the completion of every logging step to avoid VRAM leaks.
    """
    def on_log(self, args, state, control, **kwargs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("🧹 [Memory Management] Flushed RAM & VRAM cache.")

def main():
    # Pre-clear graphics hardware registers before model load
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_id = "openai/whisper-tiny"  
    output_dir = "./models/whisper_lyrics_finetuned"
    
    # Manifest paths pointing to your preprocessed chunks
    train_manifest = "./data/english_mini_batch/preprocessed_manifest.jsonl"
    eval_manifest = "./data/english_mini_batch/preprocessed_manifest.jsonl" 

    print(f"📥 Loading official Hugging Face configs for: {model_id}")
    processor = WhisperProcessor.from_pretrained(model_id, language="english", task="transcribe")

    print("💿 Constructing streaming Hugging Face data pipeline...")
    train_dataset = prepare_hf_dataset(manifest_path=train_manifest, processor=processor)
    
    # 🎯 FIX: Restrict the evaluation stream directly at the dataset level using .take()
    # This prevents the trainer from running indefinitely on the stream.
    raw_eval_dataset = prepare_hf_dataset(manifest_path=eval_manifest, processor=processor)
    eval_dataset = raw_eval_dataset.take(10)
    
    data_collator = SpeechDataCollatorWithPadding(processor=processor)

    print("🎙️ Fetching model weights...")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    # 🔒 Hardware Constraint Shield: Freeze encoder layers
    model.freeze_encoder()

    # 📊 Setup Multi-Metrics Evaluation Suite
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    normalizer = BasicTextNormalizer()

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Unmask padding elements back to standard pad token IDs
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # 1. Calculate Raw Metrics
        raw_wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

        # 2. Calculate Normalized Metrics (Strips casing/punctuation for lyrics)
        norm_preds = [normalizer(p) for p in pred_str]
        norm_labels = [normalizer(l) for l in label_str]
        
        filtered_preds = [p for p, r in zip(norm_preds, norm_labels) if len(r.strip()) > 0]
        filtered_refs = [r for p, r in zip(norm_preds, norm_labels) if len(r.strip()) > 0]
        
        normalized_wer = 100 * wer_metric.compute(predictions=filtered_preds, references=filtered_refs) if filtered_refs else raw_wer

        return {
            "raw_wer": raw_wer,
            "normalized_wer": normalized_wer,
            "cer": cer
        }

    # Strict hardware training loop optimization configurations
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,       
        gradient_accumulation_steps=16,      
        learning_rate=1e-5,
        warmup_steps=30,
        max_steps=300,                       
        gradient_checkpointing=True,         
        fp16=True,                           
        
        eval_strategy="steps",               
        eval_steps=50,                       
        per_device_eval_batch_size=1,
        predict_with_generate=True,          
        generation_max_length=225,
        
        logging_steps=10,
        save_steps=100,
        report_to="none",
        
        load_best_model_at_end=True,
        metric_for_best_model="eval_normalized_wer",
        greater_is_better=False
    )

    # Instantiate the Trainer with correct positional configurations
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,             
        data_collator=data_collator,
        processing_class=processor.feature_extractor, 
        compute_metrics=compute_metrics,       
        callbacks=[GarbageCollectionCallback()]
    )

    print("\n🏋️‍♂️ Setup Verified. Starting Hugging Face fine-tuning loop on GPU...")
    trainer.train()

    # Save out production artifacts
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"\n🎉 Success! Fine-tuned weights exported directly to: {output_dir}")

if __name__ == "__main__":
    main()