import os
import gc
import sys
import torch
from pathlib import Path
from tqdm import tqdm
import jiwer
import warnings
from transformers import WhisperForConditionalGeneration, WhisperProcessor, BitsAndBytesConfig
from peft import PeftModel

# Suppress runtime warnings to keep terminal output clean
warnings.filterwarnings("ignore")

# Force inclusion of the repo root so training_eval packages can be imported consistently
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training_eval.dataset import prepare_hf_dataset


def filter_hallucination_loops(text):
    """
    Post-processing circuit breaker for autoregressive token loops.
    Drops repeating sequence loops natively before final scoring.
    """
    words = text.split()
    if len(words) == 0:
        return text
    cleaned_words = []
    for word in words:
        if len(cleaned_words) >= 3 and all(w == word for w in cleaned_words[-3:]):
            break
        cleaned_words.append(word)
    return " ".join(cleaned_words)


class ModelEvaluator:
    """Evaluate Whisper base vs PEFT fine-tuned models with total corpus strings."""
    
    def __init__(self, model_id="openai/whisper-medium", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_id = model_id
        self.processor = WhisperProcessor.from_pretrained(model_id, language="english", task="transcribe")
        
        # Explicit normalization pipeline applied directly to text strings before scoring
        self.jiwer_cleaner = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip()
        ])
        print(f"✅ Loaded processor and normalization pipeline from {model_id}")
    
    def load_model(self, model_path=None):
        """Load model in 8-bit to stay within 4GB VRAM limits, hot-plugging adapters if path exists."""
        print(f"📥 Loading backbone architecture: {self.model_id}")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        base_model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=quantization_config
        )
        
        if model_path and os.path.exists(model_path):
            print(f"⚡ Hot-plugging trained QLoRA adapters from: {model_path}")
            model = PeftModel.from_pretrained(base_model, model_path)
            print(f"✅ Successfully loaded fine-tuned PEFT architecture")
        else:
            model = base_model
            print(f"✅ Loaded native base model weights")
        
        model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="english", task="transcribe"
        )
        model.config.suppress_tokens = []
        model.eval()
        return model
    
    def prepare_test_dataset(self, manifest_path, audio_dir):
        """Prepare test dataset partition by passing a clean folder structure to the dynamic dataset layout."""
        print(f"💿 Ingesting dataset pipeline tracking from: {audio_dir} ...")
        
        # Bypasses static test_chunks.txt tracking completely by passing split_file=None
        dataset = prepare_hf_dataset(
            manifest_path=manifest_path,
            processor=self.processor,
            audio_dir=audio_dir,
            split_file="BYPASS_SPLIT_FILE_MAPPING"
        )
        print(f"📊 Dataset target pipeline initialized. Processing {len(dataset)} entry targets.")
        return dataset
    
    def transcribe_batch(self, model, batch):
        """Transcribe a batch of samples safely with explicit type allocation, loop breaking, and language anchoring."""
        with torch.no_grad():
            features_list = []
            for item in batch:
                feat = torch.tensor(item["input_features"], dtype=torch.float32)
                features_list.append(feat)
            
            input_features = torch.stack(features_list).to(model.device)
            
            # Repetition parameters aligned perfectly to control model drift
            predicted_ids = model.generate(
                input_features=input_features,
                max_new_tokens=60,
                repetition_penalty=1.1,
                no_repeat_ngram_size=0,
                forced_decoder_ids=self.processor.get_decoder_prompt_ids(language="english", task="transcribe")
            )
            
            raw_predictions = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            # Filter loops instantly before metrics aggregation calculations
            filtered_predictions = [filter_hallucination_loops(pred) for pred in raw_predictions]
            
            ground_truths = []
            for item in batch:
                label_ids = [t for t in item["labels"] if t != -100]
                text = self.processor.tokenizer.decode(label_ids, skip_special_tokens=True)
                ground_truths.append(text)
            
            return raw_predictions, filtered_predictions, ground_truths
    
    def evaluate(self, model, test_dataset, batch_size=2):
        """Evaluate model on test dataset and return true aggregate corpus error metrics."""
        all_raw_predictions = []
        all_filtered_predictions = []
        all_ground_truths = []
        
        print(f"\n🔄 Running batch inference (batch_size={batch_size})...")
        total_batches = (len(test_dataset) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(test_dataset), batch_size), total=total_batches, desc="Transcribing"):
            batch_indices = list(range(i, min(i + batch_size, len(test_dataset))))
            batch = [test_dataset[idx] for idx in batch_indices]
            
            try:
                raw_preds, filtered_preds, ground_truths = self.transcribe_batch(model, batch)
                all_raw_predictions.extend(raw_preds)
                all_filtered_predictions.extend(filtered_preds)
                all_ground_truths.extend(ground_truths)
            except Exception as e:
                print(f"\n⚠️ Skipping problematic batch starting at index {i} due to missing chunk or processing error: {e}")
                continue
            
            del batch, raw_preds, filtered_preds, ground_truths
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print("\n📊 Aggregating unified corpus error statistics...")
        
        # 1. RAW STRUCTURAL METRICS (Unified continuous blocks to bypass averaging calculations)
        raw_truth_corpus = " ".join(all_ground_truths)
        raw_unfiltered_pred_corpus = " ".join(all_raw_predictions)
        wer_raw_unfiltered = jiwer.wer(raw_truth_corpus, raw_unfiltered_pred_corpus) * 100
        
        # 2. TRUE NORMALIZED METRICS (Text cleaned and processed before corpus metrics computation)
        clean_ground_truths = [self.jiwer_cleaner(text) for text in all_ground_truths]
        clean_predictions = [self.jiwer_cleaner(text) for text in all_filtered_predictions]
        clean_raw_unfiltered = [self.jiwer_cleaner(text) for text in all_raw_predictions]
        
        # Safe structural splits to completely eliminate empty division faults
        final_refs = [r for r in clean_ground_truths if len(r.strip()) > 0]
        final_preds = [p for r, p in zip(clean_ground_truths, clean_predictions) if len(r.strip()) > 0]
        final_raw_preds = [p for r, p in zip(clean_ground_truths, clean_raw_unfiltered) if len(r.strip()) > 0]
        
        if len(final_refs) > 0:
            norm_truth_corpus = " ".join(final_refs)
            norm_pred_corpus = " ".join(final_preds)
            norm_raw_pred_corpus = " ".join(final_raw_preds)
            
            true_normalized_wer = jiwer.wer(norm_truth_corpus, norm_pred_corpus) * 100
            true_normalized_cer = jiwer.cer(norm_truth_corpus, norm_pred_corpus) * 100
            true_raw_unfiltered_wer = jiwer.wer(norm_truth_corpus, norm_raw_pred_corpus) * 100
        else:
            true_normalized_wer = 100.0
            true_normalized_cer = 100.0
            true_raw_unfiltered_wer = wer_raw_unfiltered
        
        return {
            "wer_unfiltered": true_raw_unfiltered_wer,
            "wer_normalized": true_normalized_wer, 
            "cer_normalized": true_normalized_cer,
            "num_samples": len(test_dataset),
            "raw_predictions": all_raw_predictions,
            "filtered_predictions": all_filtered_predictions,
            "ground_truths": all_ground_truths
        }

    def _sanitize_model_filename(self, parent_folder, checkpoint_id):
        filename = f"{parent_folder}_{checkpoint_id}_comparative_report.txt"
        filename = "".join(ch for ch in filename if ch.isalnum() or ch in "._-")
        return filename

    def save_results_to_file(self, ft_results, base_results, parent_folder, checkpoint_id, output_dir=None):
        """Export comprehensive comparative matrix logs detailing performance shifts side by side."""
        if output_dir is None:
            output_dir = str(REPO_ROOT / "backend" / "evaluation_results")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        report_name = self._sanitize_model_filename(parent_folder, checkpoint_id)
        report_path = Path(output_dir) / report_name
        
        base_wer = base_results['wer_normalized']
        base_cer = base_results['cer_normalized']
        ft_wer = ft_results['wer_normalized']
        ft_cer = ft_results['cer_normalized']
        ft_raw_wer = ft_results['wer_unfiltered']
        
        with report_path.open("w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write(f"📈 COMPARATIVE BENCHMARK LOG: BASE MODEL VS {checkpoint_id}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total Test Sequence Tracks Evaluated: {ft_results['num_samples']}\n\n")

            f.write("📊 ACCURACY ERROR METRICS MATRIX:\n")
            f.write("-" * 55 + "\n")
            f.write(f"│ Metric            │ Base Model     │ Fine-Tuned ({checkpoint_id}) \n")
            f.write("-" * 55 + "\n")
            f.write(f"│ Normalized WER    │ {base_wer:>12.2f}% │ {ft_wer:>19.2f}% \n")
            f.write(f"│ Normalized CER    │ {base_cer:>12.2f}% │ {ft_cer:>19.2f}% \n")
            f.write(f"│ Raw Unfiltered WER│ {base_wer:>12.2f}% │ {ft_raw_wer:>19.2f}% \n")
            f.write("-" * 55 + "\n\n")

            f.write(f"🚀 OVERALL ACCOUNTABILITY SHIFT (Normalized):\n")
            f.write(f" • Word Error Rate Reduction: {-(ft_wer - base_wer):+0.2f}%\n")
            f.write(f" • Char Error Rate Reduction: {-(ft_cer - base_cer):+0.2f}%\n\n")

            f.write("📝 SIDE-BY-SIDE INFERENCE SAMPLES COMPARISON (FIRST 10 CHUNKS):\n" + "─"*80 + "\n")
            for i in range(min(10, len(ft_results['ground_truths']))):
                f.write(f"\n[TRACK CHUNK CHANNELS {i+1}]\n")
                f.write(f"  • GROUND TRUTH REFERENCE   : {ft_results['ground_truths'][i].strip()}\n")
                f.write(f"  • BASE MODEL OUT-OF-BOX    : {base_results['filtered_predictions'][i].strip()}\n")
                f.write(f"  • FT ADAPTER ({checkpoint_id} RAW)  : {ft_results['raw_predictions'][i].strip()}\n")
                f.write(f"  • FT ADAPTER ({checkpoint_id} CLEAN): {ft_results['filtered_predictions'][i].strip()}\n")
                f.write("-" * 80 + "\n")
                
        print(f"📁 Comparative report successfully written to: {report_path}")
        return report_path


def main():
    manifest_path = str(REPO_ROOT / "data" / "jam_alt_lines" / "pure" / "en" / "metadata.jsonl")
    
    # TARGET TRACK CONTAINER: Path to the directory where all clean vocal chunks are saved flat
    separated_vocals_dir = str(REPO_ROOT / "data" / "jam_alt_lines" / "pure" / "en" / "vocals")
    
    # Timestamped training folder name configurations
    parent_run_folder = "whisper_lyrics_400steps_20260702_145100"
    model_base_dir = str(REPO_ROOT / "backend" / "models")
    
    # Targeted checkpoint snapshot selection bounds
    target_checkpoints = ["checkpoint-150"]
    
    if not os.path.exists(separated_vocals_dir):
        print(f"❌ Target vocal chunk evaluation folder missing at: {separated_vocals_dir}")
        return
    
    # Initialize infrastructure tracking against Whisper-Medium
    evaluator = ModelEvaluator(model_id="openai/whisper-medium")
    test_dataset = evaluator.prepare_test_dataset(
        manifest_path=manifest_path,
        audio_dir=separated_vocals_dir
    )
    
    # =========================================================
    # 🎯 STEP 1: EVALUATE BASELINE OUT-OF-THE-BOX MODEL (ONCE)
    # =========================================================
    print("\n" + "="*80 + "\n🎯 EVALUATING OUT-OF-THE-BOX BASELINE MODEL\n" + "="*80)
    base_model = evaluator.load_model(model_path=None)
    base_results = evaluator.evaluate(base_model, test_dataset, batch_size=2)
    
    # Complete VRAM wipe of baseline weights to give adapters full hardware headroom
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # =========================================================
    # 🎯 STEP 2: RECURSIVELY LOOP OVER SAVED ADAPTER SNAPSHOTS
    # =========================================================
    for ckpt in target_checkpoints:
        finetuned_model_path = os.path.join(model_base_dir, parent_run_folder, ckpt)
        
        print("\n" + "="*80 + f"\n🔄 EVALUATING ADAPTER EXPERIMENT SNAPSHOT: {ckpt}\n" + "="*80)
        if os.path.exists(finetuned_model_path):
            # Dynamic adapter wrapper initialization
            finetuned_model = evaluator.load_model(model_path=finetuned_model_path)
            finetuned_results = evaluator.evaluate(finetuned_model, test_dataset, batch_size=2)
            
            # Print console metrics summary logs immediately
            print(f"\n📈 OVERALL PERFORMANCE OVERVIEW FOR {ckpt}:")
            print(f" 🔹 Base Model WER              : {base_results['wer_normalized']:.2f}%")
            print(f" 🔹 Fine-Tuned WER (Filtered)   : {finetuned_results['wer_normalized']:.2f}% (Raw Unfiltered: {finetuned_results['wer_unfiltered']:.2f}%)")
            print(f" 🔹 Fine-Tuned CER (Filtered)   : {finetuned_results['cer_normalized']:.2f}%")
            
            # Compile matrix report to file logs on disk safely
            evaluator.save_results_to_file(
                ft_results=finetuned_results, 
                base_results=base_results, 
                parent_folder=parent_run_folder, 
                checkpoint_id=ckpt
            )
            
            # Flush loop-specific GPU allocation weights before proceeding
            del finetuned_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(f"❌ Target adapter folder missing on storage layer, skipping: {finetuned_model_path}")

    print(f"\n🎉 Evaluation complete. Model-specific records compiled under: {REPO_ROOT / 'backend' / 'evaluation_results'}")


if __name__ == "__main__":
    main()