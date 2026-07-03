import torch
import gc
import os
import re
from typing import List
from transformers import WhisperForConditionalGeneration, WhisperProcessor, BitsAndBytesConfig
from peft import PeftModel
from backend.utils.preprocess import chunk_audio
from backend.utils.demucs import separate_vocals_clean
from backend.src.text_stitch import stitch_overlapping_transcripts

def clean_production_text(text: str) -> str:
    """Drops token loops and aggressively strips trailing hyphen/punctuation artifacts."""
    if not text:
        return ""
    # Strip trailing hyphens, dashes, or loose punctuation at the end of the text block
    text = re.sub(r'[-–—?.,!]+$', '', text.strip())
    
    words = text.split()
    cleaned_words = []
    for word in words:
        # Loop breaker: if the same word repeats back-to-back, slice the chain
        if len(cleaned_words) >= 2 and all(w.lower() == word.lower() for w in cleaned_words[-2:]):
            break
        cleaned_words.append(word)
    return " ".join(cleaned_words)

def run_finetuned_separated_pipeline(audio_path: str, adapter_path: str, base_model_id: str = "openai/whisper-medium") -> str:
    """
    Pipeline B: Raw Audio -> Demucs Full Track Isolation (High-Fi) -> Chunker -> Fine-Tuned Whisper Adapters
    Integrates sliding-window token stitching to handle 5-second overlapping frames.
    """
    print("⏳ Stage 1 & 2: Running High-Fidelity Vocal Separation...")
    clean_vocal_signal = separate_vocals_clean(audio_path)
    
    print("✂️ Stage 3: Slicing isolated vocals into 30s arrays...")
    clean_vocal_chunks = chunk_audio(clean_vocal_signal, target_sr=16000)
    if not clean_vocal_chunks:
        return ""

    print(f"\n📥 Stage 4: Building fine-tuned model architecture...")
    processor = WhisperProcessor.from_pretrained(base_model_id, language="english", task="transcribe")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_id,
        device_map="auto",
        quantization_config=quantization_config
    )
    
    if os.path.exists(adapter_path):
        print(f"⚡ Hot-plugging fine-tuned QLoRA adapters from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print("⚠️ Adapter path not found! Falling back to native base model weights.")
        model = base_model
        
    model.eval()

    print(f"🚀 Transcribing {len(clean_vocal_chunks)} clean vocal frames...")
    transcriptions = []
    
    for idx, vocal_array_flat in enumerate(clean_vocal_chunks):
        try:
            input_features = processor.feature_extractor(
                vocal_array_flat, 
                sampling_rate=16000
            ).input_features[0]
            
            input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).to(model.device)
            
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features=input_tensor,
                    max_new_tokens=60,
                    max_length=None,                    # Silences the HF warning spam
                    repetition_penalty=1.2,             # Punish phrase loops
                    no_repeat_ngram_size=3,             # Hard ban on repeating phrases
                    forced_decoder_ids=processor.get_decoder_prompt_ids(language="english", task="transcribe")
                )
            
            raw_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            cleaned_text = clean_production_text(raw_text)
            
            if cleaned_text.strip():
                transcriptions.append(cleaned_text.strip())
                
            del input_tensor, predicted_ids
            
        except Exception as e:
            print(f"❌ Failed processing vocal layer chunk {idx}: {e}")
            continue
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    del model, base_model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("🧵 Stitching boundary text arrays...")
    return stitch_overlapping_transcripts(transcriptions, overlap_word_threshold=6)