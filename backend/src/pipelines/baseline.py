import torch
import gc
import re
from typing import List
from transformers import WhisperForConditionalGeneration, WhisperProcessor, BitsAndBytesConfig
from backend.utils.preprocess import chunk_audio
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

def run_baseline_pipeline(audio_path: str, model_id: str = "openai/whisper-medium") -> str:
    """
    Pipeline A: Raw Audio -> Chunker -> Native Whisper Medium (No Demucs)
    Integrates sliding-window token stitching to handle 5-second overlapping frames.
    """
    print("⏳ Stage 1: Slicing raw audio into 30s arrays...")
    ram_chunks = chunk_audio(audio_path, target_sr=16000)
    if not ram_chunks:
        return ""

    print(f"📥 Loading native 8-bit baseline architecture: {model_id}")
    processor = WhisperProcessor.from_pretrained(model_id, language="english", task="transcribe")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config
    )
    model.eval()

    print(f"🚀 Transcribing {len(ram_chunks)} chunks sequentially...")
    transcriptions = []
    
    for idx, chunk_array in enumerate(ram_chunks):
        try:
            input_features = processor.feature_extractor(
                chunk_array, 
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
            chunk_text = clean_production_text(raw_text)
            
            if chunk_text.strip():
                transcriptions.append(chunk_text.strip())
                
            del input_tensor, predicted_ids
            
        except Exception as e:
            print(f"❌ Failed processing chunk indices at {idx}: {e}")
            continue
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("🧵 Stitching boundary text arrays...")
    return stitch_overlapping_transcripts(transcriptions, overlap_word_threshold=6)