import os
from datasets import load_dataset, Audio
from transformers import WhisperProcessor

def prepare_hf_dataset(manifest_path, processor):
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"❌ Missing manifest at: {manifest_path}. Run preprocess.py first!")

    # 1. Load the preprocessed JSONL file natively into an HF Dataset object
    # We use streaming=True to stream from disk, protecting your system memory
    dataset = load_dataset("json", data_files=manifest_path, split="train", streaming=True)

    # 2. Rename our manifest columns to standard Hugging Face speech definitions
    dataset = dataset.rename_column("audio_chunk_path", "audio")
    dataset = dataset.rename_column("sentence", "text")

    # 3. Explicitly cast the audio path column into an active Audio feature decoder (forces 16kHz Mono)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # 4. Map data items into Whisper-ready Log-Mel features and token arrays
    def transform_elements(batch):
        audio_data = batch["audio"]
        
        # Extract features from the raw array map decoded by HF Audio
        batch["input_features"] = processor.feature_extractor(
            audio_data["array"], 
            sampling_rate=audio_data["sampling_rate"]
        ).input_features[0]
        
        # Tokenize target text lyrics into clean label token IDs
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    # Remove raw text and structural audio metadata columns to leave only processed tensor arrays
    processed_dataset = dataset.map(
        transform_elements, 
        remove_columns=["audio", "text"]
    )
    
    return processed_dataset

class SpeechDataCollatorWithPadding:
    """Official Hugging Face padding strategy wrapper for Seq2Seq audio targets."""
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # Extract and pad input features uniformly (Whisper always pads audio inputs to 3000 frames)
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Extract and dynamically pad text label token sequences per batch sizing
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding target token boundaries with -100 to mask them from loss compute loops
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Strip standard leading BOS tags if applied by the specific variant setup
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch