import os
from datasets import load_dataset, Audio
from transformers import WhisperProcessor

def load_split_mapping(split_file):
    """Load chunk names from split mapping file safely."""
    if split_file is None or not os.path.exists(split_file):
        return None
    with open(split_file, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def prepare_hf_dataset(manifest_path, processor, audio_dir=None, split_file=None):
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"❌ Missing manifest at: {manifest_path}. Run preprocess.py first!")

    allowed_chunks = load_split_mapping(split_file)

    # 1. Load the preprocessed JSONL file into an HF Dataset object
    dataset = load_dataset("json", data_files=manifest_path, split="train")

    # Filter by split if mapping file is provided and valid
    if allowed_chunks is not None:
        def filter_by_split(example):
            # Dynamic fallback checking depending on schema layout
            col_target = "file_name" if "file_name" in example else "audio_chunk_path"
            if col_target not in example:
                return True
            filename = os.path.basename(example[col_target])
            return filename in allowed_chunks
        
        dataset = dataset.filter(filter_by_split)
        print(f"🔍 Filtered dataset from split mapping")

    # 2. Rename columns to standard Hugging Face speech definitions dynamically
    if "file_name" in dataset.column_names:
        dataset = dataset.rename_column("file_name", "audio")
    elif "audio_chunk_path" in dataset.column_names:
        dataset = dataset.rename_column("audio_chunk_path", "audio")

    if "sentence" in dataset.column_names:
        dataset = dataset.rename_column("sentence", "text")
    # If the column name is already 'text', it skips renaming entirely

    # 3. Redirect the audio path to the separated vocals directory if provided
    if audio_dir is not None:
        def rewrite_audio_path(example):
            filename = os.path.basename(example["audio"])
            example["audio"] = os.path.join(audio_dir, filename)
            return example

        dataset = dataset.map(rewrite_audio_path)

    # 4. Explicitly cast the audio path column into an active Audio feature decoder (forces 16kHz Mono)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # 5. Map individual row items into Log-Mel features and text token arrays
    def transform_elements(example):
        audio_data = example["audio"]

        # Extract features from the raw array map decoded by HF Audio
        example["input_features"] = processor.feature_extractor(
            audio_data["array"], 
            sampling_rate=audio_data["sampling_rate"]
        ).input_features[0]

        # Explicitly tokenize text directly into a list of target IDs.
        example["labels"] = processor.tokenizer(str(example["text"])).input_ids
        return example

    # Remove raw columns to leave only processed tensor arrays
    # CRITICAL: load_from_cache_file=False forces HF to drop old broken disk caches!
    processed_dataset = dataset.map(
        transform_elements, 
        remove_columns=[col for col in dataset.column_names if col not in ["input_features", "labels"]],
        batched=False,
        load_from_cache_file=False
    )
    
    return processed_dataset

class SpeechDataCollatorWithPadding:
    """Official Hugging Face padding strategy wrapper for Seq2Seq audio targets."""
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"]
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
        if "attention_mask" in labels_batch:
            labels = labels.masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch