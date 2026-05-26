import os
import json
import torch
import torchaudio

def chunk_audio_segments(manifest_path, vocals_dir, output_list_path):
    """Slices long vocal tracks into tiny line-level audio samples safely."""
    os.makedirs(os.path.dirname(output_list_path), exist_ok=True)
    
    chunks_output_dir = "./data/audio_chunks"
    os.makedirs(chunks_output_dir, exist_ok=True)

    print("📖 Reading English line coordinates from manifest...")
    processed_records = []
    
    if not os.path.exists(manifest_path):
        print(f"❌ Error: Manifest path does not exist: {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"📋 Found {len(lines)} lines in manifest. Processing chunks...")

    for idx, line in enumerate(lines):
        try:
            entry = json.loads(line.strip())
            track_id = entry["track_id"]
            vocals_file = os.path.join(vocals_dir, f"{track_id}_vocals.wav")
            
            if not os.path.exists(vocals_file):
                continue
                
            start_time = float(entry["start"])
            end_time = float(entry["end"])
            lyric_text = entry["text"].strip()
            
            if not lyric_text:
                continue  # Skip instrumental spaces

            chunk_filename = f"{track_id}_line_{idx}.wav"
            chunk_save_path = os.path.join(chunks_output_dir, chunk_filename)

            # FIXED: Avoid using torchaudio.info() for older version compatibility
            full_waveform, sr = torchaudio.load(vocals_file)
            
            start_frame = int(start_time * sr)
            end_frame = int(end_time * sr)
            
            # Slice via standard PyTorch tensor slicing
            waveform = full_waveform[:, start_frame:end_frame]
            
            if waveform.shape[1] <= 0:
                continue

            # Force Mono mixdown if file is Stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample to Whisper's mandatory 16,000 Hz standard on the fly
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)

            # Save out the slice
            torchaudio.save(chunk_save_path, waveform, 16000)

            # Add to manifest list
            processed_records.append({
                "audio_chunk_path": chunk_save_path,
                "sentence": lyric_text
            })

            # Print progress indicator every 100 chunks
            if len(processed_records) % 100 == 0:
                print(f"✨ Successfully sliced {len(processed_records)} audio lines...")

        except Exception as e:
            print(f"❌ Error processing line {idx} (Track: {track_id if 'track_id' in locals() else 'Unknown'}): {str(e)}")
            continue

    # Write out the new manifest index
    with open(output_list_path, "w", encoding="utf-8") as out_f:
        for record in processed_records:
            out_f.write(json.dumps(record) + "\n")

    print(f"\n✂️ Preprocessing complete! Isolated {len(processed_records)} clean 16kHz audio line fragments.")
    print(f"📝 New training index manifest saved to: {output_list_path}")

if __name__ == "__main__":
    chunk_audio_segments(
        manifest_path="./data/english_mini_batch/english_metadata.jsonl",
        vocals_dir="./data/separated",
        output_list_path="./data/english_mini_batch/preprocessed_manifest.jsonl"
    )