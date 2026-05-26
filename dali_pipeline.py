import os
import glob
import gzip
import pickle
import json
import subprocess
import gc  # Added for explicit garbage collection

def process_dali_annotations_and_audio(anno_dir="./data/dali_raw_annotations", output_audio_dir="./data/raw_songs"):
    """
    Directly extracts metadata from a folder of raw DALI .gz files,
    downloads the corresponding full audio track via yt-dlp, and
    compiles a clean line-level dataset manifest file.
    """
    os.makedirs(output_audio_dir, exist_ok=True)
    lines_dir = "./data/raw_lines"
    os.makedirs(lines_dir, exist_ok=True)
    manifest_path = os.path.join(lines_dir, "metadata.jsonl")

    # Locate the .gz annotation files
    gz_files = glob.glob(os.path.join(anno_dir, "*.gz"))
    if not gz_files:
        print(f"❌ Error: No DALI .gz files found inside '{anno_dir}'")
        return

    print(f"📦 Discovered {len(gz_files)} raw compressed DALI files. Starting extraction...")

    # Clear previous execution manifest to start clean
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    for idx, gz_path in enumerate(gz_files, 1):
        track_id = os.path.splitext(os.path.basename(gz_path))[0]
        print(f"\n🔄 [{idx}/{len(gz_files)}] Parsing file parameters for: {track_id}")

        try:
            # 1. Decompress and load the raw binary pickle stream
            with gzip.open(gz_path, "rb") as f:
                dali_object = pickle.load(f)

            # 2. Extract the embedded YouTube unique video string ID
            track_info = getattr(dali_object, 'info', {})
            audio_info = track_info.get('audio', {})
            youtube_id = audio_info.get('url', None)

            if not youtube_id or youtube_id == "":
                print(f"  ⏩ Skipped: No embedded YouTube video code found inside object metadata.")
                del dali_object
                gc.collect()
                continue

            # Construct the target streaming URL link
            youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"

            # 3. Pull line alignments array matching the official structural mapping
            annotations_core = getattr(dali_object, 'annotations', {})
            annot_dict = annotations_core.get('annot', {})
            raw_lines = annot_dict.get('lines', [])

            if not raw_lines:
                print(f"  ⏩ Skipped: Annotation lines array is missing or empty.")
                del dali_object
                gc.collect()
                continue

            # 4. Invoke yt-dlp via subprocess to grab and transcode the audio track
            target_wav_path = os.path.join(output_audio_dir, f"{track_id}.wav")
            print(f"  📥 Fetching media stream track from: {youtube_url}")
            
            download_cmd = [
                "yt-dlp",
                "-x",                                # Extract audio tracks only
                "--audio-format", "wav",             # Force clean WAV wrapper encoding
                "--audio-quality", "0",              # Enforce top-tier fidelity parameters
                
                # 🏎️ THE ACTUAL SPEED BOOSTERS:
                "--concurrent-fragments", "8",       # Open 8 parallel download connections per song
                "--buffer-size", "16K",              # Optimize network chunk buffering blocks
                
                "-o", os.path.join(output_audio_dir, f"{track_id}.%(ext)s"),
                youtube_url
            ]

            # Execute system shell downloader command silently in background
            result = subprocess.run(download_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            
            if result.returncode != 0 or not os.path.exists(target_wav_path):
                print(f"  ⚠️ Link Failure: The video may be deleted, private, or region-restricted.")
                del dali_object
                gc.collect()
                continue

            print(f"  💾 High-fidelity audio saved to: {target_wav_path}")

            # 5. Append formatted properties straight into your central metadata tracking registry
            with open(manifest_path, "a", encoding="utf-8") as manifest:
                for line in raw_lines:
                    text_content = line.get('text', '')
                    time_array = line.get('time', [0.0, 0.0])

                    entry = {
                        "track_id": track_id,
                        "audio_file": f"{track_id}.wav",
                        "start": float(time_array[0]),
                        "end": float(time_array[1]),
                        "text": str(text_content).strip()
                    }
                    manifest.write(json.dumps(entry) + "\n")

            print(f"  ✅ Successfully mapped {len(raw_lines)} aligned sequences to manifest.")

            # Clear memory pointers right here before moving to the next track iteration
            del dali_object, track_info, audio_info, annotations_core, annot_dict, raw_lines
            gc.collect()

        except Exception as e:
            print(f"  ❌ Parse Error on file {os.path.basename(gz_path)}: {str(e)}")
            try:
                del dali_object
            except NameError:
                pass
            gc.collect()
            continue

    print(f"\n🎉 Extraction Complete! Data paths are structured and ready inside: {lines_dir}")

if __name__ == "__main__":
    # Point this strictly to your single folder holding the .gz files
    process_dali_annotations_and_audio(anno_dir="YOUR_ANNO_DIR")
