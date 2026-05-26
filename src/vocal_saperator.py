import os
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import gc

def saperate_audio_stems(input_list_path, input_dir, output_dir):
    # Ensure the main output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_list_path):
        print(f"Input list file not found at {input_list_path}. Please check the path and try again.")
        return
    
    if not os.path.exists(input_dir):
        print(f"Input directory not found at {input_dir}. Please check the path and try again.")
        return

    selected_tracks = []
    with open(input_list_path, "r", encoding="utf-8") as f:
        selected_tracks = [line.strip() for line in f if line.strip()]

    if not selected_tracks:
        print("No tracks found in the input list. Please check the file and try again.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Initializing demucs pipeline on device: ", device)

    model = get_model("htdemucs")
    model.to(device)
    model.eval()

    # Native Demucs standard processing rate
    target_sample_rate = 44100

    for idx, track_id in enumerate(selected_tracks):
        audio_path = os.path.join(input_dir, f"{track_id}.wav")
        
        # 🎯 CHANGED: File path is now flat inside output_dir -> [track_id]_vocals.wav
        vocals_path = os.path.join(output_dir, f"{track_id}_vocals.wav")

        # Skip checkpoint looks for the specific file name now
        if os.path.exists(vocals_path):
            print(f"⏩ [{idx+1}/{len(selected_tracks)}] Already processed. Skipping: {track_id}")
            continue

        if not os.path.exists(audio_path):
            print(f"⚠️ Warning: Audio file not found for track {track_id} at {audio_path}. Skipping.")
            continue

        print(f"\n🎛️ [{idx+1}/{len(selected_tracks)}] Splitting stems for: {track_id}")

        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != target_sample_rate:
                print(f"Resampling audio from {sample_rate} Hz to {target_sample_rate} Hz")
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                waveform = resampler(waveform)

            waveform_tensor = waveform.to(device).unsqueeze(0)

            with torch.no_grad():
                split = apply_model(model, waveform_tensor, split=True)[0]

            # 🎯 CHANGED: Saving directly into the flat file path
            torchaudio.save(vocals_path, split[3].cpu(), target_sample_rate)
            print(f"💾 Saved flat track file: {vocals_path}")

            del waveform, waveform_tensor, split
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"❌ Error processing {track_id}: {str(e)}")
            try:
                del waveform, waveform_tensor, split
            except NameError:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

if __name__ == "__main__":
    os.makedirs("./data/raw_songs", exist_ok=True)
    
    saperate_audio_stems(
        input_list_path="./data/english_mini_batch/english_selected_tracks.txt", 
        input_dir="./data/raw_songs", 
        output_dir="./data/separated"
    )