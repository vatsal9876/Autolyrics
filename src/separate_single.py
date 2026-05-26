import os
import sys
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model

def separate_one_track(track_path, output_dir="./data/separated"):
    track_id = os.path.splitext(os.path.basename(track_path))[0]
    target_track_dir = os.path.join(output_dir, track_id)
    target_vocal_path = os.path.join(target_track_dir, "vocals.wav")
    
    # Skip if already processed
    if os.path.exists(target_vocal_path):
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model("htdemucs")
    model.to(device)
    model.eval()

    try:
        # Load audio explicitly using stable soundfile backend
        waveform, sample_rate = torchaudio.load(track_path, backend="soundfile")
        
        # Align sample rate with Demucs requirements
        if sample_rate != model.samplerate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=model.samplerate)
            waveform = resampler(waveform)
            sample_rate = model.samplerate
        
        waveform_tensor = waveform.to(device).unsqueeze(0)
        
        # Run separation with window splitting to respect 4GB VRAM ceiling
        with torch.no_grad():
            sources = apply_model(model, waveform_tensor, split=True)[0]
        
        # Extract vocal stem (index 3)
        vocals = sources[3].cpu()
        os.makedirs(target_track_dir, exist_ok=True)
        torchaudio.save(target_vocal_path, vocals, sample_rate)
        print(f"💾 Clean vocal stem saved for: {track_id}")
        
    except Exception as e:
        print(f"⚠️ Failed to process track {track_id}: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        separate_one_track(sys.argv[1])