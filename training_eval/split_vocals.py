import os
import gc
import sys
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from demucs.pretrained import get_model
from demucs.apply import apply_model

# Keep terminal logs clean and fast
import warnings
warnings.filterwarnings("ignore")

def separate_dataset_optimized(input_source, output_dir):
    """
    Highly optimized batch-processing pipeline for 4GB VRAM GPUs.
    Loads the model weights ONCE and processes chunks sequentially.
    
    Args:
        input_source (str or list): Path to raw chunks directory OR a explicit list of file paths.
        output_dir (str): Flat target directory to save output vocal stems.
    """
    # 1. Hardware Verification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Target Hardware Ceiling: {device.upper()}")
    
    # Parse input: can handle a directory path or a direct list of chunk paths
    if isinstance(input_source, (str, Path)) and os.path.isdir(input_source):
        audio_extensions = {".wav", ".mp3", ".flac", ".m4a"}
        chunk_paths = [p for p in Path(input_source).rglob("*") if p.suffix.lower() in audio_extensions]
    elif isinstance(input_source, list):
        chunk_paths = [Path(p) for p in input_source]
    else:
        print("❌ Error: Invalid input source. Provide a directory path or a list of file paths.")
        return

    if not chunk_paths:
        print("⚠️ No valid audio chunks detected in queue.")
        return

    print(f"📦 Queue initialized: {len(chunk_paths)} targets ready for source separation.")
    os.makedirs(output_dir, exist_ok=True)

    # 2. GPU Optimization: Load Backbone Architecture EXACTLY ONCE
    print("📥 Loading HTDemucs weights into permanent VRAM memory allocation...")
    model = get_model("htdemucs")
    model.to(device)
    model.eval()
    print("✅ Model pinned cleanly in GPU memory shell.")

    # 3. High-Speed Pipeline Processing Loop
    print("\n🚀 Initiating optimized batch separation pipeline...")
    processed_count = 0

    for track_path in tqdm(chunk_paths, desc="Batch Separating"):
        chunk_filename = track_path.name
        target_vocal_path = os.path.join(output_dir, chunk_filename)

        # Skip Layer: Fast recovery block if run was interrupted previously
        if os.path.exists(target_vocal_path):
            processed_count += 1
            continue

        if not track_path.exists():
            continue

        try:
            # High-speed soundfile I/O streaming
            waveform, sample_rate = torchaudio.load(str(track_path), backend="soundfile")
            
            # 🎯 FIXED: Convert Mono (1 channel) to Stereo (2 channels)
            if waveform.shape[0] == 1:
                # Duplicates the single channel to create a [2, time] tensor
                waveform = torch.cat([waveform, waveform], dim=0)
            elif waveform.shape[0] > 2:
                # Downmix multi-channel (like 5.1 surround) to 2 channels just in case
                waveform = waveform[:2, :]
            
            # Align track frequencies to HTDemucs requirements (44100Hz native)
            if sample_rate != model.samplerate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=model.samplerate)
                waveform = resampler(waveform)
                sample_rate = model.samplerate
            
            # Wrap array frame shapes into a standard single-batch tensor [1, 2, time]
            waveform_tensor = waveform.to(device).unsqueeze(0)
            
            # VRAM GUARDRAIL: split=True chunks the audio array internally to protect 4GB ceiling
            with torch.no_grad():
                sources = apply_model(model, waveform_tensor, split=True)[0]
            
            # Extract pure vocals tensor (Demucs index 3 = Vocals)
            vocals = sources[3].cpu()
            
            # Save flat directly into target directory by chunk name only for metadata alignment
            torchaudio.save(target_vocal_path, vocals, sample_rate)
            processed_count += 1
            
            # Explicitly drop references to massive tensor blocks to clear active scope
            del waveform_tensor, sources, vocals
            
        except Exception as e:
            print(f"\n⚠️ Failed to process chunk {chunk_filename}: {str(e)}")
            continue
            
        finally:
            # 🎯 GPU OPTIMIZATION: Enforce garbage collection and flush CUDA caching memory
            # This prevents the VRAM pool from fragmenting or building up over long runs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n🎉 Done! {processed_count} vocal chunks mapped directly inside: {output_dir}")


if __name__ == "__main__":
    # Example Production/Evaluation invocation:
    raw_chunks_input = "./data/jam_alt_lines/pure/en/audio"
    flat_vocals_output = "./data/jam_alt_lines/pure/en/vocals"
    
    separate_dataset_optimized(input_source=raw_chunks_input, output_dir=flat_vocals_output)