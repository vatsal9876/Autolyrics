import os
import time
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import gc

def saperate_audio_stems(input_dir, output_dir, batch_size=8, watch=False):
    """Separate vocals from chunk WAV files in a directory.

    Reads all WAV files from `input_dir`, preserves each filename, and writes
    separated vocals directly into `output_dir` with the same filename.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        print(f"Input directory not found at {input_dir}. Please check the path and try again.")
        return

    batch_separate_chunks(chunks_dir=input_dir, output_dir=output_dir, batch_size=batch_size, watch=watch)

# Note: use this module's functions programmatically or run the batch watcher below.


def batch_separate_chunks(chunks_dir="./data/audio_chunks", output_dir="./data/chunked_vocals", batch_size=8, sleep_idle=5, watch=False):
    """Process WAV chunks in batches from an input directory.

    Reads all WAV files from `chunks_dir` and writes separated vocal files into
    `output_dir/<track_id>/<same_filename>`. This function does not modify any metadata.
    """
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Initializing demucs pipeline on device: ", device)
    model = get_model("htdemucs")
    model.to(device)
    model.eval()

    model_sample_rate = model.samplerate if hasattr(model, 'samplerate') else 44100
    output_sample_rate = 16000

    try:
        while True:
            # Collect unprocessed chunk files
            all_chunks = [f for f in os.listdir(chunks_dir) if f.lower().endswith('.wav')]
            all_chunks.sort()
            to_process = []
            for fname in all_chunks:
                target_path = os.path.join(output_dir, fname)
                if not os.path.exists(target_path):
                    to_process.append(fname)

            if not to_process:
                if watch:
                    time.sleep(sleep_idle)
                    continue
                else:
                    # nothing to do and not watching -> exit
                    break

            batch = to_process[:batch_size]
            chunk_paths = [os.path.join(chunks_dir, b) for b in batch]

            # Load and prepare waveforms
            waveforms = []
            for path in chunk_paths:
                waveform, sr = torchaudio.load(path)
                if sr != model_sample_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=model_sample_rate)
                    waveform = resampler(waveform)

                # Demucs expects stereo input for htdemucs; duplicate mono if needed.
                if waveform.shape[0] == 1:
                    waveform = waveform.repeat(2, 1)
                elif waveform.shape[0] > 2:
                    waveform = waveform[:2, :]

                waveforms.append(waveform)

            # Pad to same length
            max_len = max(w.shape[1] for w in waveforms)
            padded = []
            for w in waveforms:
                if w.shape[1] < max_len:
                    pad = torch.zeros((w.shape[0], max_len - w.shape[1]))
                    w = torch.cat([w, pad], dim=1)
                padded.append(w)

            batch_tensor = torch.stack(padded, dim=0).to(device)

            # Run separation once for the batch
            try:
                with torch.no_grad():
                    outputs = apply_model(model, batch_tensor, split=True)

                # apply_model may return a list/tuple; normalize
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]

                # Expected outputs shape: (batch, stems, channels, samples)
                if outputs.ndim == 4:
                    for i, fname in enumerate(batch):
                        track_stem = outputs[i]
                        vocals = track_stem[3].cpu()
                        if model_sample_rate != output_sample_rate:
                            vocals = torchaudio.transforms.Resample(orig_freq=model_sample_rate, new_freq=output_sample_rate)(vocals)
                        target_path = os.path.join(output_dir, fname)
                        torchaudio.save(target_path, vocals, output_sample_rate)
                        print(f"💾 Saved vocal chunk: {fname}")
                else:
                    # Fallback for single output
                    for i, fname in enumerate(batch):
                        vocals = outputs[3].cpu()
                        if model_sample_rate != output_sample_rate:
                            vocals = torchaudio.transforms.Resample(orig_freq=model_sample_rate, new_freq=output_sample_rate)(vocals)
                        target_path = os.path.join(output_dir, fname)
                        torchaudio.save(target_path, vocals, output_sample_rate)
                        print(f"💾 Saved vocal chunk (fallback): {fname}")

                # cleanup
                del batch_tensor, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"❌ Batch separation failed: {str(e)}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    except KeyboardInterrupt:
        print("\n🛑 Batch vocal separator stopped cleanly.")


if __name__ == "__main__":
    # Default: run once over existing chunks (set watch=True to keep watching)
    os.makedirs("./data/audio_chunks", exist_ok=True)
    os.makedirs("./data/chunked_vocals", exist_ok=True)
    batch_separate_chunks(chunks_dir="./data/audio_chunks", output_dir="./data/chunked_vocals", batch_size=8, watch=False)