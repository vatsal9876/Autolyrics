import torch
import torchaudio
import julius

def slice_tensor_to_arrays(waveform: torch.Tensor, sample_rate: int, target_sr: int = 16000, chunk_duration_sec: int = 3):
    """Resamples, downmixes to mono, and slices an in-memory tensor into uniform numpy chunks."""
    # 1. Standardize sampling frequency in memory
    if sample_rate != target_sr:
        waveform = julius.resample_frac(waveform, sample_rate, target_sr)
        
    # 2. Flatten stereo structures down to a Mono channel map
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    waveform = waveform.squeeze(0)
    
    # 3. Window slicing logic
    samples_per_chunk = chunk_duration_sec * target_sr
    total_samples = waveform.shape[0]
    audio_chunks = []
    
    for start in range(0, total_samples, samples_per_chunk):
        end = start + samples_per_chunk
        if end > total_samples:
            break  # Discard uneven trail frames
        
        chunk = waveform[start:end].numpy()
        audio_chunks.append(chunk)
        
    return audio_chunks