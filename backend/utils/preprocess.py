import os
import torch
import torchaudio
import julius
from typing import List, Union
from pathlib import Path
import numpy as np

def chunk_audio(audio_input: Union[str, Path, torch.Tensor], target_sr: int = 16000) -> List[np.ndarray]:
    """
    Unified Preprocessing Engine.
    Accepts EITHER a file path (str/Path) or an in-memory 1D Torch Tensor.
    Resamples, converts to mono, and slices into 30s arrays with a 5s overlap.
    """
    # 1. Handle Input Type Dynamic Resolution
    if isinstance(audio_input, (str, Path)):
        path_str = str(audio_input)
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"❌ Input track missing: {path_str}")
            
        # Load raw audio track from disk
        waveform, sample_rate = torchaudio.load(path_str, backend="soundfile")
        
        # High-fidelity fractional resampling down to Whisper's native 16kHz
        if sample_rate != target_sr:
            waveform = julius.resample_frac(waveform, sample_rate, target_sr)
            
        # Downmix multi-channel arrays natively to Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        waveform = waveform.squeeze(0)  # Convert to a flat 1D tensor
        
    elif isinstance(audio_input, torch.Tensor):
        # Input is already a processing tensor from Demucs
        waveform = audio_input.squeeze()  # Ensure it is a flat 1D line
        
        # Safety check: if Demucs output somehow retained a batch/channel dim
        if waveform.ndim > 1:
            waveform = torch.mean(waveform, dim=0)
    else:
        raise ValueError("❌ Invalid audio_input type. Must be a file path string or a torch.Tensor.")

    # 2. Unified Rolling Window Slicing Architecture
    step_samples = 25 * target_sr     # 25s window stride
    window_samples = 30 * target_sr   # 30s total chunk size width
    total_samples = waveform.shape[0]
    
    audio_chunks = []
    start = 0
    
    while start < total_samples:
        end = start + window_samples
        chunk = waveform[start:end]
        
        # Pad tail chunk seamlessly with silence to keep Whisper tensors structurally uniform
        if chunk.shape[0] < window_samples:
            padding = torch.zeros(window_samples - chunk.shape[0])
            chunk = torch.cat([chunk, padding])
            
        audio_chunks.append(chunk.numpy())
        start += step_samples
        
        # Graceful loop break out when remaining tail data is negligible
        if start >= total_samples - (target_sr * 2):
            break
            
    return audio_chunks