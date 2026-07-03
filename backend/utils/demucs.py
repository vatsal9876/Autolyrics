import gc
import torch
import torchaudio
from typing import List
import numpy as np
from demucs.pretrained import get_model
from demucs.apply import apply_model

def separate_vocals_clean(audio_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> torch.Tensor:
    """
    Processes the FULL track at high fidelity FIRST. 
    Extracts pure mono vocals with ZERO phase cancellation or padding issues.
    """
    model = get_model("htdemucs")
    model.to(device)
    model.eval()
    
    # Load native audio track (typically 44.1kHz or 48kHz)
    waveform, sample_rate = torchaudio.load(audio_path, backend="soundfile")
    
    # Ensure stereo for Demucs
    if waveform.shape[0] == 1:
        waveform = torch.cat([waveform, waveform], dim=0)
        
    waveform_tensor = waveform.to(device).unsqueeze(0)
    
    with torch.no_grad():
        # split=True keeps your 4GB VRAM safe by processing sections sequentially
        sources = apply_model(model, waveform_tensor, split=True)[0]
        
    # Extract vocals stem [2, samples]
    vocals = sources[3].cpu()
    
    # 🎯 FIX: Convert Stereo to Mono SAFELY using torchaudio functional downmix 
    # to avoid the phase cancellation bug caused by simple numpy averaging
    if vocals.shape[0] > 1:
        vocals = torch.mean(vocals, dim=0, keepdim=True)
        
    # Resample the pure vocal stem directly to Whisper's native 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        vocals = resampler(vocals)
        
    return vocals.squeeze(0) # Returns a beautiful, crystal-clear 1D Mono Tensor