import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model

class VocalSeparator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = get_model("htdemucs")
        self.model.to(self.device)
        self.model.eval()

    def isolate_vocals_in_memory(self, mix_tensor: torch.Tensor, sr: int) -> torch.Tensor:
        """Processes an audio tensor on the GPU and returns the clean vocal stem tensor."""
        # Force stereo structure for Demucs if input is mono
        if mix_tensor.shape[0] == 1:
            mix_tensor = mix_tensor.repeat(2, 1)
            
        mix_device = mix_tensor.to(self.device)
        
        with torch.no_grad():
            # Demucs expects batch dimension: [B, Channels, Samples]
            sources = apply_model(self.model, mix_device.unsqueeze(0), shifts=1)[0]
            
        # Index 3 extracts the clean vocal channel maps, move back to CPU memory
        vocals = sources[3].cpu()
        
        del mix_device, sources
        return vocals