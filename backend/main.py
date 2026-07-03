import os
import shutil
import gc
import torch
from pathlib import Path
import sys
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import your decoupled, functional pipeline components
from backend.src.pipelines.baseline import run_baseline_pipeline
from backend.src.pipelines.finetuned_separated_pipeline import run_finetuned_separated_pipeline

app = FastAPI(
    title="OpsTwin Comparative ASR Engine",
    description="Processes any incoming audio through BOTH baseline and fine-tuned pipelines simultaneously in RAM.",
    version="2.0.0"
)

# 🚨 Configuration Constants
MODEL_BASE_DIR = "./backend/models"
DEFAULT_ADAPTER_DIR = os.path.join(MODEL_BASE_DIR, "whisper_lyrics_400steps_20260702_145100", "checkpoint-150")

class DualPipelineResponse(BaseModel):
    filename: str
    baseline_transcript: str
    custom_transcript: str

def global_vram_cleanup():
    """Forcibly flush torch variables and clean fragmentations from the RTX 3050's VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@app.on_event("startup")
def startup_check():
    print(f"🖥️ Target Hardware Baseline: {'CUDA (RTX 3050)' if torch.cuda.is_available() else 'CPU'}")
    global_vram_cleanup()



# ... rest of your code ...

@app.post("/transcribe/compare", response_model=DualPipelineResponse, status_code=200)
async def transcribe_and_compare(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Raw audio track file (.mp3, .wav, .flac)"),
    adapter_path: str = Form(DEFAULT_ADAPTER_DIR, description="Path to the custom QLoRA adapter checkpoint")):
    
    """
    **Dual Pipeline Endpoint:** Accepts a binary audio file upload and runs BOTH pipeline models sequentially.
    """
    # Create unique isolated filename framework safely
    temp_file_path = f"temp_{file.filename}"
    
    try:
        print(f"📥 Receiving stream: {file.filename}")
        # Save incoming file bytes directly to temporary path block
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 🎯 RUN PIPELINE 1: Baseline Whisper Medium
        print("\n=== 🔵 Running Pipeline A: Baseline Whisper Medium ===")
        baseline_text = run_baseline_pipeline(audio_path=temp_file_path)
        
        global_vram_cleanup()

        # 🎯 RUN PIPELINE 2: Demucs Vocal Separation + Fine-Tuned Adapters
        print("\n=== 🟢 Running Pipeline B: Demucs + Fine-Tuned Whisper ===")
        custom_text = run_finetuned_separated_pipeline(
            audio_path=temp_file_path,
            adapter_path=adapter_path
        )
        
        return DualPipelineResponse(
            filename=file.filename,
            baseline_transcript=baseline_text,
            custom_transcript=custom_text
        )

    except Exception as e:
        print(f"❌ Critical Pipeline Failure: {e}")
        raise HTTPException(status_code=500, detail=f"Comparative Pipeline Failure: {str(e)}")

    finally:
        # File management safety layers
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        background_tasks.add_task(global_vram_cleanup)

@app.get("/health", status_code=200)
async def health_check():
    vram_allocated_mb = 0
    if torch.cuda.is_available():
        vram_allocated_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
    return {
        "status": "healthy",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "current_vram_usage_mb": f"{vram_allocated_mb:.2f} MB"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)