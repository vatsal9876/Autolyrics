import os
import glob
import time
import subprocess

def run_isolated_gpu_loop(raw_songs_dir="./data/raw_songs", output_dir="./data/separated"):
    print("🚀 Launching Isolated GPU Manager...")
    print("👀 Watching for tracks... (Press Ctrl+C to stop cleanly)")
    
    # Dynamically build the absolute path to the worker script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    worker_script_path = os.path.join(current_script_dir, "separate_single.py")
    
    try:
        while True:
            wav_files = glob.glob(os.path.join(raw_songs_dir, "*.wav"))
            processed_any = False
            
            for track_path in wav_files:
                track_id = os.path.splitext(os.path.basename(track_path))[0]
                target_vocal_path = os.path.join(output_dir, track_id, "vocals.wav")
                
                if os.path.exists(target_vocal_path):
                    continue
                
                print(f"\n🎵 [Isolated Process] Splitting: {track_id}")
                
                # Execute the worker in a distinct OS process boundary
                cmd = ["python3", worker_script_path, track_path]
                subprocess.run(cmd)
                
                processed_any = True
            
            # Cool-down periods to keep idle resource usage near 0%
            if not processed_any:
                time.sleep(10)
            else:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n🛑 GPU Manager stopped cleanly.")

if __name__ == "__main__":
    run_isolated_gpu_loop()