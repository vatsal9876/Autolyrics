import os
from huggingface_hub import HfApi, hf_hub_download

def download_jamendo_subset():
    repo_id = "jamendolyrics/jam-alt-lines"
    local_dir = "./data/jam_alt_lines"
    
    api = HfApi()
    
    print("Scanning Hugging Face Repository for layout tracking...")
    try:
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as e:
        print(f"Error accessing repository: {e}")
        return

    # Filter strictly for the English subset under the 'pure' folder split
    target_prefix = "pure/en/"
    target_files = [f for f in repo_files if f.startswith(target_prefix)]
    
    if not target_files:
        print(f"No files found matching prefix '{target_prefix}'. Double check repo layout.")
        return
        
    print(f"Found {len(target_files)} files to download (Metadata + Audio clips).")
    
    for file_name in target_files:
        # Replicate the exact folder structure locally inside your workspace
        local_file_path = os.path.join(local_dir, file_name)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        print(f"Downloading: {file_name} -> {local_file_path}")
        hf_hub_download(
            repo_id=repo_id,
            filename=file_name,
            repo_type="dataset",
            local_dir=local_dir
        )
        
    print(f"\nData successfully synced and structured at: {os.path.abspath(local_dir)}")

if __name__ == "__main__":
    download_jamendo_subset()