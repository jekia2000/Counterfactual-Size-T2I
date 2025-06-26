from huggingface_hub import list_repo_files, hf_hub_download

def download_folder(repo_id, folder_path, save_dir):
    """
    Downloads all files in a specific folder of a Hugging Face repository.

    Args:
        repo_id (str): Repository ID on Hugging Face (e.g., "CaraJ/CoMat_sdxl_ft_unet").
        folder_path (str): Path to the folder in the repository (e.g., "main").
        save_dir (str): Local directory to save the downloaded files.

    Returns:
        None
    """
    import os

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # List all files in the repository
    files = list_repo_files(repo_id)

    # Filter files in the desired folder
    files_to_download = [f for f in files if f.startswith(folder_path)]

    print(f"Found {len(files_to_download)} files to download in '{folder_path}'.")

    # Download each file
    for file in files_to_download:
        print(f"Downloading {file}...")
        local_file = hf_hub_download(repo_id=repo_id, filename=file, local_dir=save_dir)
        print(f"Saved to {local_file}.")

# Example usage
repo_id = "CaraJ/CoMat_sdxl_ft_unet"
folder_path = "sdxl"  # Replace with the specific folder name if needed
save_dir = "."

download_folder(repo_id, folder_path, save_dir)

