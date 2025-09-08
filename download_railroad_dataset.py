import json
import os
import shutil

import kagglehub


def setup_kaggle_credentials():
    """Set up Kaggle API credentials."""
    # Create .kaggle directory in home directory if it doesn't exist
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    # Create kaggle.json with API credentials
    credentials = {"username": "ljk666666", "key": "0dbfbabd3b50fe1df77e77921224ac36"}

    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    with open(kaggle_json_path, "w") as f:
        json.dump(credentials, f)

    # Set proper permissions (readable only by user)
    os.chmod(kaggle_json_path, 0o600)

    print(f"Kaggle credentials saved to: {kaggle_json_path}")


def download_railroad_dataset():
    """Download the railroad worker detection dataset."""
    try:
        # Set up credentials
        setup_kaggle_credentials()

        # Download the dataset
        print("Downloading railroad worker detection dataset...")
        path = kagglehub.dataset_download("mikhailma/railroad-worker-detection-dataset")
        print(f"Dataset downloaded to: {path}")

        # Define target directory in the data folder
        data_dir = os.path.join(os.getcwd(), "data")
        target_dir = os.path.join(data_dir, "railroad-worker-detection")

        # Create target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)

        # Copy files to data directory
        if os.path.exists(path):
            print(f"Copying dataset from {path} to {target_dir}")
            # Copy all files from download path to target directory
            for item in os.listdir(path):
                source_path = os.path.join(path, item)
                target_path = os.path.join(target_dir, item)

                if os.path.isdir(source_path):
                    if os.path.exists(target_path):
                        shutil.rmtree(target_path)
                    shutil.copytree(source_path, target_path)
                else:
                    shutil.copy2(source_path, target_path)

            print(f"Dataset successfully copied to: {target_dir}")

            # List contents of the target directory
            print("\nDataset contents:")
            for root, dirs, files in os.walk(target_dir):
                level = root.replace(target_dir, "").count(os.sep)
                indent = " " * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = " " * 2 * (level + 1)
                for file in files:
                    print(f"{subindent}{file}")

        return target_dir

    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return None


if __name__ == "__main__":
    result_path = download_railroad_dataset()
    if result_path:
        print(f"\nPath to dataset files: {result_path}")
    else:
        print("Failed to download dataset")
