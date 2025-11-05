#!/usr/bin/env python3
"""
Script to download and prepare Tiny-ImageNet dataset
"""
import os
import sys
import subprocess
import zipfile
from dataset.tiny_imagenet import prepare_tiny_imagenet_val


def download_dataset():
    """Download Tiny-ImageNet dataset"""
    dataset_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_file = "tiny-imagenet-200.zip"
    extract_dir = "tiny-imagenet"
    
    # Check if already downloaded
    if os.path.exists(os.path.join(extract_dir, "tiny-imagenet-200")):
        print("Dataset already exists. Skipping download.")
        return True
    
    print("Downloading Tiny-ImageNet dataset...")
    print(f"URL: {dataset_url}")
    
    try:
        # Download using wget or curl
        if subprocess.call(["which", "wget"], stdout=subprocess.DEVNULL) == 0:
            subprocess.run(["wget", dataset_url], check=True)
        elif subprocess.call(["which", "curl"], stdout=subprocess.DEVNULL) == 0:
            subprocess.run(["curl", "-O", dataset_url], check=True)
        else:
            print("Error: wget or curl not found. Please install one of them.")
            return False
        
        # Extract
        print("Extracting dataset...")
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Clean up zip file
        os.remove(zip_file)
        print("Dataset downloaded and extracted successfully!")
        
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


def main():
    print("=" * 80)
    print("Tiny-ImageNet Dataset Preparation")
    print("=" * 80)
    
    # Download dataset
    if not download_dataset():
        print("Failed to download dataset.")
        sys.exit(1)
    
    # Prepare validation split
    print("\nPreparing validation dataset...")
    try:
        prepare_tiny_imagenet_val()
        print("\nDataset preparation completed successfully!")
        print("\nYou can now run training with:")
        print("  python train.py --num_epochs 10 --batch_size 32")
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
