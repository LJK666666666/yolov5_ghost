#!/usr/bin/env python3
"""
Script to prepare the railroad worker detection dataset for YOLOv5 training.

Reorganizes the dataset into train/valid/test splits with proper directory structure.
"""

import random
import shutil
from pathlib import Path

import yaml


def prepare_railroad_dataset():
    """Prepare the railroad worker detection dataset for YOLOv5 training."""
    # Source paths
    source_dir = Path("data/railroad-worker-detection/dataset")
    imgs_dir = source_dir / "imgs"
    txt_dir = source_dir / "txt"

    # Target directory
    target_dir = Path("data/railroad-worker-detection")

    # Create target directory structure
    for split in ["train", "valid", "test"]:
        (target_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (target_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = list(imgs_dir.glob("*.jpg"))
    print(f"Found {len(image_files)} images")

    # Filter images that have corresponding label files
    valid_images = []
    for img_file in image_files:
        label_file = txt_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            valid_images.append(img_file)

    print(f"Found {len(valid_images)} images with corresponding labels")

    # Shuffle the list for random split
    random.seed(42)  # For reproducible results
    random.shuffle(valid_images)

    # Split ratios: 70% train, 20% valid, 10% test
    total = len(valid_images)
    train_end = int(0.7 * total)
    valid_end = int(0.9 * total)

    train_images = valid_images[:train_end]
    valid_images_split = valid_images[train_end:valid_end]
    test_images = valid_images[valid_end:]

    print(f"Split: {len(train_images)} train, {len(valid_images_split)} valid, {len(test_images)} test")

    # Copy files to appropriate directories
    def copy_files(image_list, split_name):
        for img_file in image_list:
            # Copy image
            target_img = target_dir / split_name / "images" / img_file.name
            shutil.copy2(img_file, target_img)

            # Copy label
            label_file = txt_dir / f"{img_file.stem}.txt"
            target_label = target_dir / split_name / "labels" / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, target_label)

    copy_files(train_images, "train")
    copy_files(valid_images_split, "valid")
    copy_files(test_images, "test")

    # Create data.yaml configuration file
    data_config = {
        "path": "./",  # dataset root dir
        "train": "data/railroad-worker-detection/train/images",
        "val": "data/railroad-worker-detection/valid/images",
        "test": "data/railroad-worker-detection/test/images",
        "nc": 3,  # number of classes
        "names": ["person", "head", "safety_vest"],  # class names (guessed based on railroad worker detection)
    }

    # Save data.yaml
    with open(target_dir / "data.yaml", "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print("Dataset prepared successfully!")
    print(f"Configuration saved to: {target_dir / 'data.yaml'}")
    print(f"Train: {len(train_images)} images")
    print(f"Valid: {len(valid_images_split)} images")
    print(f"Test: {len(test_images)} images")


if __name__ == "__main__":
    prepare_railroad_dataset()
