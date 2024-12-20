import os
import shutil
import argparse
import random
from pathlib import Path


def split_dataset(input_dir, test_ratio, val_ratio, output_dir):
    # Ensure reproducibility
    random.seed(42)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all scan and mask files
    scan_files = sorted(input_dir.glob("n_mmni_*.nii.gz"))
    mask_files = sorted(input_dir.glob("mask_n_*.nii.gz"))
    csv_file = input_dir / "list_standardized_tongtong_2017.csv"

    # Verify file consistency
    assert len(scan_files) == len(mask_files), "Mismatch between scans and masks."
    scan_mask_pairs = list(zip(scan_files, mask_files))

    # Shuffle and split data
    random.shuffle(scan_mask_pairs)
    total = len(scan_mask_pairs)
    test_count = int(total * test_ratio)
    val_count = int(total * val_ratio)
    train_count = total - test_count - val_count

    splits = {
        "train": scan_mask_pairs[:train_count],
        "val": scan_mask_pairs[train_count:train_count + val_count],
        "test": scan_mask_pairs[train_count + val_count:]
    }

    # Copy files to output directories
    for split_name, pairs in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Copy CSV file
        shutil.copy(csv_file, split_dir / csv_file.name)

        # Copy scan-mask pairs
        for scan_file, mask_file in pairs:
            shutil.copy(scan_file, split_dir / scan_file.name)
            shutil.copy(mask_file, split_dir / mask_file.name)

    print("Dataset split complete.")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train, val, and test subsets.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory.")
    parser.add_argument("test_ratio", type=float, help="Proportion of data for testing (e.g., 0.2 for 20%).")
    parser.add_argument("val_ratio", type=float, help="Proportion of data for validation (e.g., 0.1 for 10%).")
    parser.add_argument("output_dir", type=str, help="Path to the output directory.")

    args = parser.parse_args()

    # Validate ratios
    if args.test_ratio + args.val_ratio >= 1.0:
        raise ValueError("Test ratio + validation ratio must be less than 1.0.")

    split_dataset(args.input_dir, args.test_ratio, args.val_ratio, args.output_dir)
