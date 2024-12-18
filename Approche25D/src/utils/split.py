import os
import shutil
import argparse
import random
from pathlib import Path
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

def get_classes_from_csv(csv_file, folder_path):
    """
    Extracts class labels from the CSV file using scan_id, status, and mci_type.
    Assumes the class labels are determined by 'status' and 'mci_type'.
    Skips missing scan/mask files.
    """
    class_labels = []
    
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Iterate through each row to extract the class label
    for _, row in df.iterrows():
        scan_id = row.iloc[0]  # ID from the CSV file (use .iloc for position)
        status = row.iloc[4]   # CN, AD, or MCI
        mci_type = row.iloc[5] # Stability of MCI (3 for stable, 4 for not stable)

        # Define paths for scan and mask files
        scan_file = os.path.join(folder_path, f"n_mmni_fADNI_{scan_id}_1.5T_t1w.nii.gz")
        mask_file = os.path.join(folder_path, f"mask_n_mmni_fADNI_{scan_id}_1.5T_t1w.nii.gz")
        
        # Check if both scan and mask files exist
        if os.path.exists(scan_file) and os.path.exists(mask_file):
            # Determine the class label
            if status == "CN":
                class_label = 0
            elif status == "AD":
                class_label = 1
            elif status == "MCI":
                class_label = 2 if mci_type == 3 else 3
            
            class_labels.append((scan_file, mask_file, class_label))
        else:
            print(f"Warning: Missing files for scan_id {scan_id} (Scan or Mask file missing).")

    return class_labels

def split_dataset(input_dir, test_ratio, val_ratio, output_dir):
    # Ensure reproducibility
    random.seed(42)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV file path
    csv_file = input_dir / "list_standardized_tongtong_2017.csv"

    # Get scan-mask pairs and class labels
    class_labels = get_classes_from_csv(csv_file, input_dir)

    # Create a dictionary to hold the scan-masks per class
    class_dict = defaultdict(list)
    for scan_file, mask_file, class_label in class_labels:
        class_dict[class_label].append((scan_file, mask_file))

    # Perform stratified split by class
    train_pairs = []
    val_pairs = []
    test_pairs = []

    for class_label, pairs in class_dict.items():
        # Split each class separately
        train, test_val = train_test_split(pairs, test_size=test_ratio + val_ratio, stratify=[class_label] * len(pairs))
        val, test = train_test_split(test_val, test_size=test_ratio / (test_ratio + val_ratio), stratify=[class_label] * len(test_val))

        # Add to the final lists
        train_pairs.extend(train)
        val_pairs.extend(val)
        test_pairs.extend(test)

    # Shuffle to mix the data after stratification
    random.shuffle(train_pairs)
    random.shuffle(val_pairs)
    random.shuffle(test_pairs)

    splits = {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs
    }

    # Copy files to output directories
    for split_name, pairs in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Copy CSV file
        shutil.copy(csv_file, split_dir / csv_file.name)

        # Copy scan-mask pairs
        for scan_file, mask_file in pairs:
            shutil.copy(scan_file, split_dir / Path(scan_file).name)  # Use Path().name to get the filename
            shutil.copy(mask_file, split_dir / Path(mask_file).name)  # Use Path().name to get the filename

    print("Dataset split complete.")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train, val, and test subsets with balanced classes.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory.")
    parser.add_argument("test_ratio", type=float, help="Proportion of data for testing (e.g., 0.2 for 20%).")
    parser.add_argument("val_ratio", type=float, help="Proportion of data for validation (e.g., 0.1 for 10%).")
    parser.add_argument("output_dir", type=str, help="Path to the output directory.")

    args = parser.parse_args()

    # Validate ratios
    if args.test_ratio + args.val_ratio >= 1.0:
        raise ValueError("Test ratio + validation ratio must be less than 1.0.")

    split_dataset(args.input_dir, args.test_ratio, args.val_ratio, args.output_dir)
