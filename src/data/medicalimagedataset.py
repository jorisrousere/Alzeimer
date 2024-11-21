import os
import torch
import nibabel as nib
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MedicalImageDataset(Dataset):
    def __init__(self, folder_path, csv_file):
        """
        Args:
            folder_path (str): Path to the folder containing scan and mask files.
            csv_file (str): Path to the CSV file containing metadata.
        """
        self.folder_path = folder_path
        self.csv_file = pd.read_csv(csv_file, header=None)
        self.data_info = self._get_scan_info()

    def _get_scan_info(self):
        """
        Extracts information about valid scans based on mask availability and status.
        """
        info_list = []
        for _, row in self.csv_file.iterrows():
            scan_id = row[0]  # ID from the CSV file
            status = row[4]   # CN, AD, or MCI
            mci_type = row[5] # Stability of MCI (3 for stable, 4 for not stable)

            # Define paths for scan and mask files
            scan_file = os.path.join(self.folder_path, f"n_mmni_fADNI_{scan_id}_1.5T_t1w.nii.gz")
            mask_file = os.path.join(self.folder_path, f"mask_n_mmni_fADNI_{scan_id}_1.5T_t1w.nii.gz")
            
            # Determine the class label
            if status == "CN":
                class_label = 0
            elif status == "AD":
                class_label = 1
            elif status == "MCI":
                class_label = 2 if mci_type == 3 else 3
            else:
                continue  # Skip invalid statuses
            
            # Append valid scan-mask pairs
            if os.path.exists(scan_file) and os.path.exists(mask_file):
                info_list.append((scan_file, mask_file, class_label))
        
        return info_list

    def __len__(self):
        """
        Returns the total number of frames (slices) in the dataset.
        """
        return sum(nib.load(scan_file).shape[2] for scan_file, _, _ in self.data_info)

    def __getitem__(self, idx):
        """
        Loads and returns a specific scan and mask slice pair.
        """
        # Map the global index to a specific scan and slice
        cumulative_slices = 0
        for scan_file, mask_file, class_label in self.data_info:
            scan = nib.load(scan_file).get_fdata()
            num_slices = scan.shape[2]
            if cumulative_slices + num_slices > idx:
                slice_idx = idx - cumulative_slices
                break
            cumulative_slices += num_slices
        
        # Load the corresponding slice for scan and mask
        mask = nib.load(mask_file).get_fdata()
        scan_slice = scan[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx]
        
        # Filter mask based on the class label
        mask_slice = np.where(mask_slice > 0, class_label, 0).astype(np.uint8)

        # Normalize the scan slice to [0, 1]
        scan_slice = (scan_slice - np.min(scan_slice)) / (np.max(scan_slice) - np.min(scan_slice))

        # Convert slices to PyTorch tensors
        scan_tensor = torch.tensor(scan_slice, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        mask_tensor = torch.tensor(mask_slice, dtype=torch.long)  # Class label format

        return scan_tensor, mask_tensor

# Function to create a DataLoader
def create_dataloader(folder_path, csv_filename, batch_size=4, shuffle=True):
    dataset = MedicalImageDataset(folder_path, csv_filename)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Example usage
if __name__ == "__main__":
    folder_path = '/mnt/project/computer_vision/datasets/adni1-samples'
    csv_filename = os.path.join(folder_path, 'list_standardized_tongtong_2017.csv')

    dataloader = create_dataloader(folder_path, csv_filename, batch_size=2)
    # Check a sample batch
    for scans, masks in dataloader:
        print("Scans shape:", scans.shape)  # Expected shape: (batch_size, 1, H, W)
        print("Masks shape:", masks.shape)  # Expected shape: (batch_size, H, W)
