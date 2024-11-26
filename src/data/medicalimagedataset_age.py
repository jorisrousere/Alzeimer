import os
import torch
import nibabel as nib
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F

class MedicalImageDataset(Dataset):
    def __init__(self, folder_path, csv_file, target_size=(256, 256)):
        """
        Args:
            folder_path (str): Path to the folder containing scan and mask files.
            csv_file (str): Path to the CSV file containing metadata.
            target_size (tuple): Desired output size (height, width).
        """
        self.folder_path = folder_path
        self.csv_file = pd.read_csv(csv_file, header=None)
        self.target_size = target_size
        self.data_info = self._get_scan_info()
        self.slice_indices = self._generate_slice_indices()

    def _get_scan_info(self):
        """
        Extracts information about valid scans based on mask availability and status.
        """
        info_list = []
        for _, row in self.csv_file.iterrows():
            scan_id = row[0]  # ID from the CSV file
            age = int(row[2]) 
            status = row[4]   # CN, AD, or MCI
            mci_type = row[5] # Stability of MCI (3 for stable, 4 for not stable)

            # Define paths for scan and mask files
            scan_file = os.path.join(self.folder_path, f"n_mmni_fADNI_{scan_id}_1.5T_t1w.nii.gz")
            mask_file = os.path.join(self.folder_path, f"mask_n_mmni_fADNI_{scan_id}_1.5T_t1w.nii.gz")
            
            # Determine the class label
            if status == "CN":
                class_label = age - 49
                # Append valid scan-mask pairs
                if os.path.exists(scan_file) and os.path.exists(mask_file):
                    info_list.append((scan_file, mask_file, class_label))
            else:
                continue  # Skip invalid statuses
        
        return info_list

    def _generate_slice_indices(self, threshold=0.15):
        """
        Pre-compute indices mapping each global slice to its scan, plane, and slice index,
        excluding slices where the mask has less than a given proportion of `1`s.
        
        Args:
            threshold (float): Minimum proportion of `1`s required to include a slice.
        """
        indices = []
        for scan_idx, (scan_file, mask_file, _) in enumerate(self.data_info):
            scan_shape = nib.load(scan_file).shape
            mask = nib.load(mask_file).get_fdata()

            planes = {
                "axial": scan_shape[2],
                "sagittal": scan_shape[0],
                "coronal": scan_shape[1],
            }
            for plane, num_slices in planes.items():
                for slice_idx in range(num_slices):
                    # Extract mask slice
                    if plane == "axial":
                        mask_slice = mask[:, :, slice_idx]
                    elif plane == "sagittal":
                        mask_slice = mask[slice_idx, :, :]
                    elif plane == "coronal":
                        mask_slice = mask[:, slice_idx, :]

                    # Calculate the proportion of `1`s in the mask slice
                    total_pixels = mask_slice.size
                    proportion_of_ones = np.sum(mask_slice == 1) / total_pixels

                    # Check if the proportion meets the threshold
                    if proportion_of_ones >= threshold:
                        indices.append((scan_idx, plane, slice_idx))

        return indices

    def __len__(self):
        """
        Returns the total number of slices across all planes for all scans.
        """
        return len(self.slice_indices)

    def __getitem__(self, idx):
        """
        Loads and returns a specific slice based on its global index.
        """
        scan_idx, plane, slice_idx = self.slice_indices[idx]
        scan_file, mask_file, class_label = self.data_info[scan_idx]

        # Load scan and mask
        scan = nib.load(scan_file).get_fdata()
        mask = nib.load(mask_file).get_fdata()

        # Extract slice based on plane
        if plane == "axial":
            scan_slice = scan[:, :, slice_idx]
            mask_slice = mask[:, :, slice_idx]
        elif plane == "sagittal":
            scan_slice = scan[slice_idx, :, :]
            mask_slice = mask[slice_idx, :, :]
        elif plane == "coronal":
            scan_slice = scan[:, slice_idx, :]
            mask_slice = mask[:, slice_idx, :]

        # Filter mask based on the class label
        mask_slice = np.where(mask_slice > 0, class_label, 0).astype(np.uint8)
        
        # Normalize the scan slice to [0, 1]
        epsilon = 1e-8  # Small value to prevent division by zero
        scan_slice = (scan_slice - np.min(scan_slice)) / (np.max(scan_slice) - np.min(scan_slice) + epsilon)


        # Convert slices to PyTorch tensors
        scan_tensor = torch.tensor(scan_slice, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        mask_tensor = torch.tensor(mask_slice, dtype=torch.long)  # Class label format

        # Resize scan and mask to the target size
        scan_tensor_resized = F.resize(scan_tensor, self.target_size)
        mask_tensor_resized = F.resize(mask_tensor.unsqueeze(0), self.target_size, interpolation=F.InterpolationMode.NEAREST).squeeze(0)

        return scan_tensor_resized, mask_tensor_resized
