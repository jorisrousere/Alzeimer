import os
import torch
import nibabel as nib
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision import transforms

class MedicalClassDataset(Dataset):
    def __init__(self, folder_path, csv_file, target_size=(256, 256), num_slices=3, stride=1, ranges={"axial": (40, 80), "sagittal": (40, 140), "coronal": (90, 130)}, apply_augmentation=False):
        """
        Args:
            folder_path (str): Path to the folder containing scan and mask files.
            csv_file (str): Path to the CSV file containing metadata.
            target_size (tuple): Desired output size (height, width).
            num_slices (int): Number of consecutive slices to concatenate as channels.
            stride (int): Step size between slices.
            ranges (dict): Dictionary defining min and max slice indices for each plane.
            apply_augmentation (bool): Flag to apply augmentation (contrast, brightness).
        """
        self.folder_path = folder_path
        self.csv_file = pd.read_csv(csv_file, header=None)
        self.target_size = target_size
        self.num_slices = num_slices
        self.stride = stride
        self.ranges = ranges or {}
        self.apply_augmentation = apply_augmentation
        self.data_info = self._get_scan_info()
        self.slice_indices = self._generate_slice_indices()
        
        # Define augmentation transformations if needed
        self.augmentation_transform = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5)], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(contrast=0.5)], p=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5)

        ]) if self.apply_augmentation else None

    def _get_scan_info(self):
        """Extracts information about valid scans based on mask availability and status."""
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
            # elif status == "MCI":
            #     class_label = 2 if (mci_type == 3 or mci_type == 5) else 1
            else:
                continue  # Skip invalid statuses
            
            # Append valid scan-mask pairs
            if os.path.exists(scan_file) and os.path.exists(mask_file):
                info_list.append((scan_file, mask_file, class_label))
        
        return info_list

    def _generate_slice_indices(self, threshold=0):
        """
        Pre-compute indices mapping each global slice to its scan, plane, and slice index.
        Exclude slices where the mask has less than a given proportion of `1`s.
        
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
                min_idx, max_idx = self.ranges.get(plane, (0, num_slices))
                min_idx = max(0, min_idx)
                max_idx = min(num_slices, max_idx)
                
                for slice_idx in range(min_idx + (self.num_slices // 2), (max_idx - (self.num_slices // 2)), self.stride):
                    valid = True
                    for offset in range(-self.num_slices // 2, self.num_slices // 2):
                        current_idx = slice_idx + offset
                        # Extract mask slice
                        if plane == "axial":
                            mask_slice = mask[:, :, current_idx]
                        elif plane == "sagittal":
                            mask_slice = mask[current_idx, :, :]
                        elif plane == "coronal":
                            mask_slice = mask[:, current_idx, :]

                        # Check the proportion of `1`s in the mask slice
                        total_pixels = mask_slice.size
                        proportion_of_ones = np.sum(mask_slice == 1) / total_pixels

                        if proportion_of_ones < threshold:
                            valid = False
                            break

                    if valid:
                        indices.append((scan_idx, plane, slice_idx))

        return indices

    def __len__(self):
        """Returns the total number of slices across all planes for all scans."""
        return len(self.slice_indices)

    def __getitem__(self, idx):
        """Loads and returns a specific slice based on its global index."""
        scan_idx, plane, slice_idx = self.slice_indices[idx]
        scan_file, _, class_label = self.data_info[scan_idx]

        # Load scan and mask
        scan = nib.load(scan_file).get_fdata()

        scan_slices = []
        for offset in range(-self.num_slices // 2, self.num_slices // 2):
            current_idx = slice_idx + offset

            # Extract slices based on plane
            if plane == "axial":
                scan_slice = scan[:, :, current_idx]
            elif plane == "sagittal":
                scan_slice = scan[current_idx, :, :]
            elif plane == "coronal":
                scan_slice = scan[:, current_idx, :]

            # Normalize the scan slice to [0, 1]
            epsilon = 1e-8  # Small value to prevent division by zero
            scan_slice = (scan_slice - np.min(scan_slice)) / (np.max(scan_slice) - np.min(scan_slice) + epsilon)

            scan_slices.append(torch.tensor(scan_slice, dtype=torch.float32))

        # Stack slices along the channel dimension
        scan_tensor = torch.stack(scan_slices, dim=0)

        # Resize scan and mask to the target size
        scan_tensor_resized = F.resize(scan_tensor, self.target_size)

        # Apply augmentation (contrast, brightness) if specified
        if self.apply_augmentation and self.augmentation_transform:
            scan_tensor_resized = self.augmentation_transform(scan_tensor_resized)

        return scan_tensor_resized, class_label
