import os
import torch
import nibabel as nib
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class MCIHippocampusDataset(Dataset):
    def __init__(self, folder_path, csv_file, target_size=(256, 256), threshold=0.15):
        """
        Dataset pour les slices contenant les hippocampes uniquement pour les patients MCI.

        Args:
            folder_path (str): Path to the folder containing scan files.
            csv_file (str): Path to the CSV file containing metadata.
            target_size (tuple): Desired output size (height, width).
            threshold (float): Minimum proportion of hippocampus required to include a slice.
        """
        self.folder_path = folder_path
        self.csv_file = pd.read_csv(csv_file, header=None)
        self.target_size = target_size
        self.threshold = threshold
        self.data_info = self._get_scan_info()
        self.slice_indices = self._generate_slice_indices()

    def _get_scan_info(self):
        """
        Extracts information about valid MCI scans and their stability (3, 5 = Stable, 4 = Unstable).
        """
        info_list = []
        skipped_patients = 0
        loaded_patients = 0

        for _, row in self.csv_file.iterrows():
            scan_id = row[0]  # ID from the CSV file
            status = row[4]   # MCI
            mci_type = row[5] # Stability of MCI (3 and 5 for stable, 4 for unstable)

            # Include only MCI patients
            if status == "MCI" and mci_type in [3, 4, 5]:
                # Binary label: 0 for stable (3, 5), 1 for unstable (4)
                class_label = 0 if mci_type in [3, 5] else 1

                # Scan file path
                scan_file = os.path.join(self.folder_path, f"n_mmni_fADNI_{scan_id}_1.5T_t1w.nii.gz")
                if os.path.exists(scan_file):
                    info_list.append((scan_file, class_label, scan_id))
                    loaded_patients += 1
                else:
                    print(f"Warning: Scan file {scan_file} not found. Skipping patient {scan_id}.")
                    skipped_patients += 1

        print(f"Loaded patients: {loaded_patients}, Skipped patients: {skipped_patients}")
        return info_list

    def _generate_slice_indices(self):
        """
        Pre-compute indices mapping each valid slice to its scan,
        excluding slices without sufficient hippocampus content.
        """
        indices = []
        for scan_idx, (scan_file, _, scan_id) in enumerate(self.data_info):
            scan = nib.load(scan_file).get_fdata()

            for slice_idx in range(scan.shape[2]):  # Only axial slices
                # Extract hippocampus regions
                region_1 = scan[40:80, 90:130, slice_idx]
                region_2 = scan[100:140, 90:130, slice_idx]

                # Calculate the proportion of non-zero voxels in hippocampus regions
                proportion_hippo = (
                    np.sum(region_1 > 0) + np.sum(region_2 > 0)
                ) / (region_1.size + region_2.size)

                if proportion_hippo >= self.threshold:
                    indices.append((scan_idx, slice_idx))

        return indices

    def __len__(self):
        """
        Returns the total number of valid slices in the dataset.
        """
        return len(self.slice_indices)

    def __getitem__(self, idx):
        """
        Loads and returns a specific slice based on its global index.
        """
        scan_idx, slice_idx = self.slice_indices[idx]
        scan_file, class_label, patient_id = self.data_info[scan_idx]

        # Load scan
        scan = nib.load(scan_file).get_fdata()

        # Extract the hippocampus regions from the slice
        region_1 = scan[40:80, 90:130, slice_idx]  # Shape: (40, 40)
        region_2 = scan[100:140, 90:130, slice_idx]  # Shape: (40, 40)

        # Combine the two hippocampus regions into one image of shape (80, 40)
        combined_region = np.zeros((80, 40))
        combined_region[:40, :] = region_1  # Top half
        combined_region[40:, :] = region_2  # Bottom half

        # Normalize the combined region to [0, 1]
        epsilon = 1e-8  # Small value to prevent division by zero
        combined_region = (combined_region - np.min(combined_region)) / (np.max(combined_region) - np.min(combined_region) + epsilon)

        # Convert to PyTorch tensor and resize
        scan_tensor = torch.tensor(combined_region, dtype=torch.float32).unsqueeze(0)
        scan_tensor_resized = F.resize(scan_tensor, self.target_size)

        return scan_tensor_resized, class_label, patient_id
