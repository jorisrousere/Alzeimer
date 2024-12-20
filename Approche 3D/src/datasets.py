# datasets.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import nibabel as nib

class HippocampusDataset(Dataset):
    def __init__(self, csv_file, split, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data["File"].str.contains(f"/{split}/")]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data.iloc[idx]["File"] # get the file path for the image
        label = self.data.iloc[idx]["labels_cnn"] # get the corresponding label
        img = nib.load(file_path).get_fdata() # load the 3D image using nibabel

        
        img = np.expand_dims(img, axis=0)
        img = (img - np.mean(img)) / (np.std(img) + 1e-8) # normalize the image by mean and standard deviation

        if self.transform:
            img = self.transform(img)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.float32) # convert the image and label to PyTorch tensors
