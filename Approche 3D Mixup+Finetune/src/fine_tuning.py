import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import nibabel as nib
import numpy as np
import torch
import re
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize
import h5py
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from model import Fixed4Layer3DCNN, GradCAM3D, ResidualBlock3D, ResNet3D
from dataset import HippocampusDataset3D, TestHippocampusDataset3D, HDF5Dataset, create_mixup_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
from base import evaluate_model, transform, device, train_model
from utils import EarlyStopping
from torch.utils.data import random_split
import random
from torch.utils.data import Subset
from test import evaluate_test
import argparse
from dotenv import load_dotenv

load_dotenv()
def main(args):
    data_dir = os.getenv('DATA_DIR')  
    csv_path = os.getenv('CSV_PATH')  
    dataset = HippocampusDataset3D(
        data_dir=data_dir,
        csv_path=csv_path,
        transform=transform
    )
    seed = 42
    generator = torch.Generator().manual_seed(seed)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    dataset80, dataset20 = random_split(dataset, [train_size, val_size])

    # create_mixup_dataset(dataset80, 'mixup80.h5', 800, device=device)
    # create_mixup_dataset(dataset20, 'mixup20.h5', 200, device=device)

    file_path80 = 'mixup80.h5'
    file_path20 = 'mixup20.h5'
    mixup_dataset_80 = HDF5Dataset(file_path80)
    mixup_dataset_20 = HDF5Dataset(file_path20)
    combined_dataset_80 = torch.utils.data.ConcatDataset([dataset80, mixup_dataset_80])
    combined_dataset_20 = torch.utils.data.ConcatDataset([dataset20, mixup_dataset_20])

    mci_dataset = TestHippocampusDataset3D(
        data_dir=data_dir,
        csv_path=csv_path,
        transform=transform
    )

    mci_size = len(mci_dataset)

    # 50% des mci dans l'entrainement
    random.seed(seed)
    subset_size = int(0.5 * mci_size)
    subset_indices = random.sample(range(mci_size), subset_size)
    remaining_indices = list(set(range(mci_size)) - set(subset_indices))
    mci_subset = Subset(mci_dataset, subset_indices)
    mci_test = Subset(mci_dataset, remaining_indices)

    print(f"Subset size: {len(mci_subset)}")
    print(f"Remaining size: {len(mci_test)}")

    subset_mci_size = len(mci_subset)

    train_size_mci = int(0.8*subset_mci_size)
    valid_size_mci = subset_mci_size - train_size_mci 

    train_dataset_mci, val_dataset_mci = random_split(mci_subset, [train_size_mci, valid_size_mci])

    train_dataset = torch.utils.data.ConcatDataset([combined_dataset_80, train_dataset_mci])
    val_dataset = torch.utils.data.ConcatDataset([combined_dataset_20, val_dataset_mci])

    cnn = Fixed4Layer3DCNN()

    if args.mode == 'train':
        cnn.load_state_dict(torch.load(args.input_model))

        train_model(
            cnn, train_dataset, val_dataset,
            num_epochs=args.num_epochs, batch_size=args.batch_size, lr=args.lr, device=args.device, saving=args.output_model
        )
    elif args.mode == 'evaluate':
        cnn.load_state_dict(torch.load(args.evaluation_model))
        evaluate_test(cnn, mci_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script pour l'entraînement et l'évaluation")
    
    # Argument pour le mode (entrainement ou évaluation)
    parser.add_argument(
        'mode', choices=['train', 'evaluate'], help="Choisissez l'action à effectuer: 'train' ou 'evaluate'"
    )
    
    # Argument pour le device (CPU ou CUDA)
    parser.add_argument(
        '--device', default='cpu', choices=['cpu', 'cuda'], help="Choisissez le device ('cpu' ou 'cuda')"
    )

    # Argument pour le modèle de base (input)
    parser.add_argument(
        '--input_model', required=False, help="Le chemin vers le fichier .pth du modèle de base"
    )

    # Argument pour le modèle généré après entraînement (output)
    parser.add_argument(
        '--output_model', required=False, help="Le chemin où enregistrer le modèle entraîné"
    )

    # Argument pour le modèle de validation dans le mode 'evaluate'
    parser.add_argument(
        '--evaluation_model', required=False, help="Le chemin vers le fichier .pth du modèle à évaluer"
    )

    # Arguments pour l'entraînement (en mode 'train')
    parser.add_argument(
        '--num_epochs', type=int, default=7, help="Le nombre d'époques pour l'entraînement"
    )
    parser.add_argument(
        '--batch_size', type=int, default=32, help="La taille du batch pour l'entraînement"
    )
    parser.add_argument(
        '--lr', type=float, default=1e-5, help="Le taux d'apprentissage pour l'entraînement"
    )

    args = parser.parse_args()
    main(args)