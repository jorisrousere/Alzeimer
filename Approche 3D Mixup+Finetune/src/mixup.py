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
    train_dataset = torch.utils.data.ConcatDataset([dataset80, mixup_dataset_80])
    val_dataset = torch.utils.data.ConcatDataset([dataset20, mixup_dataset_20])
    
    cnn = Fixed4Layer3DCNN()

    train_model(
        cnn, train_dataset, val_dataset, 
        num_epochs=args.num_epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.wd, device=args.device, saving=args.output_model
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script pour l'entraînement et l'évaluation")
    
    # Argument pour le device (CPU ou CUDA)
    parser.add_argument(
        '--device', default='cpu', choices=['cpu', 'cuda'], help="Choisissez le device ('cpu' ou 'cuda')"
    )
    # Argument pour le modèle généré après entraînement (output)
    parser.add_argument(
        '--output_model', required=True, help="Le chemin où enregistrer le modèle entraîné"
    )
    # Arguments pour l'entraînement (en mode 'train')
    parser.add_argument(
        '--num_epochs', type=int, default=100, help="Le nombre d'époques pour l'entraînement"
    )
    parser.add_argument(
        '--batch_size', type=int, default=32, help="La taille du batch pour l'entraînement"
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4, help="Le taux d'apprentissage pour l'entraînement"
    )
    parser.add_argument(
        '--wd', type=float, default=1e-5, help="Le wieght decay pour l'entraînement"
    )

    args = parser.parse_args()
    main(args)