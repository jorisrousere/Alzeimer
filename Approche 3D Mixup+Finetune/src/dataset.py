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
from torch.utils.data import DataLoader
import torch.nn.functional as F


transform = Compose([
    lambda x: x / 255.0,  
    Normalize(mean=[0.5], std=[0.5])
])

class HippocampusDataset3D(Dataset):
    def __init__(self, data_dir, csv_path, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.metadata = pd.read_csv(csv_path, header=None)
        self.metadata.columns = [
            'subject_id', 'patient_id', 'age', 'gender', 'diagnosis', 
            'apoe', 'education', 'mmse', 'clinical_status', 'label1', 'label2'
        ]

        self.metadata = self.metadata[self.metadata['diagnosis'].isin(['CN', 'AD'])]

        self.metadata['binary_label'] = self.metadata['diagnosis'].apply(lambda x: 1 if x == 'AD' else 0)

        # Charger les fichiers d'images et de masques, et filtrer par sujet ID
        self.image_files = []
        for f in os.listdir(data_dir):
            if f.startswith('n_mmni_') and f.endswith('.nii.gz'):
                subject_id = re.search(r'(\d{3}_S_\d{4})', f).group(1)
                if subject_id in self.metadata['subject_id'].values:
                    self.image_files.append(f)

        # Trier les fichiers pour garantir un ordre cohérent
        self.image_files = sorted(self.image_files)

        self.coords = [
            (slice(40, 80), slice(90, 130), slice(40, 80)),     # Hippocampe droit
            (slice(100, 140), slice(90, 130), slice(40, 80))    # Hippocampe gauche
        ]
    
    def __len__(self):
        return len(self.image_files) * 2

    def __getitem__(self, idx):
        original_idx = idx // 2  # Identifier l'image originale
        hippo_position = idx % 2  # Identifier l'hippocampe (droit ou gauche)

        img_path = os.path.join(self.data_dir, self.image_files[original_idx])

        # Extraire le subject_id à partir du nom du fichier
        subject_id = re.search(r'(\d{3}_S_\d{4})', self.image_files[original_idx]).group(1)

        # Charger les fichiers NIfTI
        img_nii = nib.load(img_path)

        image = img_nii.get_fdata()

        # Extraire le patch de l'hippocampe en fonction de la position
        c = self.coords[hippo_position]
        image_patch = np.copy(image[c])  # Créer une copie explicite de l'array

        if hippo_position == 0:  # Hippocampe droit
            image_patch = np.flip(image_patch, axis=0)
                
        # Récupérer le label depuis les métadonnées
        filtered_metadata = self.metadata[self.metadata['subject_id'] == subject_id]
        if not filtered_metadata.empty:
            label = filtered_metadata['binary_label'].values[0]
        else:
            print(f"Warning: No metadata found for subject ID: {subject_id}")
            label = 0  

        # Convertir en tenseurs
        image_tensor = torch.from_numpy(image_patch.copy()).float().unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Appliquer la transformation (normalisation)
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, label_tensor

class MixupAugmentedDataset(Dataset):
    def __init__(self, base_dataset, augment_factor=5, lam_values=None):
        self.base_dataset = base_dataset
        self.augment_factor = augment_factor
        
        self.lam_values = torch.tensor([i / 10 for i in range(11)]) if lam_values is None else torch.tensor(lam_values)

        self.class_0_indices = []
        self.class_1_indices = []
        
        for idx in range(len(self.base_dataset)):
            _, label = self.base_dataset[idx]
            if label == 0:
                self.class_0_indices.append(idx)
            elif label == 1:
                self.class_1_indices.append(idx)

        self.augmented_indices = self._generate_balanced_indices()

    def _generate_balanced_indices(self):
        indices_pairs = []

        min_class_count = min(len(self.class_0_indices), len(self.class_1_indices))
        
        for _ in range(min_class_count * self.augment_factor):
            idx0 = np.random.choice(self.class_0_indices)
            idx1 = np.random.choice(self.class_1_indices)
            
            indices_pairs.append((idx0, idx1))
        
        return indices_pairs

    def __len__(self):
        return len(self.base_dataset) + len(self.augmented_indices) * len(self.lam_values)

    def __getitem__(self, idx):
        if idx < len(self.base_dataset):
            return self.base_dataset[idx]
        
        aug_idx = (idx - len(self.base_dataset)) // len(self.lam_values)
        lam_idx = (idx - len(self.base_dataset)) % len(self.lam_values)

        idx0, idx1 = self.augmented_indices[aug_idx]

        img0, label0 = self.base_dataset[idx0]
        img1, label1 = self.base_dataset[idx1]

        lam = self.lam_values[lam_idx].item()

        mixed_image = lam * img0 + (1 - lam) * img1
        mixed_label = lam * label0 + (1 - lam) * label1
        
        return mixed_image, mixed_label

    def generate_and_save_dataset(self, output_path):
        base_len = len(self.base_dataset)
        total_len = len(self)
        
        with h5py.File(output_path, 'w') as f:
            images = f.create_dataset('images', shape=(total_len, *self.base_dataset[0][0].shape), dtype='f')
            labels = f.create_dataset('labels', shape=(total_len,), dtype='f')

            for idx in range(base_len):
                print(idx)
                img, label = self.base_dataset[idx]
                images[idx] = img
                labels[idx] = label

            for aug_idx, (idx0, idx1) in enumerate(self.augmented_indices):
                print(aug_idx)
                img0, label0 = self.base_dataset[idx0]
                img1, label1 = self.base_dataset[idx1]
                
                for lam_idx, lam in enumerate(self.lam_values):
                    lam = lam.item()
                    mixed_image = lam * img0 + (1 - lam) * img1
                    mixed_label = lam * label0 + (1 - lam) * label1
                    
                    images[base_len + aug_idx * len(self.lam_values) + lam_idx] = mixed_image
                    labels[base_len + aug_idx * len(self.lam_values) + lam_idx] = mixed_label
                    print(mixed_label)
            
            print(f"Dataset Mixup équilibré sauvegardé dans {output_path}")

class TestHippocampusDataset3D(Dataset):
    def __init__(self, data_dir, csv_path, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Charger et filtrer le CSV
        self.metadata = pd.read_csv(csv_path, header=None)
        self.metadata.columns = [
            'subject_id', 'patient_id', 'age', 'gender', 'diagnosis', 
            'apoe', 'education', 'mmse', 'clinical_status', 'label1', 'label2'
        ]

        # Filtrer uniquement MCI
        self.metadata = self.metadata[self.metadata['diagnosis'].isin(['MCI'])]
        self.metadata['apoe'] = self.metadata['apoe'].astype(int)
        self.metadata = self.metadata[self.metadata['apoe'].isin([3, 4])]

        # Créer un label binaire : 0 pour CN, 1 pour AD
        self.metadata['binary_label'] = self.metadata['apoe'].apply(lambda x: 0 if x == 4 else 1)

        # Charger les fichiers d'images et de masques, et filtrer par sujet ID
        self.image_files = []
        for f in os.listdir(data_dir):
            if f.startswith('n_mmni_') and f.endswith('.nii.gz'):
                subject_id = re.search(r'(\d{3}_S_\d{4})', f).group(1)
                if subject_id in self.metadata['subject_id'].values:
                    self.image_files.append(f)

        # Trier les fichiers pour garantir un ordre cohérent
        self.image_files = sorted(self.image_files)

        self.coords = [
            (slice(40, 80), slice(90, 130), slice(40, 80)),     # Hippocampe droit
            (slice(100, 140), slice(90, 130), slice(40, 80))    # Hippocampe gauche
        ]
    
    def __len__(self):
        return len(self.image_files) * 2

    def __getitem__(self, idx):
        original_idx = idx // 2  # Identifier l'image originale
        hippo_position = idx % 2  # Identifier l'hippocampe (droit ou gauche)

        img_path = os.path.join(self.data_dir, self.image_files[original_idx])

        # Extraire le subject_id à partir du nom du fichier
        subject_id = re.search(r'(\d{3}_S_\d{4})', self.image_files[original_idx]).group(1)

        # Charger les fichiers NIfTI
        img_nii = nib.load(img_path)

        image = img_nii.get_fdata()

        # Extraire le patch de l'hippocampe en fonction de la position
        c = self.coords[hippo_position]
        image_patch = np.copy(image[c])  # Créer une copie explicite de l'array

        if hippo_position == 0:  # Hippocampe droit
            image_patch = np.flip(image_patch, axis=0)
                
        # Récupérer le label depuis les métadonnées
        filtered_metadata = self.metadata[self.metadata['subject_id'] == subject_id]
        if not filtered_metadata.empty:
            label = filtered_metadata['binary_label'].values[0]
        else:
            print(f"Warning: No metadata found for subject ID: {subject_id}")
            label = 0  

        # Convertir en tenseurs
        image_tensor = torch.from_numpy(image_patch.copy()).float().unsqueeze(0)  
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Appliquer la transformation (normalisation)
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, label_tensor

class HDF5Dataset(Dataset):
    def __init__(self, h5_file_path):
        """
        Initialise le dataset à partir d'un fichier HDF5.
        :param h5_file_path: Chemin vers le fichier .h5.
        """
        self.h5_file_path = h5_file_path

        # Ouvrir le fichier une première fois pour obtenir la longueur des données
        with h5py.File(self.h5_file_path, 'r') as f:
            self.data_len = len(f['images'])  # Longueur des données (nombre d'images)
        
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        # Ouvrir le fichier au moment de l'accès pour éviter de le garder ouvert
        with h5py.File(self.h5_file_path, 'r') as f:
            image = torch.tensor(f['images'][idx], dtype=torch.float32)
            label = torch.tensor(f['labels'][idx], dtype=torch.float32)
        
        return image, label
    
def mixup_data(x, y, alpha=0.2, device='cpu'):
    """
    Génère des données mixup
    
    Args:
    - x : tenseur des images
    - y : tenseur des labels
    - alpha : paramètre de distribution Beta pour le mixup
    - device : périphérique de calcul
    
    Returns:
    - Mixed inputs, mixed labels, et lambda de mixup
    """
    if alpha > 0:
        # Échantillonnage de lambda à partir d'une distribution Beta
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    # Taille du batch
    batch_size = x.size()[0]
    
    # Génération d'un index aléatoire pour le mélange
    index = torch.randperm(batch_size).to(device)

    # Mélange des images
    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    # Mélange des labels
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Fonction de perte pour le mixup
    
    Args:
    - criterion : fonction de perte originale
    - pred : prédictions du modèle
    - y_a, y_b : labels originaux
    - lam : coefficient de mixup
    
    Returns:
    - Perte mixup
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def create_mixup_dataset(base_dataset, output_path, num_mixup_samples, alpha=0.2, device='cpu'):
    """
    Crée et enregistre un nouveau dataset enrichi avec des données mixup.
    
    Args:
    - base_dataset : le dataset de base utilisé pour générer les données mixées
    - output_path : chemin pour enregistrer le nouveau dataset (au format HDF5)
    - num_mixup_samples : nombre d'échantillons mixés à ajouter
    - alpha : paramètre pour la distribution Beta du mixup
    - device : périphérique pour le calcul (CPU ou GPU)
    """
    # Charger les données de base dans un DataLoader pour faciliter l'accès
    base_loader = DataLoader(base_dataset, batch_size=1, shuffle=True)
    
    # Liste pour stocker les données et les labels mixés
    mixed_images = []
    mixed_labels = []

    # Générer les données mixées
    for i in range(num_mixup_samples):
        print(i)
        # Prendre deux échantillons aléatoires
        idx1, idx2 = np.random.choice(len(base_dataset), size=2, replace=False)
        img1, label1 = base_dataset[idx1]
        img2, label2 = base_dataset[idx2]

        # Convertir en tenseurs et déplacer vers le périphérique
        img1, img2 = img1.to(device), img2.to(device)
        label1, label2 = torch.tensor(label1, dtype=torch.float32).to(device), torch.tensor(label2, dtype=torch.float32).to(device)
        
        # Appliquer Mixup
        mixed_img, mixed_label_a, mixed_label_b, lam = mixup_data(img1.unsqueeze(0), label1.unsqueeze(0), alpha, device)
        
        # Calculer le label mixé final
        final_label = lam * mixed_label_a + (1 - lam) * mixed_label_b
        
        # Ajouter les résultats à la liste
        mixed_images.append(mixed_img.squeeze(0).cpu().numpy())
        mixed_labels.append(final_label.squeeze(0).cpu().numpy())

    # Enregistrer les données mixées dans un fichier HDF5
    with h5py.File(output_path, 'w') as h5f:
        h5f.create_dataset('images', data=np.array(mixed_images))
        h5f.create_dataset('labels', data=np.array(mixed_labels))
    
    print(f"Dataset mixup créé et enregistré à {output_path}")