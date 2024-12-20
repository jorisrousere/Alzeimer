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
from model import Fixed4Layer3DCNN, GradCAM3D
from dataset import HippocampusDataset3D, TestHippocampusDataset3D
import matplotlib.pyplot as plt
import seaborn as sns
from utils import EarlyStopping
import argparse
from dotenv import load_dotenv

device = torch.device('cpu')
load_dotenv()
transform = Compose([
    Normalize(mean=[0.5], std=[0.5])  # Ajuste les valeurs en fonction des statistiques des données
])

# Fonction pour entrainer le modèle 
def train_model(model, train_dataset, val_dataset, num_epochs=100, batch_size=32, lr=1e-3, weight_decay=1e-5, saving='base.pth', device='cpu'):
    model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    early_stopping = EarlyStopping(patience=3, verbose=True)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
                        
            optimizer.zero_grad()
            
            # Passer les données mixées dans le modèle
            outputs = model(inputs)
            
            # Calculer la perte mixup
            loss = F.binary_cross_entropy_with_logits(outputs, labels.float().unsqueeze(1))      
            
            loss.backward()
            optimizer.step()
            
            # Calcul de la précision
            preds = torch.round(torch.sigmoid(outputs))
            running_corrects += ((preds.squeeze() == 1) == (labels > 0.5)).sum().item()
            running_loss += loss.item()
            total += labels.size(0)

            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}', 
                'Accuracy': f'{running_corrects/total:.4f}'
            })
        
        # Calculer les métriques
        train_loss = running_loss / len(train_loader)
        train_accuracy = running_corrects / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation
        model.eval()
        val_loss, val_accuracy = evaluate_model(model, val_loader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Sauvegarde du meilleur modèle
        if early_stopping(val_loss, model):
            print("Early stopping")
            break
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
    
        # Charger le meilleur modèle
        model.load_state_dict(early_stopping.best_model_state)
        torch.save(early_stopping.best_model_state, saving)

# Fonction pour évaluer le modèle sur un jeu de validation
def evaluate_model(model, val_loader, device=device):
    model.eval()  # Passer en mode évaluation
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Passer les données dans le modèle
            outputs = model(inputs)

            # Calculer la perte
            loss = F.binary_cross_entropy_with_logits(outputs, labels.float().unsqueeze(1))
            # Statistiques
            running_loss += loss.item()
            preds = torch.round(torch.sigmoid(outputs))  # Prédictions binaires
            running_corrects += ((preds.squeeze() == 1) == (labels > 0.5)).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(val_loader)
    val_accuracy = running_corrects / total
    return val_loss, val_accuracy

def main(args):
    # Charger les chemins depuis les variables d'environnement
    data_dir = os.getenv('DATA_DIR')  # Valeur par défaut si la variable n'existe pas
    csv_path = os.getenv('CSV_PATH')  # Valeur par défaut si la variable n'existe pas
    dataset = HippocampusDataset3D(
        data_dir=data_dir,
        csv_path=csv_path,
        transform=transform
    )
 
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Fixer une graine pour la reproductibilité
    seed = 42
    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )

    model = Fixed4Layer3DCNN()

    train_model(
        model, train_dataset, val_dataset, 
        num_epochs=args.num_epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.wd, saving=args.output_model, device=args.device
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