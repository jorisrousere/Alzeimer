import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import nibabel as nib
import numpy as np
from dotenv import load_dotenv
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
from base import transform
import argparse

load_dotenv()

device = torch.device('cpu')

# Fonction pour afficher les Grad-CAMs côte à côte
def plot_gradcam_3d(input_image, cams, convolution_indices):
    fig = plt.figure(figsize=(15, 12))

    # Créer des sous-graphiques : 1 pour l'image d'entrée et 4 pour les Grad-CAMs
    ax1 = fig.add_subplot(1, 5, 1, projection='3d')

    z, y, x = np.indices(input_image.shape)

    # Normalisation de l'image d'entrée
    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())

    # Affichage de l'image d'entrée
    ax1.voxels(np.ones_like(input_image), facecolors=plt.cm.gray(input_image), edgecolor='none', alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title("Image d'entrée")

    # Affichage des cartes Grad-CAM pour les convolutions 1 à 4
    for i, (cam, conv_index) in enumerate(zip(cams, convolution_indices)):
        ax = fig.add_subplot(1, 5, i+2, projection='3d')
        cam_threshold = cam > (cam.mean() + cam.std())

        # Affichage de la carte Grad-CAM
        ax.voxels(np.ones_like(input_image), facecolors=plt.cm.gray(input_image), edgecolor='none', alpha=0.6)
        ax.voxels(cam_threshold, facecolors=plt.cm.jet(cam / cam.max()), edgecolor='none', alpha=0.8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Grad-CAM Conv{conv_index}")

    plt.subplots_adjust(left=0.1, right=0.9)  # Augmenter l'espace à gauche et à droite
    plt.tight_layout()
    plt.show()

# Fonction pour calculer le Grad-CAM pour plusieurs convolutions
def grad_cam(model, images):
    model.eval()

    # Liste pour stocker les cartes de Grad-CAM pour chaque convolution
    cams = []
    convolution_indices = [1, 2, 3, 4]  # Convolutions à afficher

    # Pour chaque convolution spécifiée
    for conv_index in convolution_indices:
        grad_cam = GradCAM3D(model, getattr(model, f'conv{conv_index}'))
        
        cam = grad_cam.generate_cam(images)
        cam = F.interpolate(cam.unsqueeze(1), size=(40, 40, 40), mode='trilinear', align_corners=False)
        cam = cam.squeeze(1)  
        cam = cam.squeeze().cpu().numpy()

        cams.append(cam)

    # Prendre la première image du lot (images[0]) pour l'affichage
    input_image = images[0, 0].cpu().numpy()

    # Affichage des Grad-CAMs côte à côte
    plot_gradcam_3d(input_image.squeeze(), cams, convolution_indices)

# Fonction pour évaluer le modèle sur un jeu de test
def evaluate_test(model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.round(torch.sigmoid(outputs))

            # Convert single label tensor to a list
            all_predictions.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())  # Use .flatten()

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Précision sur le jeu de test : {accuracy:.4f}")

    # Calcul et affichage de la matrice de confusion
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"Matrice de confusion :\n{cm}")

    # Visualisation de la matrice de confusion
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Classe 0", "Classe 1"], yticklabels=["Classe 0", "Classe 1"])
    plt.ylabel('Label réel')
    plt.xlabel('Prédiction')
    plt.title('Matrice de confusion')
    plt.show()

def main(args):
    data_dir = os.getenv('DATA_DIR') 
    csv_path = os.getenv('CSV_PATH') 
    model = Fixed4Layer3DCNN()
    model.load_state_dict(torch.load(args.input_model))

    test_dataset = TestHippocampusDataset3D(
        data_dir=data_dir,
        csv_path=csv_path,
        transform=transform
    )

    if args.mode == 'gradcam':
        image, label = test_dataset[args.index] 
        print(label)
        image = image.unsqueeze(0).to(device)  
        grad_cam(model, image)

    elif args.mode == 'evaluate':
        evaluate_test(model, test_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script pour l'entraînement et l'évaluation")
    
    # Argument pour le modèle généré après entraînement (output)
    parser.add_argument(
        '--input_model', required=True, help="Le chemin où récuperer le modèle à tester"
    )
    parser.add_argument(
        '--mode', required=True, help="Le mode (grad cam ou evaluation)"
    )
    parser.add_argument(
        '--index', type=int, required=False, help="L'index pour le grad cam"
    )

    args = parser.parse_args()
    main(args)