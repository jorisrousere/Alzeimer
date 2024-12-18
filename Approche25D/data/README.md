# Approche25D - Dataloaders pour IRM Cérébrales

Ce dossier contient des scripts Python pour générer des DataLoaders PyTorch permettant de classer des IRM cérébrales en vue de détecter des maladies comme Alzheimer ou les troubles cognitifs légers (MCI).

## Architecture du dossier

```
Approche25D/data/
├── medicalclassdataset_train.py
├── medicalclassdataset_valid.py
└── README.md
```

## Description des fichiers

### 1. `medicalclassdataset_train.py`
Ce script définit la classe `MedicalClassDataset`, utilisée pour charger les données d'entraînement. Les principales caractéristiques sont :

- **Initialisation des données** :
  - Chargement des scans et des masques à partir d'un dossier.
  - Lecture des métadonnées dans un fichier CSV.
- **Prétraitement** :
  - Redimensionnement des images à une taille cible (par défaut, 256x256).
  - Normalisation des valeurs des pixels.
  - Extraction de plusieurs tranches consécutives (slices) pour l'approche en 2.5D.
- **Augmentation des données** (facultatif) :
  - Ajustement aléatoire de la luminosité, du contraste, et de la netteté.
- **Sortie** : Un tenseur 3D correspondant à plusieurs tranche IRM et son label de classe.

### 2. `medicalclassdataset_valid.py`
Similaire au script d'entraînement, ce fichier adapte la classe `MedicalClassDataset` pour valider le modèle sur les données. Les différences principales incluent :

- Les labels de classe se concentrent principalement sur les patients présentant des troubles cognitifs légers (MCI), distinguant les sous-types selon les métadonnées fournies.
- L'augmentation des données est désactivée pour garantir des résultats d'évaluation cohérents.

## Utilisation

### Initialisation de la classe `MedicalClassDataset`

```python
from medicalclassdataset_train import MedicalClassDataset

dataset = MedicalClassDataset(
    folder_path="path/vers/les/scans",
    csv_file="path/vers/metadata.csv",
    target_size=(256, 256),
    num_slices=3,
    stride=1,
    ranges={"axial": (40, 80), "sagittal": (40, 140), "coronal": (90, 130)},
    apply_augmentation=True
)
```

### Accéder aux données

Une fois la classe initialisée, vous pouvez récupérer les données avec PyTorch :

```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=16, shuffle=True)

for scans, labels in loader:
    # Traitement des données
    pass
```
