```markdown
# Projet de Modélisation et Entraînement de Modèles

Ce projet contient plusieurs scripts Python pour l'entraînement, l'évaluation et la gestion de modèles de machine learning. Il est structuré pour permettre une utilisation flexible avec différents arguments en ligne de commande.

## Structure du Projet

```
├── README.md           # Documentation du projet
├── requirements.txt    # Dépendances du projet
└── src                 # Code source
    ├── base.py         # Script de base pour l'entraînement du modèle
    ├── dataset.py      # Chargement et gestion des données
    ├── fine_tuning.py  # Script pour le fine-tuning du modèle
    ├── mixup.py        # Script pour le mixup des données
    ├── model.py        # Définition et architecture du modèle
    ├── test.py         # Script d'évaluation et de test du modèle
    └── utils.py        # Fonctions utilitaires
```

## Environnement

Le projet nécessite un environnement Python 3.6+ avec les dépendances spécifiées dans `requirements.txt`. Pour configurer l'environnement, vous pouvez utiliser un environnement virtuel :

```bash
python -m venv env
source env/bin/activate  # Pour Linux/MacOS
env\Scripts\activate     # Pour Windows
```

## Installation

Pour installer les dépendances :

```bash
pip install -r requirements.txt
```

## Fichier .env

Créez un fichier `.env` qui contient :

```
DATA_DIR=path/to/data
CSV_PATH=path/to/csv
```

## Scripts

### 1. `base.py`

Le script `base.py` est utilisé pour entraîner un modèle de CNN basique 3D à 4 couches. Les arguments sont :

- `--device`: Spécifie le périphérique à utiliser (`cpu` ou `cuda`).
- `--output_model`: Le chemin où enregistrer le modèle entraîné.
- `--num_epochs`: Le nombre d'époques pour l'entraînement.
- `--batch_size`: La taille du batch pour l'entraînement.
- `--lr`: Le taux d'apprentissage.
- `--wd`: Le poids de régularisation (weight decay).

Exemple d'utilisation :

```bash
python base.py --device cuda --output_model model.pth --num_epochs 100 --batch_size 32 --lr 1e-4 --wd 1e-5
```

### 2. `mixup.py`

Ce script permet d'entraîner un modèle avec la technique de MixUp. Les principaux arguments sont :

- `--device`: Spécifie le périphérique à utiliser (`cpu` ou `cuda`).
- `--output_model`: Le chemin où enregistrer le modèle entraîné.
- `--num_epochs`: Le nombre d'époques pour l'entraînement.
- `--batch_size`: La taille du batch pour l'entraînement.
- `--lr`: Le taux d'apprentissage.
- `--wd`: Le poids de régularisation (weight decay).

Exemple d'utilisation :

```bash
python mixup.py --device cuda --output_model model.pth --num_epochs 100 --batch_size 32 --lr 1e-4 --wd 1e-5
```

### 3. `fine_tuning.py`

Le script `fine_tuning.py` permet de fine-tuner un modèle préexistant ou d'en entraîner un nouveau. Il prend les arguments suivants :

- `mode`: Le mode d'exécution (`train` ou `evaluate`).
- `--device`: Spécifie le périphérique à utiliser (`cpu` ou `cuda`).
- `--input_model`: Le chemin vers le modèle préexistant à fine-tuner (facultatif).
- `--output_model`: Le chemin où enregistrer le modèle fine-tuné (facultatif).
- `--evaluation_model`: Le chemin vers le modèle à évaluer (facultatif).
- `--num_epochs`: Le nombre d'époques pour l'entraînement.
- `--batch_size`: La taille du batch pour l'entraînement.
- `--lr`: Le taux d'apprentissage.

Exemple d'utilisation pour l'entraînement :

```bash
python fine_tuning.py train --device cuda --output_model fine_tuned_model.pth --num_epochs 10 --batch_size 32 --lr 1e-5
```

Exemple d'utilisation pour l'évaluation :

```bash
python fine_tuning.py evaluate --device cuda --evaluation_model fine_tuned_model.pth
```

### 4. `test.py`

Le script `test.py` permet d'évaluer le modèle entraîné, y compris avec la technique Grad-CAM pour visualiser les zones importantes dans les images.

Les arguments sont :

- `--input_model`: Le chemin vers le modèle à tester.
- `--mode`: Le mode d'exécution (`grad_cam` ou `evaluation`).
- `--index`: L'index pour le Grad-CAM (facultatif).

Exemple d'utilisation pour une évaluation :

```bash
python test.py --input_model model.pth --mode evaluation
```

Exemple d'utilisation pour Grad-CAM :

```bash
python test.py --input_model model.pth --mode grad_cam --index 0
```

### 5. `dataset.py`

Le script `dataset.py` gère le chargement et le prétraitement des données utilisées pour l'entraînement du modèle.
```
