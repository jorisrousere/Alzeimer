# Approche 2.5D

Ce dépôt contient une implémentation d'une approche 2.5D pour la classification d'images provenant d'IRM du cerveau. Le projet est organisé en modules pour le traitement des données, le développement de modèles, ainsi que les workflows d'entraînement et d'inférence.

## Structure du Répertoire

```
Approche25D/
├── data
│   ├── medicalclassdataset_train.py
│   ├── medicalclassdataset_valid.py
│   └── README.md
├── network
│   ├── CNN.py
│   ├── EfficientNet.py
│   ├── README.md
│   └── SwinTransformer.py
├── README.md
└── src
    ├── inference.py
    ├── README.md
    ├── train.py
    └── utils
        └── split.py
```

### `data/`
Ce dossier contient des scripts pour prétraiter et gérer les jeux de données utilisés pour l'entraînement et la validation :
- **`medicalclassdataset_train.py`** : Définit la classe de données pour l'entraînement, incluant les augmentations et le prétraitement.
- **`medicalclassdataset_valid.py`** : Définit la classe de données pour la validation, en assurant une cohérence avec la structure des données d'entraînement.
- **`README.md`** : Fournit des explications sur le format des données, les étapes de prétraitement, et les instructions d'utilisation.

### `network/`
Ce dossier contient les architectures de réseaux de neurones utilisées dans le projet :
- **`CNN.py`** : Implémente un réseau de neurones convolutif (CNN) simple pour la classification d'images médicales.
- **`EfficientNet.py`** : Contient une implémentation de l'architecture EfficientNet pour une classification d'images performante.
- **`swinTransformer.py`** : Implémente le Swin Transformer, une architecture basée sur les transformers pour une extraction avancée des caractéristiques.
- **`README.md`** : Détaille les architectures, leurs cas d'utilisation.

### `src/`
Ce dossier regroupe les scripts pour l'entraînement, l'inférence, et les fonctions utilitaires :
- **`train.py`** : Script pour entraîner les modèles en utilisant le jeu de données fourni et les architectures définies dans `network/`. Inclut la journalisation et la sauvegarde des checkpoints.
- **`inference.py`** : Script pour effectuer des inférences sur un jeu de données de test et évaluer les performances du modèle avec des métriques comme la précision et les matrices de confusion.
- **`utils/`** : Contient des fonctions auxiliaires et des scripts, notamment :
  - **`split.py`** : Permet de diviser le jeu de données en ensembles d'entraînement, de validation, et de test tout en maintenant un équilibre des classes.
- **`README.md`** : Explique comment utiliser les scripts d'entraînement et d'inférence.

### Racine du Projet
- **`README.md`** : Présentation générale du projet, incluant les instructions d'installation et la structure globale.

## Utilisation
1. Prétraitez les données à l'aide des scripts disponibles dans `data/`.
2. Entraînez un modèle en utilisant `src/train.py` avec l'une des architectures définies dans `network/`.
3. Effectuez une inférence en utilisant `src/inference.py` pour évaluer les performances du modèle sur un jeu de données de test.

## Prérequis
- Python 3.8+
- PyTorch
- NumPy
- Scikit-learn
- Matplotlib
- tqdm
