# Approche 2.5D - Source

Ce dossier contient les scripts principaux pour la gestion, l'entraînement, et l'inférence d'un modèle de classification médicale à partir de données d'IRM cérébrales. L'architecture est organisée comme suit :

## Architecture du dossier

```
Approche25D/src
├── inference.py
├── README.md
├── train.py
└── utils
    └── split.py
```

## Description des fichiers

### `inference.py`
Script permettant l'inférence sur un ensemble de données de test.

- **Principales fonctionnalités :**
  - Chargement d'un modèle pré-entraîné.
  - Prédiction des classes des images de test.
  - Calcul des métriques, notamment la matrice de confusion et la précision globale.
  - Sauvegarde de la matrice de confusion sous forme d'image.

- **Usage :**
  ```
  python inference.py --model_path <chemin_du_modèle> 
  --test_data_path <chemin_données_test>
  --csv_path <chemin_du_csv> 
  --batch_size <taille_batch> 
  --num_classes <nombre_classes>
  --num_workers <nombre_workers> 
  --save_dir <répertoire_sauvegarde>
  ```

### `train.py`
Script pour entraîner un modèle sur des données d'entraînement et de validation.

- **Principales fonctionnalités :**
  - Chargement des données d'entraînement et de validation.
  - Entraînement du modèle.
  - Validation périodique du modèle.
  - Enregistrement des checkpoints (modèles sauvegardés) et des logs TensorBoard.
  - Calcul des métriques par classe.

- **Usage :**
  ```
  python train.py --data_path <chemin_données> 
  --csv_path <chemin_du_csv> 
  --save_dir <répertoire_checkpoints> 
  --log_dir <répertoire_logs> 
  --batch_size <taille_batch> 
  --epochs <nombre_epochs> 
  --lr <taux_apprentissage> 
  --num_classes <nombre_classes> 
  --weights_path <poids_pré-entrainés>
  ```

### `utils/split.py`
Script pour diviser un ensemble de données en trois sous-ensembles : entraînement, validation et test.

- **Principales fonctionnalités :**
  - Division stratifiée des données en respectant les proportions fournies.
  - Vérification de l'existence des fichiers nécessaires (scans et masques).
  - Copie des données dans des dossiers séparés.

- **Usage :**
  ```
  python utils/split.py <input_dir> <test_ratio> <val_ratio> <output_dir>
  ```
---

## Instructions générales

1. **Préparation des données :**
   - Assurez-vous que vos données sont bien organisées dans le dossier d'entrée et qu'un fichier CSV correspondant aux métadonnées est présent.
   - Utilisez `utils/split.py` pour créer les sous-ensembles d'entraînement, de validation et de test.

2. **Entraînement :**
   - Lancez le script `train.py` pour entraîner le modèle avec les sous-ensembles générés.

3. **Inférence :**
   - Une fois le modèle entraîné, utilisez `inference.py` pour effectuer des prédictions et évaluer les performances sur le jeu de test.
