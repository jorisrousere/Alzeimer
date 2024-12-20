# Approche 2D - Alzheimer Classification

Ce dossier contient l'approche 2D développée dans le cadre d'un projet plus large visant à classifier les patients atteints d'Alzheimer (AD), cognitivement normaux (CN) et à prédire la stabilité des patients Mild Cognitive Impairment (MCI). L'approche 2D repose sur un modèle CNN prenant en entrée des coupes IRM 2D de la région de l'hippocampe.

## Structure du dossier
```
2D/
├── README.md               # Documentation de l'approche 2D
├── datasets/               # Données utilisées pour l'approche 2D
│   ├── raw/                # Données brutes
│   └── split/              # Données prétraitées (train, val, test)
├── scripts/                # Scripts pour exécuter les tâches principales
│   ├── run_evaluate.sh     # Script pour évaluer le modèle
│   └── run_train.sh        # Script pour entraîner le modèle
└── src/                    # Code source de l'approche 2D
    ├── data_loader_adcn.py # Dataloader pour le dataset AD/CN
    ├── data_loader_mci.py  # Dataloader pour le dataset MCI
    ├── data_loader_train.py # Dataloader pour l'entraînement
    ├── evaluate_adcn.py    # Évaluation sur le dataset AD/CN
    ├── evaluate_mci.py     # Évaluation sur le dataset MCI
    ├── model2D.py          # Architecture du modèle CNN
    ├── train.py            # Fichier pour entraîner le modèle
    ├── utils/              # Outils divers
        ├── split_dataset.py # Script pour diviser les données
        └── search_grid.py  # Recherche par grille pour optimiser l'architecture
```
## Description des fichiers

### Datasets
- `datasets/raw/` : Contient les données brutes.
- `datasets/split/` : Contient les données organisées en `train/`, `val/`, et `test/`.

### Scripts
- `run_train.sh` : Script pour lancer l'entraînement du modèle.
- `run_evaluate.sh` : Script pour évaluer le modèle sur les datasets AD/CN et MCI.

### Code Source
- `data_loader_adcn.py` : Dataloader pour charger les données AD/CN.
- `data_loader_mci.py` : Dataloader pour charger les données MCI.
- `data_loader_train.py` : Dataloader pour préparer les données d'entraînement.
- `evaluate_adcn.py` : Fichier pour évaluer le modèle sur AD/CN avec différentes fonctions de pondération.
- `evaluate_mci.py` : Fichier pour évaluer le modèle sur MCI Stable/Instable.
- `model2D.py` : Définition de l'architecture du modèle CNN utilisé.
- `train.py` : Script pour entraîner le modèle sur les données AD/CN.
- `utils/split_dataset.py` : Script pour diviser les données en train/val/test.
- `utils/search_grid.py` : Recherche d'architecture optimale via une grid search.

## Résultats obtenus

Le modèle CNN a été évalué sur deux jeux de données : AD/CN et MCI Stable/Instable. Les résultats sont résumés dans le tableau suivant :

| Métrique    | AD/CN   | MCI Stable/Instable |
|-------------|---------|---------------------|
| Accuracy    | 79.0%   | 68.0%               |
| Precision   | 81.0%   | 70.0%               |
| Recall      | 78.0%   | 66.0%               |
| F1 Score    | 79.0%   | 67.0%               |
| AUC         | 81.0%   | 72.0%               |

Nous avons utilisé une pondération logarithmique pour calculer la moyenne pondérée des prédictions des coupes IRM. Cette méthode s'est révélée efficace, en particulier sur les données MCI.

## Lancer l'entraînement et l'évaluation

### Entraînement
1. Assurez-vous que les données sont organisées dans `datasets/split/` avec les sous-dossiers `train/`, `val/`, et `test/`.
2. Lancez le script d'entraînement :
```bash
./scripts/run_train.sh
```

### Évaluation

1. Évaluez sur le dataset AD/CN :
```bash
./scripts/run_evaluate.sh adcn
```

2. Lancez le script d'entraînement :
```bash
./scripts/run_evaluate.sh mci
```

Cette approche basée sur des coupes IRM 2D montre des performances prometteuses, notamment grâce à l'utilisation d'une pondération logarithmique pour l'agrégation des prédictions. Bien que des améliorations soient encore possibles, ce modèle constitue une base solide pour la classification des patients atteints d'Alzheimer et l'étude des patients MCI.
