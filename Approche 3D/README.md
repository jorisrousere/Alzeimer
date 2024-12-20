# Approche 3D - TRAITEMENT 3D DES VOLUMES IRM
## 📋 Description
Cette branche traite directement des **IRM en 3D** pour détecter des anomalies spécifiques, notamment les **hippocampes**, grâce à des réseaux de neurones 3D.  
L'objectif est de **conserver l'intégralité des informations volumétriques** afin d'améliorer la précision des modèles par rapport aux approches 2D.  

#### 🧠 **Focus sur la ROI**  
Les régions d'intérêt (**ROIs**) sont les **hippocampes**, extraits sous forme de volumes **40x40x40**. Ce format permet de capter des informations spatiales précises tout en limitant la complexité computationnelle.

## 🗂️ Structure du projet

```plaintext
BRAIN_MRI_CLASSIFICATION/
│
├── data/                          # Contient les ensembles de données
│   ├── Brain_dataset/             
│   ├── Hippocampe_dataset_mci/   
│   └── list_standardized_tongtong_2017.csv  # Fichier csv avec différentes informations
│
├── models/                        # Sauvegarde des modèles entraînés
│
├── outputs/                       # Résultats et métriques (ex : courbes ROC, logs)
│
├── notebooks/                     # Notebooks Jupyter pour l'analyse
│   └── bdd_analysis.ipynb         # Notebook pour analyser la base de données
│
├── scripts/                       # Scripts bash pour automatiser les tâches
│   ├── process_data.sh            # Prétraitement des données
│   ├── train.sh                   # Script pour lancer l'entraînement
│   ├── run_pipeline.sh            # Pipeline complet (prétraitement -> entraînement -> évaluation)
│   ├── test_pipeline.sh           # Tests automatisés de la pipeline
│   └── test.sh                    # Script pour exécuter les tests unitaires
│
├── src/                           # Code source du projet
│   ├── datasets.py                # Gestion et chargement des jeux de données
│   ├── models.py                  # Définition des architectures de réseaux neuronaux
│   ├── process_data.py            # Prétraitement des données
│   ├── train_model.py             # Entraînement du modèle
│   ├── test_models.py             # Test et évaluation des modèles
│   ├── train.py                   # Script principal d'entraînement
│   ├── train_without_val.py       # Script d'entraînement sur train + val
│   └── plot_utils.py              # Fonctions de visualisation des résultats
│
├── .gitignore                     # Liste des fichiers/dossiers à ignorer par Git
├── venv/                          # Environnement virtuel Python
├── requirements.txt               # Liste des dépendances Python
└── README.md                      # Documentation du projet
```
train_without_val.py
---

## 🚀 Installation

#### Étapes d'installation

1. Clone ce repository :
   ```bash
   git clone https://github.com/ton-projet/BRAIN_MRI_CLASSIFICATION.git
   cd BRAIN_MRI_CLASSIFICATION
   ```

2. Crée un environnement virtuel et active-le :
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   .\venv\Scripts\activate   # Windows
   ```

3. Installe les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Utilisation
Le **modèle** ainsi que les **hyperparamètres** (nombre d'époques, taille du batch, learning rate, etc.) peuvent être modifiés directement dans le script `scripts/train.sh`.

#### Prétraitement des données
Exécute le script pour prétraiter les volumes IRM et extraire les hippocampes en 40x40x40
```bash
bash scripts/process_data.sh
```

#### Entraînement du modèle
Entraîne le modèle en utilisant les paramètres spécifiés dans train.sh 
```bash
bash scripts/train.sh
```

#### Tester le modèle
Évalue les performances du modèle sur l'ensemble de test
```bash
bash scripts/test.sh
```

#### Lancer l'ensemble du pipeline
Pour lancer toutes les étapes automatiquement (prétraitement, entraînement, évaluation)
```bash
bash scripts/run_pipeline.sh
```

#### Résultats
Les courbes de l'entraînement et de l'évaluation sont sauvegardés dans le répertoire **outputs**.
Les modèles ont été évalué sur le jeux de données MCI Stable/Instable. Les résultats sont résumés dans le tableau suivant :

| **Architecture**         | **Accuracy (%)** | **F1-score** | **AUC** |
|---------------------------|------------------|--------------|---------|
| ThreeLayer3DCNN           | 62.5            | 0.66         | 0.69    |
| FourLayer3DCNN            | 69.5            | 0.73         | 0.75    |
| FiveLayer3DCNN            | 63.6            | 0.67         | 0.71    |
| ResNet3D                  | 65.8            | 0.67         | 0.71    |
| ResNet3DWithAttention     | 67.2            | 0.70         | 0.73    |

## 🎥 Démonstration vidéo
![Aperçu](demo.gif)
