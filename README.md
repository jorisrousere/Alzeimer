# Projet : Classification des IRM avec Approches 2D, 2.5D et 3D

Ce projet propose plusieurs approches pour analyser des images IRM en utilisant des architectures 2D, 2.5D et 3D afin de classifier les données MCI. Chaque approche est structurée dans un répertoire séparé pour une meilleure organisation.

## Structure du projet

```plaintext
.
├── Approche3D/
│   ├── scripts/          # Scripts pour le pré-traitement, l'entraînement, et l'évaluation
│   ├── src/              # Code source des modèles 3D
│   ├── README.md         # Documentation spécifique à l'approche 3D
│   ├── demo.gif          # Démonstration du fonctionnement du modèle 3D
│   └── requirements.txt  # Liste des dépendances nécessaires pour l'approche 3D
├── Approche25D/
│   ├── data/             # Contient les données nécessaires pour l'approche 2.5D
│   ├── network/          # Définition des architectures pour l'approche 2.5D
│   ├── src/              # Code source des scripts et modules 2.5D
│   └── README.md         # Documentation spécifique à l'approche 2.5D
├── Approche2D/
│   ├── scripts/          # Scripts pour le pré-traitement, l'entraînement, et l'évaluation
│   ├── src/              # Code source des modèles 2D
│   └── README.md         # Documentation spécifique à l'approche 2D
└── README.md             # Documentation principale du projet
```

## Détails des Approches

### Approche 2D
- **Description :** Utilisation de coupes 2D individuelles des IRM pour la classification.
- **Emplacement :** `Approche2D/`
- **Contenu :**
  - `scripts/` : Scripts pour le pré-traitement et l'évaluation.
  - `src/` : Code source des modèles convolutifs 2D.

### Approche 2.5D
- **Description :** Analyse combinée de trois coupes consécutives pour capturer une perspective intermédiaire entre 2D et 3D.
- **Emplacement :** `Approche25D/`
- **Contenu :**
  - `data/` : Données nécessaires pour l'entraînement et la validation.
  - `network/` : Définition des architectures.
  - `src/` : Codes d'executions.

### Approche 3D
- **Description :** Analyse complète en 3D des IRM en utilisant des volumes tridimensionnels.
- **Emplacement :** `Approche3D/`
- **Contenu :**
  - `scripts/` : Scripts pour le traitement, l'entraînement et le test.
  - `src/` : Code source des modèles 3D.
  - `demo.gif` : Démonstration visuelle du fonctionnement du modèle.
  - `requirements.txt` : Liste des dépendances nécessaires.

## Instructions d'installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/votre-repository.git
   cd votre-repository
   ```
2. Installez les dépendances pour chaque approche :
   ```bash
   pip install -r Approche.D/requirements.txt
   ```

## Utilisation

Chaque approche est documentée dans son propre fichier `README.md`.
