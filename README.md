# Projet : Classification des IRM avec Approches 2D, 2.5D et 3D

Ce projet propose plusieurs approches pour analyser les images IRM en utilisant des architectures 2D, 2.5D et 3D afin de classifier les données. Chaque approche est structurée dans un répertoire séparé pour une meilleure organisation.

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
Détails des Approches
Approche 2D
Description : Utilisation de coupes 2D individuelles des IRM pour la classification.
Emplacement : Dossier Approche2D/.
Contenu :
Scripts pour l'entraînement et l'évaluation.
Code source des modèles convolutifs 2D.
Approche 2.5D
Description : Analyse combinée de trois coupes consécutives pour capturer une perspective intermédiaire entre 2D et 3D.
Emplacement : Dossier Approche25D/.
Contenu :
Répertoire data/ pour stocker les données d'entraînement et de validation.
Répertoire network/ pour les architectures de réseaux spécifiques à cette approche.
Approche 3D
Description : Analyse complète en 3D des IRM en utilisant des volumes tridimensionnels.
Emplacement : Dossier Approche3D/.
Contenu :
Démonstration animée du modèle en action dans demo.gif.
Code source des modèles et scripts associés.
Liste des dépendances requises dans requirements.txt.
Instructions d'installation
Clonez ce dépôt :
bash
Copier le code
git clone https://github.com/votre-repository.git
Installez les dépendances nécessaires pour chaque approche :
bash
Copier le code
pip install -r Approche3D/requirements.txt
Utilisation
Chaque approche est documentée dans son propre fichier README.md. Consultez ces fichiers pour des instructions détaillées sur l'entraînement, l'évaluation, et les tests.
