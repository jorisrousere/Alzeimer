# Approche 2.5D - Network

## Structure du dossier

```
Approche25D/network/
├── CNN.py
├── EfficientNet.py
├── README.md
└── swinTransformer.py
```

Ce dossier contient les implémentations des différents réseaux de neurones utilisés pour la classification d'IRM cérébrales dans le cadre de la détection de la maladie d'Alzheimer.

---

## Description des fichiers

### 1. **CNN.py**
Ce fichier contient l'implémentation d'un réseau convolutif simple (CNN) avec une structure hiérarchique composée de plusieurs blocs de convolution et un classificateur fully connected.

- **`conv_block`** : Bloc de convolution comprenant deux couches convolutives, une activation ReLU et un MaxPooling.
- **`encoder`** : Partie convolutive pour l'extraction des caractéristiques.
- **`cnn_classifier`** : Classificateur fully connected prenant les caractéristiques extraites pour prédire la classe de sortie.
- **Entrées** : Images 2.5D avec 3 canaux (slices d'IRM).
- **Sorties** : Deux classes (par défaut), pour distinguer les IRM "normales" des IRM présentant des signes de la maladie.

### 2. **EfficientNet.py**
Ce fichier implémente une architecture basée sur EfficientNet-B0, un modèle pré-entraîné disponible via PyTorch.

- **Personnalisation** :
  - Modification de la couche d'entrée pour accepter des images avec un nombre variable de canaux (par défaut : 3).
  - Modification de la tête de classification pour correspondre au nombre de classes souhaitées.
- **Dropout** : Utilisé pour éviter le surapprentissage.
- **Pré-entraînement** : Le modèle est initialisé avec les poids pré-entraînés disponibles pour EfficientNet-B0.
- **Entrées** : Images redimensionnées automatiquement pour s'adapter à l'architecture EfficientNet (par défaut : 224x224).
- **Sorties** : Deux classes (par défaut), pour distinguer les IRM "normales" des IRM présentant des signes de la maladie.

### 3. **swinTransformer.py**
Ce fichier implémente une architecture basée sur Swin Transformer, un modèle de vision puissant qui exploite une structure de patchs et d'attention multi-niveaux.

- **Personnalisation** :
  - Remplacement de la première couche de convolution pour accepter un nombre variable de canaux (par défaut : 3).
  - Modification de la tête de classification avec des couches fully connected additionnelles pour la classification.
- **Transformation des entrées** : Les images sont redimensionnées à 224x224 pour correspondre à l'architecture.
- **Congélation des couches** : Toutes les couches du Swin Transformer sont gelées pour ne pas être entraînées. Seules les couches personnalisées ajoutées sont optimisées.
- **Entrées** : Images 2.5D redimensionnées (3 canaux).
- **Sorties** : Deux classes (par défaut), par exemple "normale" ou "malade".

---

## Utilisation

Ces architectures peuvent être utilisées indépendamment en fonction des besoins et des ressources disponibles. Elles sont conçues pour s'intégrer avec des DataLoaders générant des images 2.5D pour entraîner des modèles de classification des IRM cérébrales.

