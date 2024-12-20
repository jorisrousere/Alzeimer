#!/bin/bash

# Définir les variables
DATA_PATH="datasets/split_data"
CSV_PATH="datasets/split_data/train/list_standardized_tongtong_2017.csv"
SAVE_DIR="checkpoints"
LOG_DIR="logs"
BATCH_SIZE=16
EPOCHS=20
LEARNING_RATE=0.001
NUM_WORKERS=4

# Créer les répertoires de sauvegarde si nécessaire
mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

# Lancer l'entraînement
python3 src/train.py \
    --data_path "$DATA_PATH" \
    --csv_path "$CSV_PATH" \
    --save_dir "$SAVE_DIR" \
    --log_dir "$LOG_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LEARNING_RATE" \
    --num_workers "$NUM_WORKERS"
