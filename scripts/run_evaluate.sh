#!/bin/bash

# Variables communes
MODEL_PATH="models/model2D.pth"
TARGET_SIZE="256 256"
THRESHOLD=0.15
BATCH_SIZE=16
NUM_WORKERS=4
SAVE_DIR="results"

# Évaluation sur AD/CN
DATA_PATH_ADCN="datasets/adcn"
CSV_PATH_ADCN="datasets/adcn/list_standardized.csv"
SAVE_DIR_ADCN="${SAVE_DIR}/adcn"
mkdir -p "$SAVE_DIR_ADCN"

echo "Évaluation sur le jeu de données AD/CN..."
python3 src/evaluate_adcn.py \
    --data_path "$DATA_PATH_ADCN" \
    --csv_path "$CSV_PATH_ADCN" \
    --model_path "$MODEL_PATH" \
    --target_size $TARGET_SIZE \
    --threshold "$THRESHOLD" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --save_dir "$SAVE_DIR_ADCN"

# Évaluation sur MCI
DATA_PATH_MCI="datasets/mci"
CSV_PATH_MCI="datasets/mci/list_standardized.csv"
SAVE_DIR_MCI="${SAVE_DIR}/mci"
mkdir -p "$SAVE_DIR_MCI"

echo "Évaluation sur le jeu de données MCI..."
python3 src/evaluate_mci.py \
    --data_path "$DATA_PATH_MCI" \
    --csv_path "$CSV_PATH_MCI" \
    --model_path "$MODEL_PATH" \
    --target_size $TARGET_SIZE \
    --threshold "$THRESHOLD" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --save_dir "$SAVE_DIR_MCI"
