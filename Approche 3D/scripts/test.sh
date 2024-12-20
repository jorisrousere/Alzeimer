#!/bin/bash

MODEL_DIR="models"
MODEL_NAME=$1

if [ -z "$MODEL_NAME" ]; then
    echo "Usage: ./test.sh <ModelName>"
    echo "Example: ./test.sh ThreeLayer3DCNN"
    exit 1
fi

MODEL_FILE=$(ls $MODEL_DIR/${MODEL_NAME}_*.pth 2>/dev/null | head -n 1)

if [ -z "$MODEL_FILE" ]; then
    echo "No model file found in $MODEL_DIR matching the name ${MODEL_NAME}. Please ensure the model exists."
    exit 1
fi

echo "Testing model: $MODEL_FILE"
python src/test_models.py --model $MODEL_NAME --model_path $MODEL_FILE
