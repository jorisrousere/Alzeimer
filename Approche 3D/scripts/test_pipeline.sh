#!/bin/bash

MODEL_DIR="models"

MODEL_FILE=$(ls $MODEL_DIR/*.pth | head -n 1)

if [ -z "$MODEL_FILE" ]; then
    echo "No model file found in $MODEL_DIR. Please train a model first."
    exit 1
fi

MODEL_NAME=$(basename "$MODEL_FILE" | cut -d'_' -f1)

case $MODEL_NAME in
    "ThreeLayer3DCNN")
        MODEL_CLASS="ThreeLayer3DCNN"
        ;;
    "FourLayer3DCNN")
        MODEL_CLASS="FourLayer3DCNN"
        ;;
    "FiveLayer3DCNN")
        MODEL_CLASS="FiveLayer3DCNN"
        ;;
    "ResNet3D")
        MODEL_CLASS="ResNet3D"
        ;;
    "ResNet3DWithAttention")
        MODEL_CLASS="ResNet3DWithAttention"
        ;;
    *)
        echo "Unknown model name: $MODEL_NAME"
        exit 1
        ;;
esac

echo "Testing model: $MODEL_FILE"
python src/test_models.py --model $MODEL_CLASS --model_path $MODEL_FILE
