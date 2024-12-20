#!/bin/bash
## Training script for ThreeLayer3DCNN

#python src/train_model.py \
#  --model "ThreeLayer3DCNN" \
#  --dropout 0.6 \
#  --batch_size 32 \
#  --epochs 3 \
#  --lr 0.0005

# Training script for FourLayer3DCNN
# python src/train_model.py \
#   --model "ThreeLayer3DCNN" \
#   --dropout 0.5 \
#   --batch_size 16 \
#   --epochs 30 \
#   --lr 0.0001

#--------------------------------------------------------------------------------------------------------------------------------------------
## Training script for FourLayer3DCNN

# python src/train_model.py \
#   --model "FourLayer3DCNN" \
#   --dropout 0.5 \
#   --batch_size 16 \
#   --epochs 5 \
#   --lr 0.0001

# python src/train_model.py \
#   --model "FourLayer3DCNN" \
#   --dropout 0.5 \
#   --batch_size 8 \
#   --epochs 30 \
#   --lr 0.0001

# python src/train_model.py \
#   --model "FourLayer3DCNN" \
#   --dropout 0.5 \
#   --batch_size 16 \
#   --epochs 30 \
#   --lr 0.0001

# python src/train_model.py \
#   --model "FourLayer3DCNN" \
#   --dropout 0.5 \
#   --batch_size 32 \
#   --epochs 30 \
#   --lr 0.0001

# python src/train_model.py \
#   --model "FourLayer3DCNN" \
#   --dropout 0.5 \
#   --batch_size 32 \
#   --epochs 15 \
#   --lr 0.0001

#--------------------------------------------------------------------------------------------------------------------------------------------
## Training script for FiveLayer3DCNN 

# python src/train_model.py \
#   --model "FiveLayer3DCNN" \
#   --dropout 0.4 \
#   --batch_size 8 \
#   --epochs 25 \
#   --lr 0.00005

# python src/train_model.py \
#   --model "FiveLayer3DCNN" \
#   --dropout 0.6 \
#   --batch_size 8 \
#   --epochs 25 \
#   --lr 0.00005

# python src/train_model.py \
#   --model "FiveLayer3DCNN" \
#   --dropout 0.6 \
#   --batch_size 16 \
#   --epochs 25 \
#   --lr 0.00005

#--------------------------------------------------------------------------------------------------------------------------------------------
## Training script for ResNet3D 

python src/train_model.py \
   --model "ResNet3D" \
   --dropout 0.5 \
   --batch_size 16 \
   --epochs 12 \
   --lr 0.0005

# python src/train_model.py \
#   --model "ResNet3D" \
#   --dropout 0.6 \
#   --batch_size 16 \
#   --epochs 20 \
#   --lr 0.0001

#--------------------------------------------------------------------------------------------------------------------------------------------
## Training script for ResNet3DWithAttention

# python src/train_model.py \
#   --model "ResNet3DWithAttention" \
#   --dropout 0.5 \
#   --batch_size 16 \
#   --epochs 15 \
#   --lr 0.00005

# python src/train_model.py \
#   --model "ResNet3DWithAttention" \
#   --dropout 0.5 \
#   --batch_size 32 \
#   --epochs 15 \
#   --lr 0.00005

  
