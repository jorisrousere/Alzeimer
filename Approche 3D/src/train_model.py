# train_model.py

import argparse
import os 
from datetime import datetime 
import torch
from torch.utils.data import DataLoader
from datasets import HippocampusDataset
from models import (
    ResNet3D, ResidualBlock3D, FourLayer3DCNN, ThreeLayer3DCNN,
    FiveLayer3DCNN, ResNet3DWithAttention
)
from train import train_with_evaluation
from plot_utils import plot_training_curves

def format_float(value, decimal_places=6):
    formatted = f"{value:.{decimal_places}f}".rstrip('0').rstrip('.')
    return formatted.replace('.', 'p')

parser = argparse.ArgumentParser(description="Train 3D CNN models on hippocampus data")
parser.add_argument('--model', type=str, required=True,
                    choices=["FourLayer3DCNN", "ThreeLayer3DCNN", "FiveLayer3DCNN", "ResNet3D", "ResNet3DWithAttention"],
                    help="Model to train")
parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate")
parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
parser.add_argument('--epochs', type=int, default=25, help="Number of epochs")
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay")
parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
args = parser.parse_args()

device = args.device
csv_path = "data/Hippocampe_dataset_mci_only_test/hippocampi_labels.csv"

print(f"Loading data...")
train_dataset = HippocampusDataset(csv_file=csv_path, split="train")
val_dataset = HippocampusDataset(csv_file=csv_path, split="val")

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

print(f"Initializing model {args.model}...")
if args.model == "FourLayer3DCNN":
    model = FourLayer3DCNN().to(device)
elif args.model == "ThreeLayer3DCNN":
    model = ThreeLayer3DCNN(dropout_p=args.dropout).to(device)
elif args.model == "FiveLayer3DCNN":
    model = FiveLayer3DCNN(dropout_p=args.dropout).to(device)
elif args.model == "ResNet3D":
    model = ResNet3D(ResidualBlock3D, [2, 2, 2, 2], dropout_p=args.dropout).to(device)
elif args.model == "ResNet3DWithAttention":
    model = ResNet3DWithAttention(ResidualBlock3D, [2, 2, 2, 2], dropout_p=args.dropout).to(device)
else:
    raise ValueError("Invalid model choice.")

print(f"Training {args.model} for {args.epochs} epochs...")
best_model_state, history = train_with_evaluation(
    model, train_loader, val_loader, num_epochs=args.epochs,
    lr=args.lr, weight_decay=args.weight_decay, device=device
)

models_dir = "models"
outputs_dir = "outputs"
os.makedirs(models_dir, exist_ok=True) 
os.makedirs(outputs_dir, exist_ok=True)  

model_name = args.model
dropout = f"dropout{format_float(args.dropout)}"
batch_size = f"batch{args.batch_size}"
epochs_str = f"epochs{args.epochs}"
lr_str = f"lr{format_float(args.lr)}"
weight_decay_str = f"wd{format_float(args.weight_decay)}"

model_filename = f"{model_name}_{dropout}_{batch_size}_{epochs_str}_{lr_str}_{weight_decay_str}.pth"
model_path = os.path.join(models_dir, model_filename)

print("Training complete. Saving best model...")
torch.save(best_model_state, model_path)
print(f"Best model saved to '{model_path}'.")

plot_filename = f"{model_name}_{dropout}_{batch_size}_{epochs_str}_{lr_str}_{weight_decay_str}_training_curves.png"
plot_path = os.path.join(outputs_dir, plot_filename)

plot_training_curves(history, save_path=plot_path)
print(f"Training curves saved to '{plot_path}'.")
