# test_models.py

import argparse
import os  # Pour gérer les chemins de fichiers
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from models import (
    ResNet3D, ResidualBlock3D, FourLayer3DCNN, ThreeLayer3DCNN, 
    FiveLayer3DCNN, ResNet3DWithAttention
)
from datasets import HippocampusDataset
from plot_utils import plot_confusion_matrix, plot_roc_curve

def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained 3D CNN model on test data.")
    parser.add_argument('--model', type=str, required=True, choices=[
        "FourLayer3DCNN", "ThreeLayer3DCNN", "FiveLayer3DCNN", 
        "ResNet3D", "ResNet3DWithAttention"
    ], help="Name of the model architecture")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate used during training")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for testing")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay used during training")  
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device: 'cpu' or 'cuda'")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model file")
    return parser.parse_args()

def format_float(value, decimal_places=6):
    formatted = f"{value:.{decimal_places}f}".rstrip('0').rstrip('.')
    return formatted.replace('.', 'p')

def load_model(model_name, dropout, device):
    if model_name == "FourLayer3DCNN":
        model = FourLayer3DCNN()
    elif model_name == "ThreeLayer3DCNN":
        model = ThreeLayer3DCNN(dropout_p=dropout)
    elif model_name == "FiveLayer3DCNN":
        model = FiveLayer3DCNN(dropout_p=dropout)
    elif model_name == "ResNet3D":
        model = ResNet3D(ResidualBlock3D, [2, 2, 2, 2], dropout_p=dropout)
    elif model_name == "ResNet3DWithAttention":
        model = ResNet3DWithAttention(ResidualBlock3D, [2, 2, 2, 2], dropout_p=dropout)
    else:
        raise ValueError("Invalid model name.")
    return model.to(device)

def evaluate_model(model, test_loader, device):
    model.eval() # set the model to evaluation mode
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad(): # disable gradient computation
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x).squeeze(1) # forward pass
            probs = torch.sigmoid(outputs) # apply sigmoid for probabilities
            preds = (probs > 0.5).long() # binarize predictions

            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs) # calculate AUC
    except ValueError:
        auc = 0.0  

    return all_labels, all_preds, all_probs, accuracy, precision, recall, f1, auc

args = parse_args()

device = args.device
csv_path = "data/Hippocampe_dataset_mci_only_test/hippocampi_labels.csv"

test_dataset = HippocampusDataset(csv_file=csv_path, split="test")
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

model = load_model(args.model, args.dropout, device)
try:
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
except TypeError:
    model.load_state_dict(torch.load(args.model_path, map_location=device))
print(f"Loaded model: {args.model} from {args.model_path}")

labels, preds, probs, acc, prec, rec, f1, auc = evaluate_model(model, test_loader, device)
print(f"Test Metrics:\n"
        f"Accuracy: {acc:.4f}\n"
        f"Precision: {prec:.4f}\n"
        f"Recall: {rec:.4f}\n"
        f"F1-Score: {f1:.4f}\n"
        f"AUC: {auc:.4f}")

outputs_dir = "outputs"
os.makedirs(outputs_dir, exist_ok=True) 

model_name = args.model
dropout = f"dropout{format_float(args.dropout)}"
batch_size = f"batch{args.batch_size}"
weight_decay_str = f"wd{format_float(args.weight_decay)}"

cm_filename = f"{model_name}_{dropout}_{batch_size}_{weight_decay_str}_confusion_matrix.png"
cm_path = os.path.join(outputs_dir, cm_filename)

roc_filename = f"{model_name}_{dropout}_{batch_size}_{weight_decay_str}_roc_curve.png"
roc_path = os.path.join(outputs_dir, roc_filename)

plot_confusion_matrix(labels, preds, save_path=cm_path)
print(f"Confusion matrix saved to '{cm_path}'.")

plot_roc_curve(labels, probs, save_path=roc_path)
print(f"ROC curve saved to '{roc_path}'.")


