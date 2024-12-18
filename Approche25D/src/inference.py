import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from data.medicalclassdataset_valid import MedicalClassDataset  
from rendu.Alzeimer.Approche25D.network.CNN import CNN  

def load_model(model_path, num_classes, device):
    model = CNN(in_channels=3, out_channels=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate(model, test_loader, device):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for scans, labels in tqdm(test_loader, desc="Inference"):
            scans = scans.to(device)
            labels = labels.to(device)

            outputs = model(scans)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)

def save_confusion_matrix(cm, classes, save_dir, title="Confusion Matrix", normalize=True, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for medical image classification")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes for classification")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--save_dir", type=str, default="./results", help="Directory to save results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dataset = MedicalClassDataset(args.test_data_path, args.csv_path, apply_augmentation=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = load_model(args.model_path, args.num_classes, device)

    print("Evaluating the model...")
    preds, labels = evaluate(model, test_loader, device)

    acc = accuracy_score(labels, preds)
    print(f"Accuracy: {acc * 100:.2f}%")

    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(cm)

    classes = ["MCI stable", "MCI instable"]
    save_confusion_matrix(cm, classes, args.save_dir)
