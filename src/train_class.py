import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from data.medicalclassdataset import MedicalClassDataset  # Adjust for your dataset class
from network.unet_class_2 import UNet # Replace with your classification model
import numpy as np
import torchvision

def validate(model, val_loader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    total_correct = 0
    class_correct = np.zeros(num_classes, dtype=np.int64)
    class_total = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for scans, labels in tqdm(val_loader, desc="Validation"):
            scans, labels = scans.to(device), labels.to(device)
            outputs = model(scans)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()

            # Update per-class metrics
            for class_idx in range(num_classes):
                class_correct[class_idx] += ((preds == class_idx) & (labels == class_idx)).sum().item()
                class_total[class_idx] += (labels == class_idx).sum().item()

    avg_loss = running_loss / len(val_loader)
    overall_accuracy = total_correct / len(val_loader.dataset)
    class_accuracies = class_correct / np.maximum(class_total, 1)  # Avoid division by zero
    return avg_loss, overall_accuracy, class_accuracies


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    classes = ["CN", "AD", "MCI stable", "MCI not stable"]
    writer = SummaryWriter(log_dir=args.log_dir)

    train_dataset = MedicalClassDataset(os.path.join(args.data_path, "train"), args.csv_path)
    val_dataset = MedicalClassDataset(os.path.join(args.data_path, "val"), args.csv_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = UNet(in_channels=1, out_channels=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        total_correct = 0

        # Per-class metrics for training
        class_correct_train = np.zeros(args.num_classes, dtype=np.int64)
        class_total_train = np.zeros(args.num_classes, dtype=np.int64)

        for batch_idx, (scans, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")):
            scans, labels = scans.to(device), labels.to(device)
            outputs = model(scans)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Log batch loss
            if (batch_idx + 1) % args.log_freq == 0:
                batch_loss = running_loss / (batch_idx + 1)
                writer.add_scalar("Loss/batch_train", batch_loss, epoch * len(train_loader) + batch_idx + 1)
                # feature_map = model.output_unet 
                # # Normalize feature map for visualization
                # feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-5)
                # grid = torchvision.utils.make_grid(feature_map[:1], nrow=8, normalize=True, scale_each=True)
                
                # writer.add_image(f"Output/U-Net", grid, epoch + 1)
            # Calculate training accuracy
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()

            for class_idx in range(args.num_classes):
                class_correct_train[class_idx] += ((preds == class_idx) & (labels == class_idx)).sum().item()
                class_total_train[class_idx] += (labels == class_idx).sum().item()


        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = total_correct / len(train_loader.dataset)
        train_class_accuracies = class_correct_train / np.maximum(class_total_train, 1)

        writer.add_scalar("Loss/epoch_train", avg_train_loss, epoch + 1)
        writer.add_scalar("Accuracy/epoch_train", train_accuracy, epoch + 1)

        # Log per-class training accuracies
        for class_idx, class_acc in enumerate(train_class_accuracies):
            writer.add_scalar(f"Accuracy/train_class_{classes[class_idx]}", class_acc, epoch + 1)

        val_loss, val_accuracy, val_class_accuracies = validate(model, val_loader, criterion, device, args.num_classes)

        writer.add_scalar("Loss/epoch_val", val_loss, epoch + 1)
        writer.add_scalar("Accuracy/epoch_val", val_accuracy, epoch + 1)

        # Log per-class validation accuracies
        for class_idx, class_acc in enumerate(val_class_accuracies):
            writer.add_scalar(f"Accuracy/val_class_{classes[class_idx]}", class_acc, epoch + 1)

        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.save_dir, f"classification2_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

    print("Training complete!")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for medical image classification")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save TensorBoard logs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of classes for classification")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--save_freq", type=int, default=1, help="Frequency to save model checkpoints")
    parser.add_argument("--log_freq", type=int, default=10, help="Frequency to log batch loss during training")


    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    train(args)
