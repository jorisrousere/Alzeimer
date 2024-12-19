import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data.data_hippocampe import HippocampusDataset
from network.binary_cnn import BinaryCNN

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Charger les datasets
    train_dataset = HippocampusDataset(
        folder_path=os.path.join(args.data_path, "train"),
        csv_file=args.csv_path,
        target_size=(256, 256),
        threshold=0.15
    )
    val_dataset = HippocampusDataset(
        folder_path=os.path.join(args.data_path, "val"),
        csv_file=args.csv_path,
        target_size=(256, 256),
        threshold=0.15
    )

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialisation du modèle
    model = BinaryCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()  # BCEWithLogitsLoss pour utiliser les logits directement
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=3, verbose=True
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)

    best_val_accuracy = 0.0
    for epoch in range(args.epochs):
        # Mode entraînement
        model.train()
        running_loss = 0.0
        total_correct = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device).float()

            # Forward
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistiques
            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()  # Applique Sigmoid pour les prédictions
            total_correct += (preds == labels).sum().item()

        # Calcul des métriques d'entraînement
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = total_correct / len(train_loader.dataset)
        writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch + 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / len(val_loader.dataset)
        writer.add_scalar("Loss/val", avg_val_loss, epoch + 1)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch + 1)

        # Scheduler update
        scheduler.step(val_accuracy)

        # Affichage des résultats
        print(f"Epoch {epoch + 1}/{args.epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Sauvegarde du meilleur modèle
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at {best_model_path}")

        # Sauvegarde périodique du modèle
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.save_dir, f"binary_cnn_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

    print("Training complete!")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a binary CNN on hippocampus slices")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save TensorBoard logs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--save_freq", type=int, default=1, help="Frequency to save model checkpoints")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    train(args)
