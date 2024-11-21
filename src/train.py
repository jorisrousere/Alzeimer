import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # TensorBoard integration
from torchvision.utils import make_grid
from data.medicalimagedataset import MedicalImageDataset  # Replace with the correct path to your dataset class
from network.unet import UNet  # Replace with the correct path to your U-Net class


def train(args):
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Dataset and DataLoader
    print("Loading dataset...")
    train_dataset = MedicalImageDataset(args.data_path, args.csv_path)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    # Model, Loss, Optimizer
    print("Initializing model...")
    model = UNet(in_channels=1, out_channels=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()  # Suitable for multi-class segmentation
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (scans, masks) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        ):
            scans, masks = scans.to(device), masks.to(device)

            # Forward pass
            outputs = model(scans)
            loss = criterion(outputs, masks)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Log metrics and images to TensorBoard every log_freq batches
            if batch_idx % args.log_freq == 0:
                avg_loss = running_loss / (batch_idx + 1)
                writer.add_scalar("Loss/train", avg_loss, epoch * len(train_loader) + batch_idx)

                # Visualize predictions (on a small batch)
                preds = torch.argmax(outputs, dim=1)  # Convert logits to class predictions
                # Add batch dimension for TensorBoard compatibility
                preds_with_batch = preds.unsqueeze(1).float()  # Shape: (batch_size, 1, H, W)
                masks_with_batch = masks.unsqueeze(1).float()  # Shape: (batch_size, 1, H, W)

                # Log predictions and ground truths
                writer.add_images("Predictions", preds_with_batch, epoch * len(train_loader) + batch_idx)
                writer.add_images("Ground Truth", masks_with_batch, epoch * len(train_loader) + batch_idx)


        # Log average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/epoch", avg_loss, epoch + 1)

        # Save the model checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.save_dir, f"unet_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

    print("Training complete!")
    writer.close()


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Train a U-Net for medical image segmentation")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save TensorBoard logs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of segmentation classes")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--save_freq", type=int, default=5, help="Frequency to save model checkpoints")
    parser.add_argument("--log_freq", type=int, default=10, help="Frequency to log metrics and images to TensorBoard")

    args = parser.parse_args()

    # Ensure the save and log directories exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Start training
    train(args)
