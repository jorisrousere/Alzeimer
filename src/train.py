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
import numpy as np

from PIL import Image, ImageDraw, ImageFont  # For creating the legend


def apply_colormap(predictions, num_classes):
    # Define the colormap for segmentation classes
    colormap = np.array([
        [0, 0, 0],        # Background: Black
        [0, 255, 0],      # CN: Green
        [255, 0, 0],      # AD: Red
        [255, 255, 0],    # MCI stable: Yellow
        [255, 165, 0]     # MCI not stable: Orange
    ], dtype=np.uint8)

    predictions_colored = colormap[predictions]  # Map predictions to RGB colors
    return predictions_colored


def create_legend():
    colormap = {
        "Background (0)": (0, 0, 0),
        "CN (1)": (0, 255, 0),
        "AD (2)": (255, 0, 0),
        "MCI Stable (3)": (255, 255, 0),
        "MCI Not Stable (4)": (255, 165, 0),
    }
    legend_height = 50 * len(colormap)
    legend_width = 300
    legend_image = Image.new("RGB", (legend_width, legend_height), color="white")
    draw = ImageDraw.Draw(legend_image)

    y_offset = 0
    for label, color in colormap.items():
        draw.rectangle([10, y_offset + 10, 40, y_offset + 40], fill=color, outline="black")
        draw.text((50, y_offset + 10), label, fill="black")
        y_offset += 50

    legend_np = np.array(legend_image)
    return legend_np


def validate(model, val_loader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    predictions, ground_truths = [], []

    with torch.no_grad():
        for scans, masks in tqdm(val_loader, desc="Validation"):
            scans, masks = scans.to(device), masks.to(device)
            outputs = model(scans)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            masks = masks.cpu().numpy()

            predictions.extend(preds)
            ground_truths.extend(masks)

    avg_loss = running_loss / len(val_loader)

    # Convert predictions and masks to RGB for TensorBoard visualization
    preds_colored = [apply_colormap(predictions[i], num_classes) for i in range(0,len(predictions),len(predictions)//100)]
    masks_colored = [apply_colormap(ground_truths[i], num_classes) for i in range(0,len(ground_truths),len(ground_truths)//100)]

    preds_tensor = torch.tensor(np.stack(preds_colored)).permute(0, 3, 1, 2) / 255.0
    masks_tensor = torch.tensor(np.stack(masks_colored)).permute(0, 3, 1, 2) / 255.0

    return avg_loss, preds_tensor, masks_tensor


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    writer = SummaryWriter(log_dir=args.log_dir)

    legend_np = create_legend()
    legend_tensor = torch.tensor(legend_np).permute(2, 0, 1).float() / 255.0
    writer.add_image("Legend", legend_tensor)

    train_dataset = MedicalImageDataset(os.path.join(args.data_path, "train"), args.csv_path)
    val_dataset = MedicalImageDataset(os.path.join(args.data_path, "val"), args.csv_path)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = UNet(in_channels=1, out_channels=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (scans, masks) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        ):
            scans, masks = scans.to(device), masks.to(device)
            outputs = model(scans)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % args.log_freq == 0:
                avg_loss = running_loss / (batch_idx + 1)
                writer.add_scalar("Loss/train", avg_loss, epoch * len(train_loader) + batch_idx)

        avg_train_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/epoch_train", avg_train_loss, epoch + 1)

        val_loss, val_preds_tensor, val_masks_tensor = validate(
            model, val_loader, criterion, device, args.num_classes
        )
        writer.add_scalar("Loss/epoch_val", val_loss, epoch + 1)

        writer.add_images("Validation/Predictions", val_preds_tensor, epoch + 1)
        writer.add_images("Validation/Ground Truth", val_masks_tensor, epoch + 1)

        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.save_dir, f"unet_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

    print("Training complete!")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a U-Net for medical image segmentation")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save TensorBoard logs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of segmentation classes")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--save_freq", type=int, default=5, help="Frequency to save model checkpoints")
    parser.add_argument("--log_freq", type=int, default=25, help="Frequency to log metrics and images to TensorBoard")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    train(args)
