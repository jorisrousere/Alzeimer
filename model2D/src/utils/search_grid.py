import os
import itertools
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loader_train import HippocampusDataset

class CustomCNN(nn.Module):
    def __init__(self, conv_layers_config, dense_layers_config, input_size=(256, 256)):
        super(CustomCNN, self).__init__()
        self.conv_layers = nn.Sequential()
        in_channels = 1

        # Ajout des couches de convolution
        for i, (out_channels, kernel_size) in enumerate(conv_layers_config):
            self.conv_layers.add_module(f"conv_{i}", nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2))
            self.conv_layers.add_module(f"bn_{i}", nn.BatchNorm2d(out_channels))
            self.conv_layers.add_module(f"relu_{i}", nn.ReLU())
            self.conv_layers.add_module(f"pool_{i}", nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.flatten = nn.Flatten()

        # Calcul de la taille de la sortie des couches de convolution
        dummy_input = torch.zeros(1, 1, *input_size)
        conv_output_size = self._get_conv_output_size(dummy_input)

        # Ajout des couches denses
        self.fc_layers = nn.Sequential()
        for i, neurons in enumerate(dense_layers_config):
            if i == 0:
                self.fc_layers.add_module(f"fc_{i}", nn.Linear(conv_output_size, neurons))
            else:
                self.fc_layers.add_module(f"fc_{i}", nn.Linear(dense_layers_config[i - 1], neurons))
            self.fc_layers.add_module(f"dropout_{i}", nn.Dropout(0.5))
            self.fc_layers.add_module(f"relu_{i}", nn.ReLU())
        self.fc_layers.add_module("output", nn.Linear(dense_layers_config[-1], 1))

    def _get_conv_output_size(self, x):
        x = self.conv_layers(x)
        return x.numel()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

def train_and_evaluate(model, train_loader, val_loader, device, epochs=10, lr=1e-3):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_correct += (preds == labels).sum().item()

        train_accuracy = total_correct / len(train_loader.dataset)

        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images).squeeze()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()

        val_accuracy = val_correct / len(val_loader.dataset)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

    return best_val_accuracy

def grid_search(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Chargement des datasets
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Hyperparamètres pour la recherche par grille
    num_conv_layers_options = [2, 3, 4, 5]
    num_dense_layers_options = [2, 3, 4]
    filters_options = [32, 64, 128, 256]
    kernel_sizes_options = [3, 5, 7]
    dense_neurons_options = [64, 128, 256, 512]

    results = []

    # Grid search sur les combinaisons
    for num_conv, num_dense in itertools.product(num_conv_layers_options, num_dense_layers_options):
        conv_config = [(filters_options[i % len(filters_options)], kernel_sizes_options[i % len(kernel_sizes_options)]) for i in range(num_conv)]
        dense_config = [dense_neurons_options[i % len(dense_neurons_options)] for i in range(num_dense)]
        print(f"Testing configuration: Convolutional Layers: {conv_config}, Dense Layers: {dense_config}")
        model = CustomCNN(conv_config, dense_config)
        val_accuracy = train_and_evaluate(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)
        results.append((conv_config, dense_config, val_accuracy))
        print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Sauvegarde des résultats
    results_path = os.path.join(args.save_dir, "grid_search_results.txt")
    with open(results_path, "w") as f:
        for conv_config, dense_config, val_acc in results:
            f.write(f"Conv: {conv_config}, Dense: {dense_config}, Val Acc: {val_acc:.4f}\n")
    print(f"Grid search complete. Results saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Search for CNN Architectures")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--save_dir", type=str, default="./grid_search_results", help="Directory to save grid search results")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs per configuration")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    grid_search(args)
