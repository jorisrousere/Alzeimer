import torch
import torch.nn as nn


class BinaryCNN(nn.Module):
    def __init__(self):
        super(BinaryCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()

        # Calcul dynamique de la taille d'entrée de fc1
        dummy_input = torch.zeros(1, 1, 256, 256)  # Batch size de 1, taille d'entrée (256, 256)
        conv_output_size = self._get_conv_output_size(dummy_input)
        self.fc1 = nn.Linear(conv_output_size, 512)

        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)

    def _get_conv_output_size(self, x):
        x = self.conv_layers(x)
        return x.numel()  # Nombre total d'éléments dans le tenseur

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)  # Pas de Sigmoid ici pour utiliser BCEWithLogitsLoss
        return x

