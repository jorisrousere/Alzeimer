import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(CNN, self).__init__()

        # CNN for binary classification
        self.cnn_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(16384, 256),  # Assuming input size of 128x128
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),  # Assuming input size of 128x128
            nn.ReLU(inplace=True),
            nn.Linear(128, out_channels)
        )

        self.encoder = nn.Sequential(
            self.conv_block(in_channels, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512),
            self.conv_block(512, 1024)
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, stride=2)
        )

    def forward(self, x):

        output_encoder = self.encoder(x)
    
        output = self.cnn_classifier(output_encoder)
        return output