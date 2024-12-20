# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ThreeLayer3DCNN: A simple 3D CNN model with three convolutional layers and a fully connected layer.
class ThreeLayer3DCNN(nn.Module):
    def __init__(self, dropout_p=0.5):
        super(ThreeLayer3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)

        self.pool = nn.MaxPool3d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((6, 8, 8))
        self.dropout = nn.Dropout(dropout_p)

        self.fc = nn.Linear(64 * 6 * 8 * 8, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        return self.fc(x)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FourLayer3DCNN: A simple 3D CNN model with four convolutional layers for better feature extraction
class FourLayer3DCNN(nn.Module):
    def __init__(self):
        super(FourLayer3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(128)
        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculer dynamiquement la taille de sortie
        temp_input = torch.zeros(1, 1, 64, 64, 64)  # Exemple d'entrée
        temp_output = self._forward_features(temp_input)
        self.fc = nn.Linear(temp_output.view(-1).size(0), 1)

    def _forward_features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        return self.fc(x)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FiveLayer3DCNN: A simple 3D CNN model with four convolutional layers for even better feature extraction
class FiveLayer3DCNN(nn.Module):
    def __init__(self, dropout_p=0.5, input_size=(1, 64, 64, 64)):
        super(FiveLayer3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(128)
        self.conv5 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(256)

        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout(dropout_p)

        C, D, H, W = input_size
        D_out = D // (2 ** 5)
        H_out = H // (2 ** 5)
        W_out = W // (2 ** 5)
        self.fc = nn.Linear(256 * D_out * H_out * W_out, 1)

    def forward(self, x):
        x = F.interpolate(x, size=(64, 64, 64), mode='trilinear', align_corners=False)
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        return self.fc(x)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ResNet3D: A 3D ResNet model designed with residual connections 
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=1, dropout_p=0.5):
        super(ResNet3D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ResNet3DWithAttention: A ResNet3D model enhanced with SE attention blocks
class SEBlock3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock3D, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1, 1)
        return x * y

class ResNet3DWithAttention(nn.Module):
    def __init__(self, block, layers, num_classes=1, dropout_p=0.5):
        super(ResNet3DWithAttention, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.attn1 = SEBlock3D(64)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.attn2 = SEBlock3D(128)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.attn3 = SEBlock3D(256)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.attn4 = SEBlock3D(512)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.attn1(self.layer1(x))
        x = self.attn2(self.layer2(x))
        x = self.attn3(self.layer3(x))
        x = self.attn4(self.layer4(x))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)