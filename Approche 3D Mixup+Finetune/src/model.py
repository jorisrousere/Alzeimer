import torch
import torch.nn as nn
import torch.nn.functional as F

class Fixed4Layer3DCNN(nn.Module):
    def __init__(self):
        super(Fixed4Layer3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)  # BatchNorm après conv1
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)  # BatchNorm après conv2
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)  # BatchNorm après conv3
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(128)  # BatchNorm après conv4

        self.pool = nn.MaxPool3d(2)  # MaxPooling
        self.dropout = nn.Dropout(0.5)  # Dropout pour éviter le surapprentissage
        self.fc = nn.Linear(128 * 2 * 2 * 2, 1)  # Fully connected

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> BatchNorm1 -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> BatchNorm2 -> ReLU -> Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv3 -> BatchNorm3 -> ReLU -> Pool
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # Conv4 -> BatchNorm4 -> ReLU -> Pool
        x = torch.flatten(x, start_dim=1)  # Flatten pour passer dans la couche dense
        x = self.dropout(x)  # Dropout
        x = self.fc(x)  # Fully connected
        return x

class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)


    def generate_cam(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        self.model.zero_grad()
        
        if class_idx is not None:
            output = output[:, class_idx]
        output.backward()

        weights = torch.mean(self.gradients, dim=(2, 3, 4))  # Pas de abs()
        cam = torch.sum(weights[:, :, None, None, None] * self.activations, dim=1)
        cam = F.relu(cam)  # Pas de abs()

        if cam.max() != cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

class ResidualBlock3D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm3d(out_channels)
        self.conv2 = torch.nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm3d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class ResNet3D(torch.nn.Module):
    def __init__(self, block, layers, num_classes=1, dropout_p=0.5):
        super(ResNet3D, self).__init__()
        self.in_channels = 64
        self.conv1 = torch.nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm3d(64)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = torch.nn.Dropout(dropout_p)
        self.fc = torch.nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = torch.nn.Sequential(
                torch.nn.Conv3d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm3d(out_channels)
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return torch.nn.Sequential(*layers)

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
    