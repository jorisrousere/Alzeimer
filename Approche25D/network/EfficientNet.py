import torch
import torch.nn as nn
from torchvision.models.efficientnet import efficientnet_b0
from torchvision import transforms

class EfficientNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, device='cuda'):
        super(EfficientNet, self).__init__()

        # EfficientNetB0 
        self.effnet = efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')

        self.device = device
        self.effnet.to(self.device)

        # Replace the first convolution layer to match input channels
        # The original model assumes input channels = 3 (RGB images), so we replace it with a custom one
        self.effnet.features[0][0] = nn.Conv2d(
            in_channels, 
            self.effnet.features[0][0].out_channels, 
            kernel_size=self.effnet.features[0][0].kernel_size, 
            stride=self.effnet.features[0][0].stride, 
            padding=self.effnet.features[0][0].padding,
            bias=False
        )

        # Modify the classifier head to match the number of output classes
        num_features = self.effnet.classifier[1].in_features
        self.effnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, out_channels)
        )

        self.out_channels = out_channels

    def forward(self, x):
        
        x = self.effnet(x)

        return x
