
import torch
import torch.nn as nn
import timm
from torchvision import transforms

class SwinTransformer(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, device='cuda'):
        super(SwinTransformer, self).__init__()

        # Swin Transformer
        self.swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True)

        self.device = device
        self.swin.to(self.device)

        # Replace the first convolution layer to match input channels
        # The original model assumes input channels = 3 (RGB images), so we replace it with a custom one
        self.swin.patch_embed.proj = nn.Conv2d(in_channels, self.swin.embed_dim, kernel_size=(3, 3), stride=(1, 1))

        # Modify the classifier head to match the number of output classes
        self.swin.head = nn.Linear(self.swin.head.in_features, out_channels)

        # Freeze all layers in the Swin Transformer
        for param in self.swin.parameters():
            param.requires_grad = False

        # Add custom dense layers for classification
        self.fc1 = nn.Linear(1568, 512)
        self.fc2 = nn.Linear(512, out_channels)

        # Define a resize transformation to ensure the input image size matches (224x224)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224
        ])

        self.out_channels = out_channels

    def forward(self, x):

        # Apply the transformation to resize 
        x = self.transform(x)

        # Swin Transformer model
        x = self.swin(x)

        x = nn.Flatten()(x)

        # Classifier
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        
        return x
