import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=100):
        super(UNet, self).__init__()

        # Encoder (downsampling path)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder (upsampling path)
        self.upconv4 = self.upconv(1024, 512)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv(128, 64)
        self.dec1 = self.conv_block(128, 64)


        self.conv1 = nn.Conv2d(64, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 1, kernel_size=3)

        self.maxpool2d = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(0.3)
        self.output =  nn.Linear(128, out_channels)
        self.dense = nn.Linear(3844, 128)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Calculate padding
        orig_height, orig_width = x.size(2), x.size(3)
        target_height = ((orig_height - 1) // 16 + 1) * 16  # Make divisible by 16
        target_width = ((orig_width - 1) // 16 + 1) * 16    # Make divisible by 16

        pad_height = target_height - orig_height
        pad_width = target_width - orig_width

        x = nn.functional.pad(x, (0, pad_width, 0, pad_height))

        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.maxpool2d(enc1))
        enc3 = self.enc3(self.maxpool2d(enc2))
        enc4 = self.enc4(self.maxpool2d(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.maxpool2d(enc4))
 
        # Decoder
        dec4 = torch.cat((self.upconv4(bottleneck), enc4), dim=1)
        dec4 = self.dec4(dec4)
        dec3 = torch.cat((self.upconv3(dec4), enc3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = torch.cat((self.upconv2(dec3), enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = torch.cat((self.upconv1(dec2), enc1), dim=1)
        dec1 = self.dec1(dec1)

        
        x = self.conv1(dec1)
        x = self.maxpool2d(x)

        x = self.conv2(x)
        x = self.maxpool2d(x)

        x = self.flatten(x)

        x = self.dense(x)

        x = self.dropout(x)

        output = self.output(x)

        return output
