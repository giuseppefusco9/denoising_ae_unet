import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Doppia Convoluzione (Conv -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNetDenoiseAttack(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # --- ENCODER (Downsampling) ---
        self.inc   = DoubleConv(in_channels, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16,  32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32,  64))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64,  128))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # --- BOTTLENECK ---
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # --- DECODER (Upsampling con Skip Connections) ---
        self.up1      = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)

        self.up2      = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)

        self.up3      = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)

        self.up4      = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(64, 32)

        self.up5      = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_up5 = DoubleConv(32, 16)

        self.outc = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x, detector=None):
        # --- Encoder ---
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        b  = self.bottleneck(x5)

        # --- Decoder con skip connections ---
        t1 = self.conv_up1(torch.cat([self.up1(b),  x5], dim=1))
        t2 = self.conv_up2(torch.cat([self.up2(t1), x4], dim=1))
        t3 = self.conv_up3(torch.cat([self.up3(t2), x3], dim=1))
        t4 = self.conv_up4(torch.cat([self.up4(t3), x2], dim=1))
        t5 = self.conv_up5(torch.cat([self.up5(t4), x1], dim=1))

        reconstructed_imgs = (torch.tanh(self.outc(t5)) + 1) / 2

        if detector is not None:
            detector_outputs    = detector.detect(reconstructed_imgs)
            detected_bit_logits = detector_outputs["preds"][:, 1:]
            return reconstructed_imgs, detected_bit_logits

        return reconstructed_imgs