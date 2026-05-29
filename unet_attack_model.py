import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # Encoder (Downsampling) - Lavora sulla Luminanza (1 canale)
        self.inc = DoubleConv(1, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # Decoder (Upsampling) con Skip Connections
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256) # 256 (up) + 256 (skip) = 512
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128) # 128 (up) + 128 (skip) = 256
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)   # 64 (up) + 64 (skip) = 128
        
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(64, 32)     # 32 (up) + 32 (skip) = 64
        
        self.up5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_up5 = DoubleConv(32, 16)     # 16 (up) + 16 (skip) = 32
        
        # Output finale della sola luminanza (1 canale)
        self.outc = nn.Conv2d(16, 1, kernel_size=1)

    def rgb_to_ycbcr(self, rgb):
        # Matrice di conversione formale standard BT.601
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 0.5
        cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 0.5
        return y, cb, cr

    def ycbcr_to_rgb(self, y, cb, cr):
        # Matrice di conversione inversa formale
        r = y + 1.402 * (cr - 0.5)
        g = y - 0.34414 * (cb - 0.5) - 0.71414 * (cr - 0.5)
        b = y + 1.772 * (cb - 0.5)
        return torch.cat([r, g, b], dim=1)

    def forward(self, x):
        # 1. Riceve RGB (3, 256, 256) ed estrae i componenti
        y, cb, cr = self.rgb_to_ycbcr(x)
        
        # 2. Encoder sulla Luminanza (Salataggio delle skip connections)
        x1 = self.inc(y)        # Canali: 16
        x2 = self.down1(x1)     # Canali: 32
        x3 = self.down2(x2)     # Canali: 64
        x4 = self.down3(x3)     # Canali: 128
        x5 = self.down4(x4)     # Canali: 256
        
        # Bottleneck
        b = self.bottleneck(x5) # Canali: 512
        
        # 3. Decoder con Concatetazione Esplicita delle SKIP CONNECTIONS
        # Layer 1
        t1 = self.up1(b)
        t1 = torch.cat([t1, x5], dim=1) # Unisce l'up-sampling con il livello 256 dell'encoder
        t1 = self.conv_up1(t1)
        
        # Layer 2
        t2 = self.up2(t1)
        t2 = torch.cat([t2, x4], dim=1) # Unisce con il livello 128
        t2 = self.conv_up2(t2)
        
        # Layer 3
        t3 = self.up3(t2)
        t3 = torch.cat([t3, x3], dim=1) # Unisce con il livello 64
        t3 = self.conv_up3(t3)
        
        # Layer 4
        t4 = self.up4(t3)
        t4 = torch.cat([t4, x2], dim=1) # Unisce con il livello 32
        t4 = self.conv_up4(t4)
        
        # Layer 5
        t5 = self.up5(t4)
        t5 = torch.cat([t5, x1], dim=1) # Unisce con il livello 16
        t5 = self.conv_up5(t5)
        
        # Convoluzione finale per ricreare la sola luminanza modificata
        y_reconstructed = self.outc(t5) # Volume (1, 256, 256)
        
        # 4. Riassemblaggio finale in RGB usando i canali cromatici originari immutati
        rgb_output = self.ycbcr_to_rgb(y_reconstructed, cb, cr)
        
        return rgb_output # Volume CORRETTO (3, 256, 256)