import torch
import torch.nn as nn

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        c = self.conv(x)
        p = self.pool(c)
        return c, p

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        # 1. Fa l'upsampling geometrico (raddoppia H e W)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        # 2. Correzione: kernel_size=3 e padding=1 conserva la dimensione spaziale (es. 32x32 rimane 32x32)
        self.conv_trans = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) 
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = self.conv_trans(x)
        
        # Concatenazione perfetta lungo i canali
        concat = torch.cat([x, skip], dim=1)
        return self.conv(concat)

class UNetDenoiseAttack(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetDenoiseAttack, self).__init__()
        
        # Configurazione filtri simmetrica [8, 16, 32, 64, 128]
        self.down1 = DownBlock(in_channels, 8)
        self.down2 = DownBlock(8, 16)
        self.down3 = DownBlock(16, 32)
        self.down4 = DownBlock(32, 64)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling Path
        self.up1 = UpBlock(128, 64)
        self.up2 = UpBlock(64, 32)
        self.up3 = UpBlock(32, 16)
        self.up4 = UpBlock(16, 8)
        
        # Output Reconstruction
        self.out_conv = nn.Sequential(
            nn.Conv2d(8, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        c1, p1 = self.down1(x)
        c2, p2 = self.down2(p1)
        c3, p3 = self.down3(p2)
        c4, p4 = self.down4(p3)
        
        bn = self.bottleneck(p4)
        
        u1 = self.up1(bn, c4)
        u2 = self.up2(u1, c3)
        u3 = self.up3(u2, c2)
        u4 = self.up4(u3, c1)
        
        return self.out_conv(u4)

if __name__ == "__main__":
    dummy_input = torch.randn(1, 3, 256, 256)
    model = UNetDenoiseAttack()
    output = model(dummy_input)
    print(f"Input U-Net Shape: {dummy_input.shape}")
    print(f"Output U-Net Shape: {output.shape}")