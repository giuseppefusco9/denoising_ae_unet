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
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        # CORREZIONE: kernel_size=1 e padding=0 mantiene inalterate H e W (es. 32x32 rimane 32x32)
        self.conv_trans = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) 
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        # 1. Espansione spaziale (es. da 16x16 a 32x32)
        x = self.up(x)
        
        # 2. Riduzione dei canali (mantiene 32x32)
        x = self.conv_trans(x)
        
        # 3. Controllo di sicurezza per il padding (in caso di input dispari nel dataset)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        if diffY > 0 or diffX > 0:
            import torch.nn.functional as F
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            
        # 4. Concatenazione perfetta
        concat = torch.cat([x, skip], dim=1)
        return self.conv(concat)
    
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_trans = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=0) # Dimezza i canali prima del concat
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        # 1. Applica prima l'Upsampling spaziale
        x = self.up(x)
        
        # 2. Calcola e applica il padding spaziale SE ALTEZZA O LARGHEZZA NON COINCIDONO
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        if diffY > 0 or diffX > 0:
            import torch.nn.functional as F
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            
        # 3. Riduci i canali dopo aver allineato le dimensioni spaziali
        x = self.conv_trans(x)
        
        # 4. Esegui la concatenazione (Skip Connection)
        concat = torch.cat([x, skip], dim=1)
        return self.conv(concat)

class UNetDenoiseAttack(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetDenoiseAttack, self).__init__()
        
        # Ridotto la complessità di base [8, 16, 32, 64, 128] come nel tuo notebook per addestrare velocemente su 1080Ti
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
        
        # Output Layer
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
    print("Modello U-Net istanziato con successo per lo scenario avversariale.")