import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        if diffY > 0 or diffX > 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            
        concat = torch.cat([x, skip], dim=1)
        return self.conv(concat)

class UNetDenoiseAttack(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        """
        U-Net Estesa a 5 Livelli con Gestione Interna della Luminanza (YCbCr)
        Accetta RGB in ingresso ed emette RGB in uscita.
        """
        super(UNetDenoiseAttack, self).__init__()
        
        # Encoder Path (Lavora su 1 canale: Luminanza Y)
        self.down1 = DownBlock(1, 16)
        self.down2 = DownBlock(16, 32)
        self.down3 = DownBlock(32, 64)
        self.down4 = DownBlock(64, 128)
        self.down5 = DownBlock(128, 256) # Layer aggiuntivo di profondità
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder Path
        self.up1 = UpBlock(512, 256) # Layer aggiuntivo di risalita
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up4 = UpBlock(64, 32)
        self.up5 = UpBlock(32, 16)
        
        # Output Reconstruction per il singolo canale Y
        self.out_conv = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, rgb_tensor):
        # STEP 1: Conversione interna da RGB a YCbCr (Formule BT.601)
        r = rgb_tensor[:, 0:1, :, :]
        g = rgb_tensor[:, 1:2, :, :]
        b = rgb_tensor[:, 2:3, :, :]
        
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

        # STEP 2: Processamento dell'architettura U-Net sul solo canale Y
        c1, p1 = self.down1(y)
        c2, p2 = self.down2(p1)
        c3, p3 = self.down3(p2)
        c4, p4 = self.down4(p3)
        c5, p5 = self.down5(p4)
        
        bn = self.bottleneck(p5)
        
        u1 = self.up1(bn, c5)
        u2 = self.up2(u1, c4)
        u3 = self.up3(u2, c3)
        u4 = self.up4(u3, c2)
        u5 = self.up5(u4, c1)
        
        y_attacked = self.out_conv(u5)
        
        # STEP 3: Riconversione interna da YCbCr a RGB usando Cb e Cr originali
        r_new = y_attacked + 1.402 * (cr - 128.0)
        g_new = y_attacked - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
        b_new = y_attacked + 1.772 * (cb - 128.0)
        
        rgb_attacked = torch.cat([r_new, g_new, b_new], dim=1)
        return torch.clamp(rgb_attacked, 0.0, 1.0)

if __name__ == "__main__":
    dummy_input = torch.randn(1, 3, 256, 256)
    model = UNetDenoiseAttack()
    output = model(dummy_input)
    print(f"Input RGB Shape: {dummy_input.shape}")
    print(f"Output RGB Shape: {output.shape}")