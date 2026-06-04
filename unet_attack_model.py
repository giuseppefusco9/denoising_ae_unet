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
        
        # --- ENCODER (Downsampling RGB) ---
        self.inc = DoubleConv(in_channels, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # --- BOTTLENECK ---
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # --- DECODER (Upsampling con Skip Connections) ---
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(64, 32)
        
        self.up5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_up5 = DoubleConv(32, 16)
        
        self.outc = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x, detector=None):
        # 1. Pipeline di compressione ed estrazione delle feature (Encoder)
        x1 = self.inc(x)        
        x2 = self.down1(x1)     
        x3 = self.down2(x2)     
        x4 = self.down3(x3)     
        x5 = self.down4(x4)     
        
        b = self.bottleneck(x5) 
        
        # 2. Pipeline di ricostruzione con concatenazione speculare (Decoder)
        t1 = self.up1(b)
        t1 = torch.cat([t1, x5], dim=1) 
        t1 = self.conv_up1(t1)
        
        t2 = self.up2(t1)
        t2 = torch.cat([t2, x4], dim=1) 
        t2 = self.conv_up2(t2)
        
        t3 = self.up3(t2)
        t3 = torch.cat([t3, x3], dim=1) 
        t3 = self.conv_up3(t3)
        
        t4 = self.up4(t3)
        t4 = torch.cat([t4, x2], dim=1) 
        t4 = self.conv_up4(t4)
        
        t5 = self.up5(t4)
        t5 = torch.cat([t5, x1], dim=1) 
        t5 = self.conv_up5(t5)
        
        # Convoluzione lineare grezza
        raw_output = self.outc(t5) 
        
        # --- INNOVAZIONE: SCHEDULING DI NORMALIZZAZIONE CORRETTO VIA SIGMOIDE ---
        # Forza ogni canale del volume d'uscita a rientrare rigidamente nel range [0.0, 1.0]
        reconstructed_imgs = torch.sigmoid(raw_output)
        
        # 3. Passaggio differenziabile protetto all'interno del detector
        if detector is not None:
            detector_outputs = detector.detect(reconstructed_imgs)
            detected_bit_logits = detector_outputs["preds"][:, 1:]
            return reconstructed_imgs, detected_bit_logits
            
        return reconstructed_imgs