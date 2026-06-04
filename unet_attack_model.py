import torch
import torch.nn as nn
import videoseal

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
        
        # Convoluzione finale d'uscita (RGB a 3 canali)
        self.outc = nn.Conv2d(16, out_channels, kernel_size=1)
        
        # --- INCAPSULAMENTO DETECTOR PIXELSEAL (FROZEN) ---
        self.detector = videoseal.load("pixelseal")
        self.detector.eval() # Imposta permanentemente in modalità valutazione
        
        # Congelamento immediato di tutti i parametri del modello di Meta
        for param in self.detector.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 1. Fase di attacco ed elaborazione neurale (U-Net)
        x1 = self.inc(x)        
        x2 = self.down1(x1)     
        x3 = self.down2(x2)     
        x4 = self.down3(x3)     
        x5 = self.down4(x4)     
        
        b = self.bottleneck(x5) 
        
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
        
        reconstructed_imgs = self.outc(t5) 
        
        # 2. Estrazione automatica dello spazio latente tramite il detector interno
        detector_outputs = self.detector.detect(reconstructed_imgs)
        # Isoliamo i logiti dei bit dal secondo indice in poi, coerentemente con il tuo script
        detected_bit_logits = detector_outputs["preds"][:, 1:]
        
        # Restituiamo sia l'immagine modificata sia i logiti differenziabili
        return reconstructed_imgs, detected_bit_logits