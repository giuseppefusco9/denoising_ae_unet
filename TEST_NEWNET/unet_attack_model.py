"""
unet_attack_model.py
====================
UNet per rimozione watermark (PixelSeal) — reimplementazione potenziata.

Miglioramenti rispetto all'originale:
  1. ResidualDoubleConv  : blocchi residuali interni al double-conv per
                          migliore propagazione del gradiente e convergenza.
  2. CBAM               : Convolutional Block Attention Module (channel + spatial)
                          applicato dopo ogni livello encoder per focalizzare le
                          feature sul segnale del watermark.
  3. Attention Gates     : attention gate sulle skip connection, che sopprime le
                          feature irrilevanti prima della concatenazione con il
                          decoder, riducendo artefatti.
  4. Canali aumentati    : 32-64-128-256-512 (vs 16-32-64-128-256) per maggiore
                          capacità di rappresentazione.
  5. Dropout nel bottleneck : riduce l'overfitting sul segnale di watermark.
  6. Output head migliorato : due Conv finali con attivazione intermedia prima
                              del tanh residuale, per output più stabili.
  7. deep_supervision    : flag opzionale per ottenere loss ausiliarie durante il
                          training (utile con encoder molto profondi).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Blocco base: Residual Double Conv
# ---------------------------------------------------------------------------

class ResidualDoubleConv(nn.Module):
    """
    (Conv 3x3 → BN → GELU) × 2  con skip connection 1x1 se i canali cambiano.
    GELU al posto di ReLU: migliore gradiente nelle regioni negative, dimostrato
    efficace nei modelli di denoising (es. DnCNN, NAFNet).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # Proiezione residuale (1×1) solo se i canali non coincidono
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))


# ---------------------------------------------------------------------------
# CBAM – Convolutional Block Attention Module
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation sul canale con riduzione r."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        return x * scale


class SpatialAttention(nn.Module):
    """Attention spaziale via statistiche di canale (avg + max)."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_f = x.mean(dim=1, keepdim=True)
        max_f, _ = x.max(dim=1, keepdim=True)
        scale = torch.sigmoid(self.conv(torch.cat([avg_f, max_f], dim=1)))
        return x * scale


class CBAM(nn.Module):
    """Channel + Spatial Attention in sequenza."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


# ---------------------------------------------------------------------------
# Attention Gate per skip connections
# ---------------------------------------------------------------------------

class AttentionGate(nn.Module):
    """
    Filtra le skip connection enfatizzando le regioni rilevanti per il decoder.
    g  = segnale guida dal decoder (risoluzione ridotta, upsamplato)
    x  = feature map dell'encoder (skip)
    """

    def __init__(self, f_g: int, f_x: int, f_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(f_x, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # g può avere h/w diverso da x → interpolazione
        g_up = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        alpha = self.psi(self.relu(self.W_g(g_up) + self.W_x(x)))
        return x * alpha


# ---------------------------------------------------------------------------
# Modello principale: UNetDenoiseAttack (v2)
# ---------------------------------------------------------------------------

class UNetDenoiseAttack(nn.Module):
    """
    UNet per watermark removal con:
      - ResidualDoubleConv in ogni stadio
      - CBAM dopo ogni encoder stage
      - Attention Gates sulle skip connections
      - Dropout nel bottleneck
      - Output head a due stadi con tanh residuale

    Parametri:
        in_channels  (int): canali input (3 = RGB).
        out_channels (int): canali output (3 = RGB).
        base_ch      (int): canali al primo livello dell'encoder (default 32).
        dropout_p    (float): dropout nel bottleneck (default 0.3).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_ch: int = 32,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        b = base_ch  # 32

        # ── ENCODER ──────────────────────────────────────────────────────────
        self.enc1 = ResidualDoubleConv(in_channels, b)        # 3  → 32
        self.enc2 = ResidualDoubleConv(b,     b * 2)          # 32 → 64
        self.enc3 = ResidualDoubleConv(b * 2, b * 4)          # 64 → 128
        self.enc4 = ResidualDoubleConv(b * 4, b * 8)          # 128→ 256
        self.enc5 = ResidualDoubleConv(b * 8, b * 16)         # 256→ 512

        self.pool = nn.MaxPool2d(2)

        # CBAM dopo ogni stage encoder (analizza le feature prima del pool)
        self.cbam1 = CBAM(b)
        self.cbam2 = CBAM(b * 2)
        self.cbam3 = CBAM(b * 4)
        self.cbam4 = CBAM(b * 8)
        self.cbam5 = CBAM(b * 16)

        # ── BOTTLENECK ───────────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            ResidualDoubleConv(b * 16, b * 32),               # 512 → 1024
            nn.Dropout2d(p=dropout_p),
        )

        # ── DECODER ──────────────────────────────────────────────────────────
        # Ogni stadio: up-conv → attention gate → concat → ResidualDoubleConv

        # Livello 5 (1024 → 512)
        self.up5      = nn.ConvTranspose2d(b * 32, b * 16, kernel_size=2, stride=2)
        self.ag5      = AttentionGate(f_g=b * 16, f_x=b * 16, f_int=b * 8)
        self.dec5     = ResidualDoubleConv(b * 32, b * 16)

        # Livello 4 (512 → 256)
        self.up4      = nn.ConvTranspose2d(b * 16, b * 8, kernel_size=2, stride=2)
        self.ag4      = AttentionGate(f_g=b * 8,  f_x=b * 8,  f_int=b * 4)
        self.dec4     = ResidualDoubleConv(b * 16, b * 8)

        # Livello 3 (256 → 128)
        self.up3      = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.ag3      = AttentionGate(f_g=b * 4,  f_x=b * 4,  f_int=b * 2)
        self.dec3     = ResidualDoubleConv(b * 8,  b * 4)

        # Livello 2 (128 → 64)
        self.up2      = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.ag2      = AttentionGate(f_g=b * 2,  f_x=b * 2,  f_int=b)
        self.dec2     = ResidualDoubleConv(b * 4,  b * 2)

        # Livello 1 (64 → 32)
        self.up1      = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.ag1      = AttentionGate(f_g=b,      f_x=b,      f_int=b // 2)
        self.dec1     = ResidualDoubleConv(b * 2,  b)

        # ── OUTPUT HEAD ──────────────────────────────────────────────────────
        # Due conv finali: 32 → 16 → out_channels, con GELU intermedio
        self.head = nn.Sequential(
            nn.Conv2d(b, b // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(b // 2),
            nn.GELU(),
            nn.Conv2d(b // 2, out_channels, kernel_size=1),
        )

        # Inizializzazione pesi
        self._init_weights()

    # -------------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # -------------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        detector=None,
    ):
        """
        Args:
            x        : tensore input [B, 3, H, W] normalizzato in [0,1].
            detector : modello videoseal opzionale (per estrarre bit logits).

        Returns:
            Se detector is None  → reconstructed_imgs [B, 3, H, W]
            Se detector is not None → (reconstructed_imgs, detected_bit_logits)
        """
        # ── ENCODER ──
        s1 = self.cbam1(self.enc1(x))               # [B,  32, H,    W   ]
        s2 = self.cbam2(self.enc2(self.pool(s1)))    # [B,  64, H/2,  W/2 ]
        s3 = self.cbam3(self.enc3(self.pool(s2)))    # [B, 128, H/4,  W/4 ]
        s4 = self.cbam4(self.enc4(self.pool(s3)))    # [B, 256, H/8,  W/8 ]
        s5 = self.cbam5(self.enc5(self.pool(s4)))    # [B, 512, H/16, W/16]

        # ── BOTTLENECK ──
        b_out = self.bottleneck(self.pool(s5))       # [B,1024, H/32, W/32]

        # ── DECODER ──
        d5 = self.dec5(torch.cat([self.up5(b_out), self.ag5(self.up5(b_out), s5)], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d5),    self.ag4(self.up4(d5),    s4)], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4),    self.ag3(self.up3(d4),    s3)], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3),    self.ag2(self.up2(d3),    s2)], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2),    self.ag1(self.up1(d2),    s1)], dim=1))

        # ── OUTPUT HEAD ──
        # Residual learning: il modello predice il rumore del watermark
        residual = torch.tanh(self.head(d1))         # [-1, +1]
        reconstructed_imgs = torch.clamp(x + residual, 0.0, 1.0)

        if detector is not None:
            with torch.set_grad_enabled(torch.is_grad_enabled()):
                detector_outputs    = detector.detect(reconstructed_imgs)
                detected_bit_logits = detector_outputs["preds"][:, 1:]
            return reconstructed_imgs, detected_bit_logits

        return reconstructed_imgs