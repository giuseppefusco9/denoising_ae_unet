"""
genSummaryTab.py
================
Genera un riepilogo strutturale dettagliato di UNetDenoiseAttack (v2).

Miglioramenti:
  - Analisi parametri per componente (encoder / CBAM / attention gates /
    bottleneck / decoder / head)
  - Stima FLOPs tramite thop (opzionale, graceful fallback)
  - Export in summary.txt + summary_by_component.csv
"""

import torch
import torch.nn as nn
import pandas as pd
import videoseal

try:
    from torchinfo import summary as torchinfo_summary
    HAS_TORCHINFO = True
except ImportError:
    HAS_TORCHINFO = False
    print("⚠  torchinfo non trovato — il summary dettagliato non sarà generato.")

try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("ℹ  thop non trovato — la stima FLOPs non sarà disponibile.")

from unet_attack_model import UNetDenoiseAttack


# ---------------------------------------------------------------------------
# Wrapper (necessario affinché torchinfo tratti il detector come modulo frozen)
# ---------------------------------------------------------------------------

class AttackAndEvaluationWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet     = UNetDenoiseAttack(in_channels=3, out_channels=3, base_ch=32)
        self.detector = videoseal.load("pixelseal")
        self.detector.eval()
        for p in self.detector.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor):
        return self.unet(x, detector=self.detector)


# ---------------------------------------------------------------------------
# Inizializzazione
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

model_full = AttackAndEvaluationWrapper().to(device)
unet       = model_full.unet

# ---------------------------------------------------------------------------
# 1. Conteggio parametri per componente
# ---------------------------------------------------------------------------

def count_params(module: nn.Module) -> dict[str, int]:
    total     = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


component_map = {
    "Encoder Stage 1 (enc1)":    unet.enc1,
    "Encoder Stage 2 (enc2)":    unet.enc2,
    "Encoder Stage 3 (enc3)":    unet.enc3,
    "Encoder Stage 4 (enc4)":    unet.enc4,
    "Encoder Stage 5 (enc5)":    unet.enc5,
    "CBAM 1":                    unet.cbam1,
    "CBAM 2":                    unet.cbam2,
    "CBAM 3":                    unet.cbam3,
    "CBAM 4":                    unet.cbam4,
    "CBAM 5":                    unet.cbam5,
    "Bottleneck":                unet.bottleneck,
    "Decoder Stage 5 (dec5+ag5)": nn.ModuleList([unet.up5, unet.ag5, unet.dec5]),
    "Decoder Stage 4 (dec4+ag4)": nn.ModuleList([unet.up4, unet.ag4, unet.dec4]),
    "Decoder Stage 3 (dec3+ag3)": nn.ModuleList([unet.up3, unet.ag3, unet.dec3]),
    "Decoder Stage 2 (dec2+ag2)": nn.ModuleList([unet.up2, unet.ag2, unet.dec2]),
    "Decoder Stage 1 (dec1+ag1)": nn.ModuleList([unet.up1, unet.ag1, unet.dec1]),
    "Output Head":                unet.head,
}

rows = []
for name, mod in component_map.items():
    cp = count_params(mod)
    rows.append({
        "Componente":       name,
        "Params Totali":    cp["total"],
        "Params Trainable": cp["trainable"],
    })

total_unet = count_params(unet)
rows.append({
    "Componente":       "─── UNET TOTALE ───",
    "Params Totali":    total_unet["total"],
    "Params Trainable": total_unet["trainable"],
})

df_comp = pd.DataFrame(rows)
df_comp.to_csv("summary_by_component.csv", index=False, sep=";")

print("Parametri per componente:")
print(df_comp.to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 2. FLOPs (opzionale)
# ---------------------------------------------------------------------------

if HAS_THOP:
    dummy = torch.randn(1, 3, 256, 256).to(device)
    macs, params = thop_profile(unet, inputs=(dummy,), verbose=False)
    flops = 2 * macs   # FLOPs ≈ 2 × MACs
    print(f"FLOPs (256×256, batch=1): {flops / 1e9:.2f} GFLOPs")
    print(f"MACs  (256×256, batch=1): {macs  / 1e9:.2f} GMACs\n")

# ---------------------------------------------------------------------------
# 3. Summary torchinfo completo
# ---------------------------------------------------------------------------

if HAS_TORCHINFO:
    model_summary = torchinfo_summary(
        model_full,
        input_size=(4, 3, 256, 256),          # batch=4 per stime più realistiche
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        row_settings=["depth", "var_names"],
        depth=5,
        verbose=0,
        device=device,
    )
    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(str(model_summary))
    print("✅ summary.txt generato.")
else:
    # Fallback minimale
    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write("torchinfo non disponibile.\n\n")
        f.write(df_comp.to_string(index=False))
    print("✅ summary.txt generato (senza torchinfo).")

print("✅ summary_by_component.csv generato.")
print("\nSommario UNet v2:")
print(f"  Parametri totali    : {total_unet['total']:>12,}")
print(f"  Parametri trainable : {total_unet['trainable']:>12,}")