"""
train_unet.py
=============
Training loop per UNetDenoiseAttack (v2).

Miglioramenti apportati:
  1. Loss Avversariale Corretta: Sostituita BCEWithLogitsLoss (che con target 0.5 mandava in 
     stallo i gradienti) con L1Loss ancorata a 0.0 per distruggere il segnale.
  2. Bilanciamento Pesi: Priorità all'attacco avversariale (W_ADV=8.0) rispetto alla 
     conservazione visiva (W_IMG=2.0, W_SSIM=1.5) per forzare l'eliminazione del watermark.
  3. Controllo dei Gradienti: Monitoraggio attivo ad ogni epoca per verificare che l'estrattore 
     stia propagando i gradienti alla UNet senza interrompere il grafo computazionale.
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import pandas as pd
import matplotlib.pyplot as plt
import videoseal

from loader_dataset import WatermarkDenoisingDataset
from unet_attack_model import UNetDenoiseAttack


# ============================================================================
# SSIM Loss (differenziabile, implementazione leggera senza dipendenze extra)
# ============================================================================

class SSIMLoss(nn.Module):
    """
    Structural Similarity Index loss (1 − SSIM).
    Finestra gaussiana 11×11, k1=0.01, k2=0.03, val_range=1.0.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.ws = window_size
        kernel_1d = self._gaussian_kernel(window_size, sigma)
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]       # (ws, ws)
        self.register_buffer(
            "kernel", kernel_2d.unsqueeze(0).unsqueeze(0)
        )
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k = self.kernel.expand(C, 1, self.ws, self.ws)  # [C, 1, ws, ws]
        pad = self.ws // 2

        mu_x  = nn.functional.conv2d(x, k, padding=pad, groups=C)
        mu_y  = nn.functional.conv2d(y, k, padding=pad, groups=C)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sig_x2  = nn.functional.conv2d(x * x, k, padding=pad, groups=C) - mu_x2
        sig_y2  = nn.functional.conv2d(y * y, k, padding=pad, groups=C) - mu_y2
        sig_xy  = nn.functional.conv2d(x * y, k, padding=pad, groups=C) - mu_xy

        num   = (2 * mu_xy + self.C1) * (2 * sig_xy + self.C2)
        denom = (mu_x2 + mu_y2 + self.C1) * (sig_x2 + sig_y2 + self.C2)
        return (num / denom).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - self._ssim(pred, target)


# ============================================================================
# Utility: LR con linear warmup + cosine annealing
# ============================================================================

def warmup_cosine_lambda(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ============================================================================
# Utility: peso avversariale con rampa sigmoide
# ============================================================================

def sigmoid_ramp(ep: int, phase1_end: int, phase2_end: int, w_max: float) -> float:
    """Rampa sigmoide da 0 a w_max nell'intervallo [phase1_end, phase2_end]."""
    if ep <= phase1_end:
        return 0.0
    if ep >= phase2_end:
        return w_max
    t = (ep - phase1_end) / (phase2_end - phase1_end)   # 0..1
    s = 1.0 / (1.0 + math.exp(-10.0 * (t - 0.5)))
    return w_max * s


# ============================================================================
# 1. CONFIGURAZIONE
# ============================================================================

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

print("=" * 55)
print(f"  Dispositivo principale : {device}")
print(f"  GPU disponibili       : {num_gpus}")
print("=" * 55)

# ── Iperparametri ──
BATCH_SIZE     = 16          
ACCUM_STEPS    = 2           
EPOCHS         = 120
LEARNING_RATE  = 2e-4
WARMUP_EPOCHS  = 5
CROP_SIZE      = 256
MAX_GRAD_NORM  = 1.0
EARLY_STOP_PAT = 20          

# ── Pesi loss bilanciati per favorire l'attacco avversariale ──
W_IMG   = 2.0    # Ridotto (era 8.0) per permettere modifiche ai pixel del WM
W_SSIM  = 1.5    # Ridotto (era 4.0)
W_ADV   = 8.0    # Incrementato drasticamente (era 1.5) per distruggere i bit

# ── Fasi curriculum accelerate ──
PHASE1_END = 4    # L'attacco avversariale entra prima in gioco (era 8)
PHASE2_END = 30   # Raggiunge la massima potenza a epoca 30 (era 50)

# ============================================================================
# 2. DATASET
# ============================================================================

print("\nCaricamento dataset...")
train_ds = WatermarkDenoisingDataset(root_dir="dataset_minSize/train", crop_size=CROP_SIZE)
val_ds   = WatermarkDenoisingDataset(root_dir="dataset_minSize/val",   crop_size=CROP_SIZE)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, persistent_workers=True,
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True,
)
print(f"  Train: {len(train_ds):,} immagini | Val: {len(val_ds):,} immagini")

# ============================================================================
# 3. MODELLI, LOSS, OTTIMIZZATORE
# ============================================================================

model = UNetDenoiseAttack(in_channels=3, out_channels=3, base_ch=32, dropout_p=0.3).to(device)

print("\nCaricamento detector PixelSeal...")
detector = videoseal.load("pixelseal").to(device)
detector.eval()
for p in detector.parameters():
    p.requires_grad = False

criterion_l1   = nn.L1Loss().to(device)
criterion_ssim = SSIMLoss(window_size=11).to(device)

if num_gpus > 1:
    model = nn.DataParallel(model)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

scheduler = optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda ep: warmup_cosine_lambda(ep, WARMUP_EPOCHS, EPOCHS),
)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("checkpoints/progress_images", exist_ok=True)

# ============================================================================
# 4. FUNZIONE DI LOSS
# ============================================================================

def compute_loss(
    reconstructed: torch.Tensor,
    logits_recon: torch.Tensor,
    clean: torch.Tensor,
    w_img: float,
    w_ssim: float,
    w_adv: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tre componenti:
      loss_l1   : fedeltà pixel (L1)
      loss_ssim : fedeltà strutturale (1 - SSIM)
      loss_adv  : porta i bit logit a 0.0 via L1 per massimizzare l'incertezza
                  dell'estrattore (bit accuracy -> 0.5) senza saturazione.
    """
    loss_l1   = criterion_l1(reconstructed, clean)
    loss_ssim = criterion_ssim(reconstructed, clean)

    # Ancoraggio dei logit a 0.0 usando la L1 per un gradiente stabile e costante
    adv_target = torch.zeros_like(logits_recon)
    loss_adv   = criterion_l1(logits_recon, adv_target)

    total = w_img * loss_l1 + w_ssim * loss_ssim + w_adv * loss_adv
    return total, loss_l1, loss_ssim, loss_adv


# ============================================================================
# 5. TRAINING LOOP
# ============================================================================

history = {k: [] for k in
    ["train_total", "train_l1", "train_ssim", "train_adv",
     "val_total",   "val_l1",   "val_ssim",   "val_adv", "lr"]}

best_val   = float("inf")
no_improve = 0                        

print(f"\nInizio addestramento — {EPOCHS} epoche, batch effettivo={BATCH_SIZE*ACCUM_STEPS}\n")

for epoch in range(EPOCHS):
    ep = epoch + 1
    w_adv_cur = sigmoid_ramp(ep, PHASE1_END, PHASE2_END, W_ADV)

    # ── TRAIN ──────────────────────────────────────────────────────────────
    model.train()
    running = {k: 0.0 for k in ["total", "l1", "ssim", "adv"]}
    optimizer.zero_grad()

    for step, (wm_imgs, clean_imgs) in enumerate(train_loader):
        wm_imgs, clean_imgs = wm_imgs.to(device), clean_imgs.to(device)

        recon, logits = model(wm_imgs, detector=detector)

        total, l_l1, l_ssim, l_adv = compute_loss(
            recon, logits, clean_imgs, W_IMG, W_SSIM, w_adv_cur
        )

        if num_gpus > 1:
            total = total.mean(); l_l1 = l_l1.mean()
            l_ssim = l_ssim.mean(); l_adv = l_adv.mean()

        # Gradient accumulation
        (total / ACCUM_STEPS).backward()

        # --- MONITORAGGIO GRADIENTI (DEBUG TRACCIAMENTO ADVERSARIAL) ---
        if step % 50 == 0:
            first_layer_grad = model.module.enc1.conv[0].weight.grad if num_gpus > 1 else model.enc1.conv[0].weight.grad
            if first_layer_grad is not None:
                grad_mean = first_layer_grad.abs().mean().item()
                if grad_mean == 0.0 and w_adv_cur > 0:
                    print(f"⚠️ [Step {step}]: Gradiente nullo rilevato! Il detector potrebbe bloccare il grafo.")
            else:
                print(f"⚠️ [Step {step}]: Gradiente non inizializzato (None).")

        running["total"] += total.item()
        running["l1"]    += l_l1.item()
        running["ssim"]  += l_ssim.item()
        running["adv"]   += l_adv.item()

        if (step + 1) % ACCUM_STEPS == 0:
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()

    # Flush residuo
    if len(train_loader) % ACCUM_STEPS != 0:
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()

    n = len(train_loader)
    for k in running:
        history[f"train_{k}"].append(running[k] / n)

    # ── VALIDATION ─────────────────────────────────────────────────────────
    model.eval()
    val_run = {k: 0.0 for k in ["total", "l1", "ssim", "adv"]}
    preview_saved = False

    with torch.no_grad():
        for wm_imgs, clean_imgs in val_loader:
            wm_imgs, clean_imgs = wm_imgs.to(device), clean_imgs.to(device)
            recon, logits = model(wm_imgs, detector=detector)

            total, l_l1, l_ssim, l_adv = compute_loss(
                recon, logits, clean_imgs, W_IMG, W_SSIM, w_adv_cur
            )
            if num_gpus > 1:
                total = total.mean(); l_l1 = l_l1.mean()
                l_ssim = l_ssim.mean(); l_adv = l_adv.mean()

            val_run["total"] += total.item()
            val_run["l1"]    += l_l1.item()
            val_run["ssim"]  += l_ssim.item()
            val_run["adv"]   += l_adv.item()

            if not preview_saved:
                grid = torch.cat([wm_imgs[:1], clean_imgs[:1], recon[:1]], dim=0)
                vutils.save_image(
                    grid,
                    f"checkpoints/progress_images/epoch_{ep:03d}.png",
                    normalize=True,
                )
                preview_saved = True

    nv = len(val_loader)
    for k in val_run:
        history[f"val_{k}"].append(val_run[k] / nv)

    cur_lr = optimizer.param_groups[0]["lr"]
    history["lr"].append(cur_lr)
    scheduler.step()

    cur_val = history["val_total"][-1]
    flag    = ""
    if cur_val < best_val:
        best_val   = cur_val
        no_improve = 0
        sd = model.module.state_dict() if num_gpus > 1 else model.state_dict()
        torch.save(sd, "checkpoints/unet_best.pth")
        flag = "  ✔ best"
    else:
        no_improve += 1

    # Checkpoint di salvataggio dello stato corrente
    sd_last = model.module.state_dict() if num_gpus > 1 else model.state_dict()
    torch.save(
        {"epoch": ep, "state_dict": sd_last, "optimizer": optimizer.state_dict()},
        "checkpoints/unet_last.pth",
    )

    print(
        f"Ep [{ep:3d}/{EPOCHS}] "
        f"w_adv={w_adv_cur:.3f} lr={cur_lr:.2e} | "
        f"Train → tot={history['train_total'][-1]:.4f}  "
        f"l1={history['train_l1'][-1]:.5f}  "
        f"ssim={history['train_ssim'][-1]:.5f}  "
        f"adv={history['train_adv'][-1]:.4f} | "
        f"Val → tot={cur_val:.4f}  "
        f"l1={history['val_l1'][-1]:.5f}  "
        f"ssim={history['val_ssim'][-1]:.5f}  "
        f"adv={history['val_adv'][-1]:.4f}"
        + flag
    )

    if no_improve >= EARLY_STOP_PAT:
        print(f"\nEarly stopping: nessun miglioramento per {EARLY_STOP_PAT} epoche.")
        break

print("\nAddestramento completato.")
EPOCHS_RAN = len(history["train_total"])

# ============================================================================
# 6. ESPORTAZIONE DATI STORICI (CSV)
# ============================================================================

pd.DataFrame({
    "Epoca":        range(1, EPOCHS_RAN + 1),
    "Train_Total":  history["train_total"],
    "Train_L1":     history["train_l1"],
    "Train_SSIM":   history["train_ssim"],
    "Train_Adv":    history["train_adv"],
    "Val_Total":    history["val_total"],
    "Val_L1":       history["val_l1"],
    "Val_SSIM":     history["val_ssim"],
    "Val_Adv":      history["val_adv"],
    "LR":           history["lr"],
}).to_csv("unet_training_history.csv", index=False, sep=";")

# ============================================================================
# 7. GRAFICO DELLE METRICHE
# ============================================================================

epochs_ax = range(1, EPOCHS_RAN + 1)
fig, axes = plt.subplots(3, 1, figsize=(13, 12))
colors = {"train": "#2563eb", "val": "#dc2626"}
ls_val  = "--"

ax = axes[0]
ax.plot(epochs_ax, history["train_total"], label="Train total", color=colors["train"], lw=2)
ax.plot(epochs_ax, history["val_total"],   label="Val total",   color=colors["val"],   lw=2, ls=ls_val)
ax.set_ylabel("Loss totale"); ax.set_title("Loss Totale Combinata"); ax.legend(); ax.grid(True, ls=":")
ax2 = ax.twinx()
ax2.plot(epochs_ax, history["lr"], color="gray", lw=1, alpha=0.6, label="LR")
ax2.set_ylabel("Learning Rate", color="gray"); ax2.tick_params(axis="y", labelcolor="gray")

ax = axes[1]
ax.plot(epochs_ax, history["train_l1"],   label="Train L1",   color="#1d4ed8", lw=2)
ax.plot(epochs_ax, history["val_l1"],     label="Val L1",     color="#1d4ed8", lw=2, ls=ls_val)
ax.plot(epochs_ax, history["train_ssim"], label="Train SSIM", color="#7c3aed", lw=2)
ax.plot(epochs_ax, history["val_ssim"],   label="Val SSIM",   color="#7c3aed", lw=2, ls=ls_val)
ax.set_ylabel("Loss immagine"); ax.set_title("Componenti: L1 e SSIM"); ax.legend(); ax.grid(True, ls=":")

ax = axes[2]
ax.plot(epochs_ax, history["train_adv"], label="Train Adv (L1 anchored)",  color="#ea580c", lw=2)
ax.plot(epochs_ax, history["val_adv"],   label="Val Adv (L1 anchored)",    color="#ea580c", lw=2, ls=ls_val)
ax.set_xlabel("Epoche"); ax.set_ylabel("Loss avversariale"); ax.set_title("Loss Avversariale (bit-logit)"); ax.legend(); ax.grid(True, ls=":")

plt.tight_layout()
plt.savefig("unet_training_curves.png", dpi=300)
plt.close()
print("✅ CSV e grafici aggiornati salvati con successo.")