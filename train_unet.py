import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import videoseal

from loader_dataset import WatermarkDenoisingDataset
from unet_attack_model import UNetDenoiseAttack

# ==========================================
# 1. CONFIGURAZIONE HARDWARE E PARAMETRI
# ==========================================
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus  = torch.cuda.device_count()

print("=" * 50)
print(f"Dispositivo Principale: {device}")
print("=" * 50)

BATCH_SIZE    = 12
EPOCHS        = 100
LEARNING_RATE = 2e-5
CROP_SIZE     = 512
MAX_GRAD_NORM = 1.0

# ==========================================
# PESI DELLA LOSS
# ==========================================
W_IMG = 1.0
W_ADV = 0.0
PHASE1_END = -1
PHASE2_END = -1

# ==========================================
# 2. PREPARAZIONE DATI
# ==========================================
print("Caricamento dataset...")
train_dataset = WatermarkDenoisingDataset(root_dir="dataset_minSize/train", crop_size=CROP_SIZE)
val_dataset   = WatermarkDenoisingDataset(root_dir="dataset_minSize/val",   crop_size=CROP_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

# ==========================================
# 3. INIZIALIZZAZIONE MODELLI
# ==========================================
print("Caricamento detector PixelSeal...")
detector = videoseal.load("pixelseal").to(device)
detector.eval()

model = UNetDenoiseAttack(in_channels=3, out_channels=3, detector=detector).to(device)


for param in detector.parameters():
    param.requires_grad = False

criterion_img  = nn.L1Loss().to(device)
criterion_bits = nn.L1Loss().to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("checkpoints/progress_images", exist_ok=True)

history = {
    "train_total": [], "train_img": [], "train_adv": [],
    "val_total":   [], "val_img":   [], "val_adv":   []
}

# ==========================================
# 4. TRAINING LOOP
# ==========================================
print(f"\nInizio Addestramento su {EPOCHS} epoche...\n")
best_val_loss = float('inf')


def compute_loss(reconstructed_imgs, logits_reconstructed,
                 clean_imgs, w_img, w_adv):
    """
    Due componenti:
    - loss_img : L1 pixel tra immagine ricostruita e clean (fidelity).
    - loss_adv : L1 tra bit-logit attaccati e quelli di un'immagine pulita
                 (il messaggio estratto deve sembrare quello di un'immagine senza WM).
    """
    loss_img = criterion_img(reconstructed_imgs, clean_imgs)

    with torch.no_grad():
        out_clean        = detector.detect(clean_imgs)
        logits_clean_ref = out_clean["preds"][:, 1:].detach()

    loss_adv = criterion_bits(logits_reconstructed, logits_clean_ref)

    total = w_img * loss_img + w_adv * loss_adv
    return total, loss_img, loss_adv


for epoch in range(EPOCHS):
    model.train()
    ep = epoch + 1

    # --- Scheduling del peso avversariale ---
    if ep <= PHASE1_END:
        w_adv_cur = 0.0
    elif ep <= PHASE2_END:
        t         = (ep - PHASE1_END) / (PHASE2_END - PHASE1_END)
        w_adv_cur = W_ADV * t
    else:
        w_adv_cur = W_ADV

    epoch_losses = {"total": 0.0, "img": 0.0, "adv": 0.0}

    for wm_imgs, clean_imgs in train_loader:
        wm_imgs    = wm_imgs.to(device)
        clean_imgs = clean_imgs.to(device)

        optimizer.zero_grad()

        reconstructed_imgs, logits_reconstructed = model(wm_imgs)

        total_loss, loss_img, loss_adv = compute_loss(
            reconstructed_imgs, logits_reconstructed,
            clean_imgs, W_IMG, w_adv_cur
        )

        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()

        epoch_losses["total"] += total_loss.item()
        epoch_losses["img"]   += loss_img.item()
        epoch_losses["adv"]   += loss_adv.item()

    n_train = len(train_loader)
    for k in epoch_losses:
        history[f"train_{k}"].append(epoch_losses[k] / n_train)

    # --- Validazione ---
    model.eval()
    val_losses  = {"total": 0.0, "img": 0.0, "adv": 0.0}
    saved_preview = False

    with torch.no_grad():
        for wm_imgs, clean_imgs in val_loader:
            wm_imgs    = wm_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            reconstructed_imgs, logits_reconstructed = model(wm_imgs)

            total_val, loss_img, loss_adv = compute_loss(
                reconstructed_imgs, logits_reconstructed,
                clean_imgs, W_IMG, w_adv_cur
            )

            val_losses["total"] += total_val.item()
            val_losses["img"]   += loss_img.item()
            val_losses["adv"]   += loss_adv.item()

            if not saved_preview:
                grid = torch.cat([wm_imgs[0:1], clean_imgs[0:1], reconstructed_imgs[0:1]], dim=0)
                vutils.save_image(
                    grid,
                    f"checkpoints/progress_images/epoch_{ep:03d}.png",
                    normalize=True
                )
                saved_preview = True

    n_val = len(val_loader)
    for k in val_losses:
        history[f"val_{k}"].append(val_losses[k] / n_val)

    cur_val_total = history["val_total"][-1]
    lr_scheduler.step(cur_val_total)

    print(
        f"Ep [{ep:3d}/{EPOCHS}] | w_adv={w_adv_cur:.2f} | "
        f"Train → total={history['train_total'][-1]:.4f}  img={history['train_img'][-1]:.5f}  "
        f"adv={history['train_adv'][-1]:.4f} | "
        f"Val → total={cur_val_total:.4f}  img={history['val_img'][-1]:.5f}  "
        f"adv={history['val_adv'][-1]:.4f}"
    )

    if cur_val_total < best_val_loss:
        best_val_loss     = cur_val_total
        state_to_save     = model.state_dict()
        torch.save(state_to_save, "checkpoints/unet_best.pth")
        print("  -> Nuovo record. Pesi salvati.")

print("\nAddestramento Completato.")

# ==========================================
# 5. CSV
# ==========================================
pd.DataFrame({
    "Epoca":         range(1, EPOCHS + 1),
    "Train_Total":   history["train_total"],
    "Train_Img_L1":  history["train_img"],
    "Train_Adv_L1":  history["train_adv"],
    "Val_Total":     history["val_total"],
    "Val_Img_L1":    history["val_img"],
    "Val_Adv_L1":    history["val_adv"],
}).to_csv("unet_separated_losses.csv", index=False, sep=";")

# ==========================================
# 6. GRAFICO A DUE PANNELLI
# ==========================================
epochs_axis = range(1, EPOCHS + 1)
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Pannello 1: Loss totale
ax = axes[0]
ax.plot(epochs_axis, history["train_total"], label="Train Total", color="purple", lw=2)
ax.plot(epochs_axis, history["val_total"],   label="Val Total",   color="purple", lw=2, ls="--")
ax.set_ylabel("Loss Totale")
ax.set_title("Loss Totale Combinata")
ax.legend(); ax.grid(True, ls=":")

# Pannello 2: Componenti separate (stesso ordine di grandezza grazie al bilanciamento)
ax = axes[1]
ax.plot(epochs_axis, history["train_img"], label="Train Img (L1)",      color="blue",   lw=2)
ax.plot(epochs_axis, history["val_img"],   label="Val Img (L1)",        color="blue",   lw=2, ls="--")
ax.plot(epochs_axis, history["train_adv"], label="Train Bit-Logit (L1)", color="orange", lw=2)
ax.plot(epochs_axis, history["val_adv"],   label="Val Bit-Logit (L1)",  color="orange", lw=2, ls="--")
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss Componente (Valore Puro)")
ax.set_title("Analisi Separata delle Componenti")
ax.legend(); ax.grid(True, ls=":")

plt.tight_layout()
plt.savefig("unet_separated_loss_plot.png", dpi=300)
plt.close()

print("✅ CSV e grafico a due pannelli salvati con successo.")