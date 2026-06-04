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
print(f"Numero di GPU rilevate: {num_gpus}")
print("=" * 50)

BATCH_SIZE    = 32
EPOCHS        = 100
LEARNING_RATE = 2e-5
CROP_SIZE     = 256
MAX_GRAD_NORM = 1.0

# ==========================================
# FIX 2 — RIBILANCIAMENTO DEI PESI DELLA LOSS
# ==========================================
# Il problema originale: alpha_eff = 0.8 * (1/0.05) = 16.0
#                        lambda_eff = 0.2 * (1/2.2)  = 0.09
# Il segnale avversariale era ~180x più debole di quello di immagine.
# Qui usiamo pesi semplici e diretti senza normalizzazione moltiplicativa.
#
# W_IMG: peso della L1 (Image Fidelity). ~0.05 di partenza, vogliamo tenerla bassa.
# W_ADV: peso della loss avversariale. Calibrato per avere lo stesso ordine di grandezza.
# W_DET: peso per spingere il detection score verso zero (loss aggiuntiva).
W_IMG = 10.0   # scala la L1 (~0.05 raw) → contributo ~0.5
W_ADV = 1.0    # scala la MSE bit-logit (~1.0-2.0 raw) → contributo ~1-2
W_DET = 5.0    # peso dedicato al detection score (canale 0)

# Target del detection score: vogliamo che il detector "non rilevi" il watermark.
# Il canale 0 di preds è il logit di presenza; spingiamo verso valori negativi/zero.
DETECTION_TARGET = -3.0   # logit target per "nessun watermark"

# Scheduler: introduciamo l'adversarial loss molto prima (epoch 5, non 25)
# per evitare che il modello consolidi un minimo di sola ricostruzione.
PHASE1_END = 5     # solo image loss (warm-up brevissimo)
PHASE2_END = 40    # transizione lineare verso il target finale

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
model = UNetDenoiseAttack(in_channels=3, out_channels=3).to(device)

print("Caricamento detector PixelSeal...")
detector = videoseal.load("pixelseal").to(device)
detector.eval()
for param in detector.parameters():
    param.requires_grad = False

# FIX 3 — LOSS CORRETTE
# - criterion_img   : L1 per fidelity (robusto agli outlier, preferibile a MSE)
# - criterion_bits  : MSE tra i bit-logit attaccati e quelli dell'immagine pulita
#   (vogliamo imitare la distribuzione "no watermark")
# - criterion_detect: BCEWithLogitsLoss spinge il detection score (c0) verso 0/negativo.
#   Usiamo un target fisso negativo per massimizzare la confusione del detector.
criterion_img    = nn.L1Loss().to(device)
criterion_bits   = nn.MSELoss().to(device)
criterion_detect = nn.MSELoss().to(device)   # regressione verso DETECTION_TARGET

if num_gpus > 1:
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# LR scheduler: riduci di 0.5 se la val loss non migliora per 10 epoche consecutive
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("checkpoints/progress_images", exist_ok=True)

history = {
    "train_total": [], "train_img": [], "train_adv": [], "train_det": [],
    "val_total":   [], "val_img":   [], "val_adv":   [], "val_det":   []
}

# ==========================================
# 4. TRAINING LOOP
# ==========================================
print(f"\nInizio Addestramento su {EPOCHS} epoche...\n")
best_val_loss = float('inf')


def compute_loss(reconstructed_imgs, detection_score, logits_reconstructed,
                 clean_imgs, wm_imgs, w_img, w_adv, w_det):
    """
    Calcola le tre componenti della loss in modo chiaro e separato.

    - loss_img    : L1 tra immagine ricostruita e clean (fidelity).
    - loss_adv    : MSE tra i bit-logit attaccati e quelli di un'immagine pulita
                    (fa sì che il messaggio estratto sembri rumore, non watermark).
    - loss_detect : Regressione del detection score verso DETECTION_TARGET
                    (il detector non deve rilevare presenza watermark).
    """
    # Fidelity
    loss_img = criterion_img(reconstructed_imgs, clean_imgs)

    # Bit-logit adversarial: target = logit di immagine pulita (no gradiente)
    with torch.no_grad():
        out_clean        = detector.detect(clean_imgs)
        logits_clean_ref = out_clean["preds"][:, 1:].detach()

    loss_adv = criterion_bits(logits_reconstructed, logits_clean_ref)

    # Detection score adversarial: spingi c0 verso DETECTION_TARGET
    det_target = torch.full_like(detection_score, DETECTION_TARGET)
    loss_det   = criterion_detect(detection_score, det_target)

    total = w_img * loss_img + w_adv * loss_adv + w_det * loss_det
    return total, loss_img, loss_adv, loss_det


for epoch in range(EPOCHS):
    model.train()
    ep = epoch + 1

    # --- Scheduling dei pesi avversariali ---
    if ep <= PHASE1_END:
        # Warm-up: solo image fidelity, nessun segnale avversariale
        w_adv_cur = 0.0
        w_det_cur = 0.0
    elif ep <= PHASE2_END:
        # Transizione lineare
        t          = (ep - PHASE1_END) / (PHASE2_END - PHASE1_END)
        w_adv_cur  = W_ADV * t
        w_det_cur  = W_DET * t
    else:
        # Regime stazionario
        w_adv_cur = W_ADV
        w_det_cur = W_DET

    epoch_losses = {"total": 0.0, "img": 0.0, "adv": 0.0, "det": 0.0}

    for wm_imgs, clean_imgs in train_loader:
        wm_imgs    = wm_imgs.to(device)
        clean_imgs = clean_imgs.to(device)

        optimizer.zero_grad()

        # FIX: il forward ora restituisce anche detection_score (c0)
        reconstructed_imgs, logits_reconstructed, detection_score = model(
            wm_imgs, detector=detector
        )

        total_loss, loss_img, loss_adv, loss_det = compute_loss(
            reconstructed_imgs, detection_score, logits_reconstructed,
            clean_imgs, wm_imgs, W_IMG, w_adv_cur, w_det_cur
        )

        if num_gpus > 1:
            total_loss = total_loss.mean()
            loss_img   = loss_img.mean()
            loss_adv   = loss_adv.mean()
            loss_det   = loss_det.mean()

        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()

        epoch_losses["total"] += total_loss.item()
        epoch_losses["img"]   += loss_img.item()
        epoch_losses["adv"]   += loss_adv.item()
        epoch_losses["det"]   += loss_det.item()

    n_train = len(train_loader)
    for k in epoch_losses:
        history[f"train_{k}"].append(epoch_losses[k] / n_train)

    # --- Validazione ---
    model.eval()
    val_losses = {"total": 0.0, "img": 0.0, "adv": 0.0, "det": 0.0}
    saved_preview = False

    with torch.no_grad():
        for wm_imgs, clean_imgs in val_loader:
            wm_imgs    = wm_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            reconstructed_imgs, logits_reconstructed, detection_score = model(
                wm_imgs, detector=detector
            )

            total_val, loss_img, loss_adv, loss_det = compute_loss(
                reconstructed_imgs, detection_score, logits_reconstructed,
                clean_imgs, wm_imgs, W_IMG, w_adv_cur, w_det_cur
            )

            if num_gpus > 1:
                total_val = total_val.mean()
                loss_img  = loss_img.mean()
                loss_adv  = loss_adv.mean()
                loss_det  = loss_det.mean()

            val_losses["total"] += total_val.item()
            val_losses["img"]   += loss_img.item()
            val_losses["adv"]   += loss_adv.item()
            val_losses["det"]   += loss_det.item()

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
        f"Ep [{ep:3d}/{EPOCHS}] | w_adv={w_adv_cur:.2f} w_det={w_det_cur:.2f} | "
        f"Train → total={history['train_total'][-1]:.4f}  img={history['train_img'][-1]:.5f}  "
        f"adv={history['train_adv'][-1]:.4f}  det={history['train_det'][-1]:.4f} | "
        f"Val → total={cur_val_total:.4f}  img={history['val_img'][-1]:.5f}  "
        f"adv={history['val_adv'][-1]:.4f}  det={history['val_det'][-1]:.4f}"
    )

    if cur_val_total < best_val_loss:
        best_val_loss     = cur_val_total
        state_to_save     = model.module.state_dict() if num_gpus > 1 else model.state_dict()
        torch.save(state_to_save, "checkpoints/unet_best.pth")
        print("  -> Nuovo record. Pesi salvati.")

print("\nAddestramento Completato.")

# ==========================================
# 5. CSV
# ==========================================
pd.DataFrame({
    "Epoca":           range(1, EPOCHS + 1),
    "Train_Total":     history["train_total"],
    "Train_Img_L1":    history["train_img"],
    "Train_Adv_MSE":   history["train_adv"],
    "Train_Det_MSE":   history["train_det"],
    "Val_Total":       history["val_total"],
    "Val_Img_L1":      history["val_img"],
    "Val_Adv_MSE":     history["val_adv"],
    "Val_Det_MSE":     history["val_det"],
}).to_csv("unet_separated_losses.csv", index=False, sep=";")

# ==========================================
# 6. GRAFICO A TRE PANNELLI
# ==========================================
epochs_axis = range(1, EPOCHS + 1)
fig, axes = plt.subplots(3, 1, figsize=(12, 14))

# Pannello 1: Loss totale
ax = axes[0]
ax.plot(epochs_axis, history["train_total"], label="Train Total", color="purple", lw=2)
ax.plot(epochs_axis, history["val_total"],   label="Val Total",   color="purple", lw=2, ls="--")
ax.set_ylabel("Loss Totale")
ax.set_title("Loss Totale Combinata")
ax.legend(); ax.grid(True, ls=":")

# Pannello 2: Componente Image (L1)
ax = axes[1]
ax.plot(epochs_axis, history["train_img"], label="Train Img (L1)", color="blue", lw=2)
ax.plot(epochs_axis, history["val_img"],   label="Val Img (L1)",   color="blue", lw=2, ls="--")
ax.set_ylabel("L1 Loss (Image Fidelity)")
ax.set_title("Fidelity dell'Immagine (deve scendere e stabilizzarsi)")
ax.legend(); ax.grid(True, ls=":")

# Pannello 3: Componenti avversariali
ax = axes[2]
ax.plot(epochs_axis, history["train_adv"], label="Train Bit-Logit (MSE)", color="orange", lw=2)
ax.plot(epochs_axis, history["val_adv"],   label="Val Bit-Logit (MSE)",   color="orange", lw=2, ls="--")
ax.plot(epochs_axis, history["train_det"], label="Train Detection Score", color="red", lw=2)
ax.plot(epochs_axis, history["val_det"],   label="Val Detection Score",   color="red", lw=2, ls="--")
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss Avversariale")
ax.set_title("Componenti Avversariali (devono SCENDERE)")
ax.legend(); ax.grid(True, ls=":")

plt.tight_layout()
plt.savefig("unet_separated_loss_plot.png", dpi=300)
plt.close()

print("✅ CSV e grafico a tre pannelli salvati con successo.")