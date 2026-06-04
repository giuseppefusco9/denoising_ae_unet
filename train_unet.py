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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
 
print("="*50)
print(f"Dispositivo Principale: {device}")
print(f"Numero di GPU rilevate: {num_gpus}")
print("="*50)
 
BATCH_SIZE = 32      
EPOCHS = 60          
LEARNING_RATE = 2e-4 
CROP_SIZE = 256      
 
MAX_GRAD_NORM = 1.0
 
# ==========================================
# 2. PREPARAZIONE DATI
# ==========================================
print("Caricamento dataset...")
train_dataset = WatermarkDenoisingDataset(root_dir="dataset_minSize/train", crop_size=CROP_SIZE)
val_dataset = WatermarkDenoisingDataset(root_dir="dataset_minSize/val", crop_size=CROP_SIZE)
 
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
 
# ==========================================
# 3. INIZIALIZZAZIONE MODELLI E DIRECTORY
# ==========================================
model = UNetDenoiseAttack(in_channels=3, out_channels=3).to(device)
 
print("Caricamento detector PixelSeal...")
detector = videoseal.load("pixelseal").to(device)
detector.eval()
for param in detector.parameters():
    param.requires_grad = False
 
criterion_img = nn.L1Loss().to(device)
criterion_logits = nn.MSELoss().to(device)
 
if num_gpus > 1:
    model = nn.DataParallel(model)
 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
 
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("checkpoints/progress_images", exist_ok=True)
 
history = {"train_total": [], "val_total": []}
 
# ==========================================
# 4. TRAINING LOOP CON LOSS SCHEDULER
# ==========================================
print(f"\nInizio Addestramento con Loss Scheduling e Gradient Clipping...\n")
 
best_val_loss = float('inf')
 
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    # --- CALCOLO DINAMICO DEI PESI (LOSS SCHEDULER) ---
    current_epoch_1based = epoch + 1
    if current_epoch_1based <= 15:
        # Fase 1 (Epoche 1-15): 100% Immagine, 0% Detector
        alpha = 16.0
        lambda_val = 0.0
    elif current_epoch_1based <= 45:
        # Fase 2 (Epoche 16-45): Transizione lineare progressiva verso 80-20
        # Calcoliamo un fattore di interpolazione che va da 0 a 1 nei 30 passi della transizione
        t = (current_epoch_1based - 15) / 30.0
        alpha = 16.0 - (16.0 - 12.8) * t
        lambda_val = 0.0 + (0.09 - 0.0) * t
    else:
        # Fase 3 (Epoche 46-60): Consolidamento fisso a quota 80-20
        alpha = 12.8
        lambda_val = 0.09
 
    for wm_imgs, clean_imgs in train_loader:
        wm_imgs = wm_imgs.to(device)
        clean_imgs = clean_imgs.to(device)
        optimizer.zero_grad()
        reconstructed_imgs, logits_reconstructed = model(wm_imgs, detector=detector)
        loss_fidelity = criterion_img(reconstructed_imgs, clean_imgs)
        with torch.no_grad():
            outputs_clean = detector.detect(clean_imgs)
            logits_clean_target = outputs_clean["preds"][:, 1:].detach()
        loss_adv = criterion_logits(logits_reconstructed, logits_clean_target)
        # Applicazione dei pesi schedulati dell'epoca corrente
        total_loss = (alpha * loss_fidelity) + (lambda_val * loss_adv)
        if num_gpus > 1:
            total_loss = total_loss.mean()
        total_loss.backward()
        # Gradient Clipping per normalizzare la risalita da ConvNeXtV2
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()
        train_loss += total_loss.item()
    avg_train_loss = train_loss / len(train_loader)
    history["train_total"].append(avg_train_loss)
    # --- VALIDAZIONE ED ESTRAZIONE IMMAGINI INTERMEDIE ---
    model.eval()
    val_loss = 0.0
    # Variabile di supporto per salvare una sola immagine di progresso a epoca
    saved_preview_this_epoch = False
    with torch.no_grad():
        for wm_imgs, clean_imgs in val_loader:
            wm_imgs = wm_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            reconstructed_imgs, logits_reconstructed = model(wm_imgs, detector=detector)
            loss_fidelity = criterion_img(reconstructed_imgs, clean_imgs)
            outputs_clean = detector.detect(clean_imgs)
            logits_clean_target = outputs_clean["preds"][:, 1:].detach()
            loss_adv = criterion_logits(logits_reconstructed, logits_clean_target)
            total_val_loss = (alpha * loss_fidelity) + (lambda_val * loss_adv)
            if num_gpus > 1:
                total_val_loss = total_val_loss.mean()
            val_loss += total_val_loss.item()
            # --- ISPEZIONE VISIVA INTERMEDIA ---
            # Salva il primo campione del primo batch di validazione dell'epoca corrente
            if not saved_preview_this_epoch:
                # Creiamo una griglia con: Immagine con Watermark | Target Pulito | Output U-Net
                preview_grid = torch.cat([wm_imgs[0:1], clean_imgs[0:1], reconstructed_imgs[0:1]], dim=0)
                vutils.save_image(
                    preview_grid, 
                    f"checkpoints/progress_images/epoch_{current_epoch_1based:02d}_alpha{alpha:.1f}_lamb{lambda_val:.3f}.png",
                    normalize=True
                )
                saved_preview_this_epoch = True
    avg_val_loss = val_loss / len(val_loader)
    history["val_total"].append(avg_val_loss)
    print(f"Epoca [{current_epoch_1based}/{EPOCHS}] | Pesi: Alpha={alpha:.2f}, Lambda={lambda_val:.4f}")
    print(f"  -> Train Loss Riscalata: {avg_train_loss:.5f} | Val Loss Riscalata: {avg_val_loss:.5f}")
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        state_dict_to_save = model.module.state_dict() if num_gpus > 1 else model.state_dict()
        torch.save(state_dict_to_save, "checkpoints/unet_best.pth")
        print("  [Record registrato: Pesi della U-Net salvati]")
 
print("\nAddestramento Completato.")
 
# ==========================================
# 5. ESPORTAZIONE LOG CSV
# ==========================================
summary_df = pd.DataFrame({
    "Epoca": range(1, EPOCHS + 1),
    "Train_Total_Loss": history["train_total"],
    "Val_Total_Loss": history["val_total"]
})
summary_df.to_csv("unet_summary.csv", index=False, sep=";")
 
# ==========================================
# 6. GENERAZIONE GRAFICO CLASSICO A PANNELLO SINGOLO
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 8))
line1, = ax1.plot(range(1, EPOCHS + 1), history["train_total"], label='Train Loss', color='blue', linewidth=2.5)
ax1.plot(range(1, EPOCHS + 1), history["val_total"], label='Val Loss', color=line1.get_color(), linestyle='--')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss Complessiva Schedulata')
ax1.grid(True, linestyle=":")
ax1.legend(loc='upper right')
ax1.set_title('Andamento Loss con Schedulazione Progressiva e Gradient Clipping')
plt.savefig("unet_loss_plot.png", dpi=300, bbox_inches='tight')
plt.close()
 
print("✅ Pipeline conclusa. Grafico nominale rigenerato in 'unet_loss_plot.png'!")