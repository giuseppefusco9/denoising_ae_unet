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
EPOCHS = 100          # Aggiornato a 100 epoche come richiesto
LEARNING_RATE = 2e-5  # Aggiornato a 2e-5 per una convergenza più dolce
CROP_SIZE = 256      

MAX_GRAD_NORM = 1.0

# --- FATTORI DI NORMALIZZAZIONE STATICA (BASELINE STABILE 1:1) ---
# Compensano lo sbilanciamento di partenza (Loss Img ~0.05, Loss Detector ~2.2)
NORM_IMG = 1.0 / 0.05        
NORM_DETECTOR = 1.0 / 2.2    

# --- TARGET DI BILANCIAMENTO NOMINALE FINALE (80 - 20) ---
TARGET_ALPHA = 0.8
TARGET_LAMBDA = 0.2

# ==========================================
# 2. PREPARAZIONE DATI
# ==========================================
print("Caricamento dataset...")
train_dataset = WatermarkDenoisingDataset(root_dir="dataset_minSize/train", crop_size=CROP_SIZE)
val_dataset = WatermarkDenoisingDataset(root_dir="dataset_minSize/val", crop_size=CROP_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ==========================================
# 3. INIZIALIZZAZIONE MODELLI E COMPONENTI DI COSTO
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

# Storico esteso per supportare la visualizzazione multi-curva del doppio pannello
history = {
    "train_total": [], "train_img": [], "train_adv": [],
    "val_total": [], "val_img": [], "val_adv": []
}

# ==========================================
# 4. TRAINING LOOP CON LOSS SCHEDULER SU 100 EPOCHE
# ==========================================
print(f"\nInizio Addestramento Schedulato (Target Finale: {TARGET_ALPHA*100:.0f}/{TARGET_LAMBDA*100:.0f})...\n")

best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    epoch_losses = {"total": 0.0, "img": 0.0, "adv": 0.0}
    current_epoch_1based = epoch + 1
    
    # --- SCHEDULER RIMODULATO SU CORSA DI 100 EPOCHE ---
    if current_epoch_1based <= 25:
        # Fase 1 (Epoche 1-25): 100% Immagine pura riscalata, 0% Attacco
        current_alpha_target = 1.0
        current_lambda_target = 0.0
    elif current_epoch_1based <= 75:
        # Fase 2 (Epoche 26-75): Transizione lineare distribuita su 50 epoche
        t = (current_epoch_1based - 25) / 50.0
        current_alpha_target = 1.0 - (1.0 - TARGET_ALPHA) * t
        current_lambda_target = 0.0 + (TARGET_LAMBDA - 0.0) * t
    else:
        # Fase 3 (Epoche 76-100): Consolidamento e rifinitura fissa a quota 80-20
        current_alpha_target = TARGET_ALPHA
        current_lambda_target = TARGET_LAMBDA

    # Proiezione dinamica sui coefficienti di scala numerica effettivi
    alpha = current_alpha_target * NORM_IMG
    lambda_val = current_lambda_target * NORM_DETECTOR

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
        
        total_loss = (alpha * loss_fidelity) + (lambda_val * loss_adv)
        
        if num_gpus > 1:
            total_loss = total_loss.mean()
            loss_fidelity = loss_fidelity.mean()
            loss_adv = loss_adv.mean()
            
        total_loss.backward()
        
        # Gradient Clipping a 1.0 per assorbire i contraccolpi di ConvNeXtV2
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        
        optimizer.step()
        
        epoch_losses["total"] += total_loss.item()
        epoch_losses["img"] += loss_fidelity.item()
        epoch_losses["adv"] += loss_adv.item()
        
    num_batches_train = len(train_loader)
    history["train_total"].append(epoch_losses["total"] / num_batches_train)
    history["train_img"].append(epoch_losses["img"] / num_batches_train)
    history["train_adv"].append(epoch_losses["adv"] / num_batches_train)
    
    # --- CICLO DI VALIDAZIONE ED ESTRAZIONE PREVIEW VISIVE ---
    model.eval()
    val_epoch_losses = {"total": 0.0, "img": 0.0, "adv": 0.0}
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
                loss_fidelity = loss_fidelity.mean()
                loss_adv = loss_adv.mean()
                
            val_epoch_losses["total"] += total_val_loss.item()
            val_epoch_losses["img"] += loss_fidelity.item()
            val_epoch_losses["adv"] += loss_adv.item()
            
            # Esportazione del campione visivo a triplo pannello ad ogni fine epoca
            if not saved_preview_this_epoch:
                preview_grid = torch.cat([wm_imgs[0:1], clean_imgs[0:1], reconstructed_imgs[0:1]], dim=0)
                vutils.save_image(
                    preview_grid, 
                    f"checkpoints/progress_images/epoch_{current_epoch_1based:03d}_target_{current_alpha_target:.2f}_{current_lambda_target:.2f}.png",
                    normalize=True
                )
                saved_preview_this_epoch = True
            
    num_batches_val = len(val_loader)
    history["val_total"].append(val_epoch_losses["total"] / num_batches_val)
    history["val_img"].append(val_epoch_losses["img"] / num_batches_val)
    history["val_adv"].append(val_epoch_losses["adv"] / num_batches_val)
    
    print(f"Epoca [{current_epoch_1based}/{EPOCHS}] | Target: {current_alpha_target*100:.1f}% Img - {current_lambda_target*100:.1f}% Det")
    print(f"  [TRAIN] Total Sched: {history['train_total'][-1]:.5f} | Img Puro (L1): {history['train_img'][-1]:.5f} | Det Puro (MSE): {history['train_adv'][-1]:.5f}")
    print(f"  [VAL]   Total Sched: {history['val_total'][-1]:.5f} | Img Puro (L1): {history['val_img'][-1]:.5f} | Det Puro (MSE): {history['val_adv'][-1]:.5f}")
    
    if history["val_total"][-1] < best_val_loss:
        best_val_loss = history["val_total"][-1]
        state_dict_to_save = model.module.state_dict() if num_gpus > 1 else model.state_dict()
        torch.save(state_dict_to_save, "checkpoints/unet_best.pth")
        print("  -> Nuovo record registrato. Pesi salvati correttamente.")

print("\nAddestramento Completato.")

# ==========================================
# 5. SALVATAGGIO DEI LOG SEPARATI IN CSV
# ==========================================
summary_df = pd.DataFrame({
    "Epoca": range(1, EPOCHS + 1),
    "Train_Total_Loss": history["train_total"],
    "Train_Img_Loss": history["train_img"],
    "Train_Adv_Loss": history["train_adv"],
    "Val_Total_Loss": history["val_total"],
    "Val_Img_Loss": history["val_img"],
    "Val_Adv_Loss": history["val_adv"]
})
summary_df.to_csv("unet_separated_losses.csv", index=False, sep=";")

# ==========================================
# 6. GENERAZIONE GRAFICO COMPLETO A DOPPIO PANNELLO REGOLED
# ==========================================
plt.figure(figsize=(12, 10))

# --- SUBPLOT 1: LOSS COMPLESSIVA COMBINATA RISCALATA ---
plt.subplot(2, 1, 1)
plt.plot(range(1, EPOCHS + 1), history["train_total"], label='Train Total Schedulata', color='purple', linewidth=2.5)
plt.plot(range(1, EPOCHS + 1), history["val_total"], label='Val Total Schedulata', color='purple', linestyle='--')
plt.ylabel('Loss Combinata Riscalata')
plt.grid(True, linestyle=":")
plt.legend(loc='upper right')
plt.title('Andamento Regolarizzato della Loss Complessiva Schedulata')

# --- SUBPLOT 2: COMPONENTI PURE SEPARATE (ISOLAMENTO DEGLI ORDINI DI GRANDEZZA) ---
plt.subplot(2, 1, 2)
plt.plot(range(1, EPOCHS + 1), history["train_img"], label='Train Img (L1)', color='blue')
plt.plot(range(1, EPOCHS + 1), history["val_img"], label='Val Img (L1)', color='blue', linestyle='--')
plt.plot(range(1, EPOCHS + 1), history["train_adv"], label='Train Detector (MSE)', color='orange')
plt.plot(range(1, EPOCHS + 1), history["val_adv"], label='Val Detector (MSE)', color='orange', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Valore Loss Componente (Valore Puro)')
plt.grid(True, linestyle=":")
plt.legend(loc='upper right')
plt.title('Analisi di Convergenza Separata delle Componenti')

plt.tight_layout()
plt.savefig("unet_separated_loss_plot.png", dpi=300)
plt.close()

print("✅ File CSV ed esportazione grafica a doppio pannello conclusa con successo in 'unet_separated_loss_plot.png'!")