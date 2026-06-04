import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import pandas as pd
import matplotlib.pyplot as plt
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
EPOCHS = 100        
LEARNING_RATE = 2e-5 
CROP_SIZE = 256      

# BILANCIAMENTO TARGET (80% IMMAGINE - 20% DETECTOR)
ALPHA = 0.8 
LAMBDA = 0.2 

# Parametro per il Gradient Clipping (Norma massima del gradiente)
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
# 3. INIZIALIZZAZIONE MODELLI E LOSS
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

# Storici separati per monitorare l'equilibrio del gradiente
history = {
    "train_total": [], "train_img": [], "train_adv": [],
    "val_total": [], "val_img": [], "val_adv": []
}

# ==========================================
# 4. TRAINING LOOP CON GRADIENT CLIPPING
# ==========================================
print(f"\nInizio Addestramento Bilanciato 80/20 (Con Gradient Clipping a max_norm={MAX_GRAD_NORM})...\n")

best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    epoch_losses = {"total": 0.0, "img": 0.0, "adv": 0.0}
    
    for wm_imgs, clean_imgs in train_loader:
        wm_imgs = wm_imgs.to(device)
        clean_imgs = clean_imgs.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass coordinato multi-GPU
        reconstructed_imgs, logits_reconstructed = model(wm_imgs, detector=detector)
        
        # Calcolo delle loss pure (senza iperparametri per lo storico corretto)
        loss_fidelity = criterion_img(reconstructed_imgs, clean_imgs)
        
        with torch.no_grad():
            outputs_clean = detector.detect(clean_imgs)
            logits_clean_target = outputs_clean["preds"][:, 1:].detach()
        
        loss_adv = criterion_logits(logits_reconstructed, logits_clean_target)
        
        # Aggregazione bilanciata 80-20
        total_loss = (ALPHA * loss_fidelity) + (LAMBDA * loss_adv)
        
        if num_gpus > 1:
            total_loss = total_loss.mean()
            loss_fidelity = loss_fidelity.mean()
            loss_adv = loss_adv.mean()
            
        total_loss.backward()
        
        # --- INNOVAZIONE METODOLOGICA: GRADIENT CLIPPING ---
        # Taglia le anomalie provocate dalla risalita dei gradienti attraverso ConvNeXtV2
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        
        optimizer.step()
        
        epoch_losses["total"] += total_loss.item()
        epoch_losses["img"] += loss_fidelity.item()
        epoch_losses["adv"] += loss_adv.item()
        
    num_batches_train = len(train_loader)
    history["train_total"].append(epoch_losses["total"] / num_batches_train)
    history["train_img"].append(epoch_losses["img"] / num_batches_train)
    history["train_adv"].append(epoch_losses["adv"] / num_batches_train)
    
    # --- VALIDAZIONE ---
    model.eval()
    val_epoch_losses = {"total": 0.0, "img": 0.0, "adv": 0.0}
    
    with torch.no_grad():
        for wm_imgs, clean_imgs in val_loader:
            wm_imgs = wm_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            reconstructed_imgs, logits_reconstructed = model(wm_imgs, detector=detector)
            loss_fidelity = criterion_img(reconstructed_imgs, clean_imgs)
            
            outputs_clean = detector.detect(clean_imgs)
            logits_clean_target = outputs_clean["preds"][:, 1:].detach()
            
            loss_adv = criterion_logits(logits_reconstructed, logits_clean_target)
            
            total_val_loss = (ALPHA * loss_fidelity) + (LAMBDA * loss_adv)
            
            if num_gpus > 1:
                total_val_loss = total_val_loss.mean()
                loss_fidelity = loss_fidelity.mean()
                loss_adv = loss_adv.mean()
                
            val_epoch_losses["total"] += total_val_loss.item()
            val_epoch_losses["img"] += loss_fidelity.item()
            val_epoch_losses["adv"] += loss_adv.item()
            
    num_batches_val = len(val_loader)
    history["val_total"].append(val_epoch_losses["total"] / num_batches_val)
    history["val_img"].append(val_epoch_losses["img"] / num_batches_val)
    history["val_adv"].append(val_epoch_losses["adv"] / num_batches_val)
    
    print(f"Epoca [{epoch+1}/{EPOCHS}]")
    print(f"  [TRAIN] Total: {history['train_total'][-1]:.5f} | Img (L1): {history['train_img'][-1]:.5f} | Detector (MSE): {history['train_adv'][-1]:.5f}")
    print(f"  [VAL]   Total: {history['val_total'][-1]:.5f} | Img (L1): {history['val_img'][-1]:.5f} | Detector (MSE): {history['val_adv'][-1]:.5f}")
    
    if history["val_total"][-1] < best_val_loss:
        best_val_loss = history["val_total"][-1]
        state_dict_to_save = model.module.state_dict() if num_gpus > 1 else model.state_dict()
        torch.save(state_dict_to_save, "checkpoints/unet_best.pth")
        print("  -> Nuovo minimo registrato. Modello salvato.")

print("\nAddestramento Completato.")

# ==========================================
# 5. SALVATAGGIO DEI LOG SEPARATI
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
# 6. ELABORAZIONE GRAFICA MULTI-CURVA (CORRETTA)
# ==========================================
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
# Corretto: rimosso fontweight='bold', inserito linewidth=2.5
plt.plot(range(1, EPOCHS + 1), history["train_total"], label='Train Total', color='purple', linewidth=2.5)
plt.plot(range(1, EPOCHS + 1), history["val_total"], label='Val Total', color='purple', linestyle='--')
plt.ylabel('Loss Combinata Riscalata')
plt.grid(True, linestyle=":")
plt.legend()
plt.title('Andamento Regolarizzato della Loss Complessiva')

plt.subplot(2, 1, 2)
plt.plot(range(1, EPOCHS + 1), history["train_img"], label='Train Img (L1)', color='blue')
plt.plot(range(1, EPOCHS + 1), history["val_img"], label='Val Img (L1)', color='blue', linestyle='--')
plt.plot(range(1, EPOCHS + 1), history["train_adv"], label='Train Detector (MSE)', color='orange')
plt.plot(range(1, EPOCHS + 1), history["val_adv"], label='Val Detector (MSE)', color='orange', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Valore Loss Singola Componente (Valore Puro)')
plt.grid(True, linestyle=":")
plt.legend()

plt.tight_layout()
plt.savefig("unet_separated_loss_plot.png", dpi=300)
plt.close()
print("✅ Grafici multi-curva salvati senza errori di rendering in 'unet_separated_loss_plot.png'!")