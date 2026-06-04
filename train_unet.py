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
print("="*50)

BATCH_SIZE = 32      
EPOCHS = 60          
LEARNING_RATE = 2e-4 
CROP_SIZE = 256      

# IPERPARAMETRI DI BILANCIAMENTO
ALPHA = 0.0   # Peso per il confronto pixel-by-pixel delle immagini
LAMBDA = 1.0  # Peso per l'allineamento dei logiti latenti (Attacco)

# ==========================================
# 2. PREPARAZIONE DATI
# ==========================================
print("Caricamento dataset in corso...")
train_dataset = WatermarkDenoisingDataset(root_dir="dataset_minSize/train", crop_size=CROP_SIZE)
val_dataset = WatermarkDenoisingDataset(root_dir="dataset_minSize/val", crop_size=CROP_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"Dati Caricati: {len(train_dataset)} Train | {len(val_dataset)} Val.")

# ==========================================
# 3. INIZIALIZZAZIONE MODELLI E FUNZIONI DI COSTO
# ==========================================
model = UNetDenoiseAttack(in_channels=3, out_channels=3).to(device)

print("Caricamento detector PixelSeal...")
detector = videoseal.load("pixelseal")
detector.to(device)
detector.eval()

for param in detector.parameters():
    param.requires_grad = False

# Entrambe le anime della loss usano la L1Loss per garantire stabilità e linearità
criterion_img = nn.L1Loss().to(device)
criterion_logits = nn.L1Loss().to(device)

if num_gpus > 1:
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
os.makedirs("checkpoints", exist_ok=True)

train_loss_history = []
val_loss_history = []

# ==========================================
# 5. TRAINING LOOP (ALLINEAMENTO LOGITI PULITI)
# ==========================================
print(f"\nInizio Addestramento (L1 Immagine * {ALPHA} + L1 Logiti Puliti * {LAMBDA})...\n")

best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    
    for wm_imgs, clean_imgs in train_loader:
        wm_imgs = wm_imgs.to(device)
        clean_imgs = clean_imgs.to(device)
        
        optimizer.zero_grad()
        
        # 1. Passaggio nella U-Net (Lavora interamente in RGB)
        reconstructed_imgs = model(wm_imgs)
        
        # 2. Loss di fedeltà visiva tra l'immagine d'uscita e l'originale pulita
        loss_fidelity = criterion_img(reconstructed_imgs, clean_imgs)
        
        # 3. Estrazione dei logiti dei bit dal campione d'uscita (indice 1 in poi)
        outputs_reconstructed = detector.detect(reconstructed_imgs)
        logits_reconstructed = outputs_reconstructed["preds"][:, 1:]
        
        # 4. Estrazione dei logiti dall'immagine originale pulita (SENZA GRADIENTI)
        # Questa firma rappresenta lo "stato di zero" o di puro rumore casuale
        with torch.no_grad():
            outputs_clean = detector.detect(clean_imgs)
            logits_clean_target = outputs_clean["preds"][:, 1:].detach()
        
        # 5. Calcolo della Loss Avversariale: forziamo i logiti sotto attacco 
        # a collassare sopra il profilo dei logiti della mappa pulita originaria
        loss_adv = criterion_logits(logits_reconstructed, logits_clean_target)
        
        # Combinazione lineare delle Loss
        total_loss = (ALPHA * loss_fidelity) + (LAMBDA * loss_adv)
            
        total_loss.backward()
        optimizer.step()
        
        train_loss += total_loss.item()
        
    avg_train_loss = train_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)
    
    # --- VALIDAZIONE ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for wm_imgs, clean_imgs in val_loader:
            wm_imgs = wm_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            reconstructed_imgs = model(wm_imgs)
            loss_fidelity = criterion_img(reconstructed_imgs, clean_imgs)
            
            outputs_reconstructed = detector.detect(reconstructed_imgs)
            logits_reconstructed = outputs_reconstructed["preds"][:, 1:]
            
            outputs_clean = detector.detect(clean_imgs)
            logits_clean_target = outputs_clean["preds"][:, 1:].detach()
            
            loss_adv = criterion_logits(logits_reconstructed, logits_clean_target)
            
            total_val_loss = (ALPHA * loss_fidelity) + (LAMBDA * loss_adv)
            if num_gpus > 1:
                total_val_loss = total_val_loss.mean()
            val_loss += total_val_loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    val_loss_history.append(avg_val_loss)
    
    print(f"Epoca [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        state_dict_to_save = model.module.state_dict() if num_gpus > 1 else model.state_dict()
        torch.save(state_dict_to_save, "checkpoints/unet_best.pth")
        print("Nuovo record registrato. Modello salvato.")

print("\nAddestramento Completato.")

# ==========================================
# 6. EXPORT LOG E GRAFICI
# ==========================================
summary_df = pd.DataFrame({
    "Epoca": range(1, EPOCHS + 1),
    "Train_Loss": train_loss_history,
    "Val_Loss": val_loss_history
})
summary_df.to_csv("unet_summary.csv", index=False, sep=";")

fig, ax1 = plt.subplots(figsize=(10, 8))
line1, = ax1.plot(range(1, EPOCHS + 1), train_loss_history, label='Train Loss', color='blue')
ax1.plot(range(1, EPOCHS + 1), val_loss_history, label='Val Loss', color=line1.get_color(), linestyle='--')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss Combinata (L1 Img + L1 Logiti Puliti)')
ax1.grid(True, linestyle=":")
ax1.legend(loc='upper right')
ax1.set_title('Andamento Loss con Allineamento Diretto nello Spazio Latente dei Logiti')
plt.savefig("unet_loss_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print("Grafici salvati.")