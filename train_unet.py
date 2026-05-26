import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import pandas as pd
import matplotlib.pyplot as plt

# Importiamo il dataloader e il modello
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
# 3. INIZIALIZZAZIONE MODELLO E LOSS MSE
# ==========================================
model = UNetDenoiseAttack(in_channels=3, out_channels=3).to(device)
criterion = nn.MSELoss().to(device)

if num_gpus > 1:
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

os.makedirs("checkpoints", exist_ok=True)

# ==========================================
# 4. STORICO DEL TRAINING (Le nostre liste History)
# ==========================================
train_loss_history = []
val_loss_history = []

# ==========================================
# 5. TRAINING LOOP
# ==========================================
print("\nInizio Addestramento con Loss MSE...\n")

best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    
    for wm_imgs, clean_imgs in train_loader:
        wm_imgs = wm_imgs.to(device)
        clean_imgs = clean_imgs.to(device)
        
        optimizer.zero_grad()
        
        reconstructed_imgs = model(wm_imgs)
        loss = criterion(reconstructed_imgs, clean_imgs)
        
        if num_gpus > 1:
            loss = loss.mean()
            
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
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
            loss = criterion(reconstructed_imgs, clean_imgs)
            
            if num_gpus > 1:
                loss = loss.mean()
                
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    val_loss_history.append(avg_val_loss)
    
    print(f"Epoca [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        state_dict_to_save = model.module.state_dict() if num_gpus > 1 else model.state_dict()
        torch.save(state_dict_to_save, "checkpoints/unet_best.pth")
        print("Nuovo record: Modello salvato.")

print("\nAddestramento Completato.")

# ==========================================
# 6. SALVATAGGIO SUMMARY (File CSV)
# ==========================================
print("\nSalvataggio del Summary in corso...")
summary_df = pd.DataFrame({
    "Epoca": range(1, EPOCHS + 1),
    "Train_Loss": train_loss_history,
    "Val_Loss": val_loss_history
})
summary_file = "checkpoints/unet_summary.csv"
summary_df.to_csv(summary_file, index=False, sep=";")
print(f"Summary salvato con successo in: {summary_file}")

# ==========================================
# 7. GENERAZIONE E SALVATAGGIO GRAFICO
# ==========================================
print("Generazione del grafico della Loss...")
fig, ax1 = plt.subplots(figsize=(10, 8))

line1, = ax1.plot(range(1, EPOCHS + 1), train_loss_history, label='train_loss', color='orange')
ax1.plot(range(1, EPOCHS + 1), val_loss_history, label='val_loss', color=line1.get_color(), linestyle='--')

ax1.set_xlim([1, EPOCHS])
ax1.set_ylim([0, max(max(train_loss_history), max(val_loss_history)) * 1.1]) 
ax1.set_ylabel('Loss', color=line1.get_color())
ax1.tick_params(axis='y', labelcolor=line1.get_color())
ax1.set_xlabel('Epochs')
ax1.grid(True, linestyle=":")
ax1.legend(loc='upper right')
ax1.set_title('Andamento della Loss (MSE) - UNet')

plot_file = "checkpoints/unet_loss_plot.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Grafico salvato con successo in: {plot_file}")
print("="*50)