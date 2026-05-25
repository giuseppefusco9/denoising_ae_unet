import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import os
import pandas as pd
import matplotlib.pyplot as plt

# Importiamo il dataloader e il modello
from loader_dataset import WatermarkDenoisingDataset
from unet_attack_model import UNetDenoiseAttack

# ==========================================
# 0. DEFINIZIONE DELLA PERCEPTUAL LOSS (VGG)
# ==========================================
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Carica i pesi di una VGG-16 pre-addestrata su ImageNet
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).eval()
        # Estrae i primi layer (fino al secondo blocco convoluzionale, prima del pooling)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:9])
        # Congela i gradienti della VGG (modulo puramente valutativo)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        features_x = self.feature_extractor(x)
        features_y = self.feature_extractor(y)
        # Distanza Euclidea nello spazio delle caratteristiche profonde
        return nn.functional.mse_loss(features_x, features_y)

# ==========================================
# 1. CONFIGURAZIONE HARDWARE E PARAMETRI
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

print("="*50)
print(f"Dispositivo Principale: {device}")
print(f"GPU Disponibili rilevate: {num_gpus}")
print("="*50)

BATCH_SIZE = 32      
EPOCHS = 60          
LEARNING_RATE = 2e-4 
CROP_SIZE = 256      

# Pesi di bilanciamento della Multi-Task Loss
ALPHA = 1.0  # Peso associato alla precisione dei pixel (MSE)
BETA = 0.5   # Peso associato alla nitidezza e semantica (Perceptual)

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
# 3. INIZIALIZZAZIONE MODELLO E FUNZIONI DI COSTO
# ==========================================
model = UNetDenoiseAttack(in_channels=3, out_channels=3).to(device)

# Istanziazione delle funzioni di costo sul dispositivo principale
criterion_pixel = nn.MSELoss().to(device)
criterion_perceptual = PerceptualLoss().to(device)

if num_gpus > 1:
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

os.makedirs("checkpoints", exist_ok=True)

# ==========================================
# 4. STORICO DEL TRAINING (Liste History)
# ==========================================
train_loss_history = []
val_loss_history = []

# ==========================================
# 5. TRAINING LOOP (Loss Combinata)
# ==========================================
print(f"\nInizio Addestramento Multi-Task Loss (MSE * {ALPHA} + Perceptual * {BETA})...\n")

best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    
    for wm_imgs, clean_imgs in train_loader:
        wm_imgs = wm_imgs.to(device)
        clean_imgs = clean_imgs.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass attraverso la U-Net
        reconstructed_imgs = model(wm_imgs)
        
        # Calcolo combinato della Loss
        loss_pixel = criterion_pixel(reconstructed_imgs, clean_imgs)
        loss_perceptual = criterion_perceptual(reconstructed_imgs, clean_imgs)
        
        total_loss = (ALPHA * loss_pixel) + (BETA * loss_perceptual)
        
        # Nel caso di DataParallel, riduciamo a scalare se necessario
        if num_gpus > 1:
            total_loss = total_loss.mean()
            
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
            
            loss_pixel = criterion_pixel(reconstructed_imgs, clean_imgs)
            loss_perceptual = criterion_perceptual(reconstructed_imgs, clean_imgs)
            
            total_val_loss = (ALPHA * loss_pixel) + (BETA * loss_perceptual)
            
            if num_gpus > 1:
                total_val_loss = total_val_loss.mean()
                
            val_loss += total_val_loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    val_loss_history.append(avg_val_loss)
    
    print(f"Epoca [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
    
    # Salvataggio dei pesi basato sulla combinazione ottima delle loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        state_dict_to_save = model.module.state_dict() if num_gpus > 1 else model.state_dict()
        torch.save(state_dict_to_save, "checkpoints/unet_best.pth")
        print("Nuovo record di validazione: Modello salvato.")

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

line1, = ax1.plot(range(1, EPOCHS + 1), train_loss_history, label='Train Loss (Total)', color='orange')
ax1.plot(range(1, EPOCHS + 1), val_loss_history, label='Val Loss (Total)', color=line1.get_color(), linestyle='--')

ax1.set_xlim([1, EPOCHS])
ax1.set_ylim([0, max(max(train_loss_history), max(val_loss_history)) * 1.1]) 
ax1.set_ylabel('Loss Combinata', color=line1.get_color())
ax1.tick_params(axis='y', labelcolor=line1.get_color())
ax1.set_xlabel('Epochs')
ax1.grid(True, linestyle=":")
ax1.legend(loc='upper right')
ax1.set_title('Andamento Multi-Task Loss (MSE + Perceptual VGG) - U-Net')

plot_file = "checkpoints/unet_loss_plot.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Grafico salvato con successo in: {plot_file}")
print("="*50)