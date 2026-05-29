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

# IPERPARAMETRI DI BILANCIAMENTO DELLA LOSS MULTI-TASK
ALPHA = 1.0   # Peso per la fedeltà dell'immagine (L1 Loss)
LAMBDA = 0.2  # Peso per l'abbattimento della Bit Accuracy (BCE sui bit)

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

# Congelamento totale del detector (Giudice non addestrabile)
for param in detector.parameters():
    param.requires_grad = False

# Loss di Fedeltà: L1 per preservare i contorni netti dell'immagine RGB
criterion_fidelity = nn.L1Loss().to(device)

# BCE con Logiti come surrogata differenziabile per attaccare i bit della Bit Accuracy
criterion_bit_accuracy = nn.BCEWithLogitsLoss().to(device)

if num_gpus > 1:
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

os.makedirs("checkpoints", exist_ok=True)

# ==========================================
# 4. STORICO DEL TRAINING
# ==========================================
train_loss_history = []
val_loss_history = []

# ==========================================
# 5. TRAINING LOOP AVVERSARIALE (SU BIT ACCURACY)
# ==========================================
print(f"\nInizio Addestramento (L1 Fidelity * {ALPHA} + Soft Bit-Accuracy BCE * {LAMBDA})...\n")

best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    
    for wm_imgs, clean_imgs in train_loader:
        wm_imgs = wm_imgs.to(device)
        clean_imgs = clean_imgs.to(device)
        
        optimizer.zero_grad()
        
        # 1. Forward pass attraverso la U-Net articolata
        reconstructed_imgs = model(wm_imgs)
        
        # 2. Calcolo della Loss di Fedeltà Visiva (Pixel-to-Pixel)
        loss_fid = criterion_fidelity(reconstructed_imgs, clean_imgs)
        
        # 3. Estrazione dei logiti dal detector sull'immagine sotto attacco
        detector_outputs = detector.detect(reconstructed_imgs)
        
        # Allineamento con il tuo script: i bit del messaggio partono dall'indice 1 in poi
        detected_bit_logits = detector_outputs["preds"][:, 1:] 
        
        # Estrazione dei bit target originali dall'immagine di partenza (senza memorizzare i gradienti)
        with torch.no_grad():
            original_outputs = detector.detect(wm_imgs)
            # Prendiamo lo stato dei bit originali mappandoli come probabilità [0, 1] via Sigmoide
            original_probs = torch.sigmoid(original_outputs["preds"][:, 1:])
        
        # STRATEGIA AVVERSARIALE: Inversione del payload.
        # Spingiamo i logiti estratti dalla U-Net verso l'esatto opposto del bit corretto (1.0 - probabilità_reale)
        # Se un bit originale era 1, il target avversariale diventa 0, azzerando la Bit Accuracy finale.
        inverse_payloads = 1.0 - original_probs
        
        # Calcolo della loss surrogata differenziabile sui bit
        loss_adv = criterion_bit_accuracy(detected_bit_logits, inverse_payloads)
        
        # Loss Totale Combinata
        total_loss = (ALPHA * loss_fid) + (LAMBDA * loss_adv)
        
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
            
            loss_fid = criterion_fidelity(reconstructed_imgs, clean_imgs)
            
            detector_outputs = detector.detect(reconstructed_imgs)
            detected_bit_logits = detector_outputs["preds"][:, 1:]
            
            original_outputs = detector.detect(wm_imgs)
            original_probs = torch.sigmoid(original_outputs["preds"][:, 1:])
            inverse_payloads = 1.0 - original_probs
            
            loss_adv = criterion_bit_accuracy(detected_bit_logits, inverse_payloads)
            
            total_val_loss = (ALPHA * loss_fid) + (LAMBDA * loss_adv)
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
        print("Nuovo record di validazione avversariale: Modello salvato.")

print("\nAddestramento Completato.")

# ==========================================
# 6. SALVATAGGIO SUMMARY ED ELABORAZIONE GRAFICA
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
ax1.set_ylabel('Loss Combinata Avversariale')
ax1.grid(True, linestyle=":")
ax1.legend(loc='upper right')
ax1.set_title('Andamento Loss Bilanciata (L1 + Soft Bit-Accuracy BCE) - U-Net')
plt.savefig("unet_loss_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print("Grafici e log salvati con successo.")