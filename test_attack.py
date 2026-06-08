import torch
import torchvision.transforms.functional as F
from PIL import Image
import videoseal
import os
import pandas as pd
import math
from unet_attack_model import UNetDenoiseAttack

# ==========================================
# 1. CONFIGURAZIONE DEL TEST
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIR = "dataset_minSize/test" 

OUTPUT_DIR = "results_unet"
CSV_FILE = "risultati_unet_raptor.csv"
msg_size = 256

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. CARICAMENTO E PULIZIA PESI MULTI-GPU
# ==========================================
print("Caricamento UNet su raptor01...")
model = UNetDenoiseAttack(in_channels=3, out_channels=3).to(device)

# Carichiamo i pesi addestrati
state_dict = torch.load("checkpoints/unet_best.pth", map_location=device, weights_only=True)

from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k 
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.eval()

print("Caricamento PixelSeal per la verifica...")
pixelseal = videoseal.load("pixelseal").eval()

# ==========================================
# 3. LOOP DI INFERENZA AD ALTA RISOLUZIONE
# ==========================================
clean_dir = os.path.join(INPUT_DIR, 'clean_img')
wm_dir = os.path.join(INPUT_DIR, 'wm_img')

clean_files = [f for f in os.listdir(clean_dir) if not f.startswith('.')]
wm_files = [f for f in os.listdir(wm_dir) if not f.startswith('.')]

# Logica di accoppiamento corretta
valid_pairs = []
for c_file in clean_files:
    matching_wm = [w for w in wm_files if c_file in w]
    if matching_wm:
        valid_pairs.append((c_file, matching_wm[0]))

results = []

print(f"Inizio test su {len(valid_pairs)} immagini...")

for idx, (clean_name, wm_name) in enumerate(valid_pairs):
    print(f"--- Processing [{idx+1}/{len(valid_pairs)}]: {clean_name} ---")
    wm_path = os.path.join(wm_dir, wm_name)
    clean_path = os.path.join(clean_dir, clean_name)
    
    img_wm = Image.open(wm_path).convert("RGB")
    img_clean = Image.open(clean_path).convert("RGB")
    
    # Rimuoviamo la logica di padding math.ceil(w/16)*16 e usiamo il tensore nativo 512x512
    img_wm_tensor = F.to_tensor(img_wm).unsqueeze(0).to(device)
    img_clean_tensor = F.to_tensor(img_clean).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # --- A. Analisi REALE Immagine Pulita ---
        det_clean = pixelseal.detect(img_clean_tensor.cpu())
        logit_clean = det_clean["preds"][:, 0].item()
        bits_clean = (det_clean["preds"][:, 1:] > 0).float()
        
        # --- B. Analisi REALE Immagine Watermarked ---
        det_before = pixelseal.detect(img_wm_tensor.cpu())
        logit_before = det_before["preds"][:, 0].item()
        bits_before = (det_before["preds"][:, 1:] > 0).float()
        
        # --- C. Esecuzione Attacco (Ora lavora su un volume coerente 512x512) ---
        cleaned_tensor = model(img_wm_tensor)
        
        # --- D. Analisi REALE Immagine Attaccata ---
        det_after = pixelseal.detect(cleaned_tensor.cpu())
        logit_after = det_after["preds"][:, 0].item()
        bits_after = (det_after["preds"][:, 1:] > 0).float()
        
        # Calcolo Bit Accuracy Reali
        bit_acc_clean = (bits_clean == bits_before).sum().item() / msg_size
        bit_acc_before = (bits_before == bits_before).sum().item() / msg_size 
        bit_acc_after = (bits_after == bits_before).sum().item() / msg_size

    # Salviamo l'immagine direttamente senza dover tagliare via il padding
    final_img = F.to_pil_image(cleaned_tensor[0].cpu())
    final_img.save(os.path.join(OUTPUT_DIR, f"cleaned_{clean_name}"))
    
    # --------------------------------------------------
    # SALVATAGGIO DEI DATI 100% REALI
    # --------------------------------------------------
    results.append({
        "nomeImg": clean_name,
        "modello usato": "pixelseal",
        "categoria": "img_dirette",
        "stato": "Pulita",
        "bit accuracy": round(bit_acc_clean, 4),
        "wm presence (logit c0)": round(logit_clean, 4)
    })

    results.append({
        "nomeImg": clean_name,
        "modello usato": "pixelseal",
        "categoria": "img_dirette",
        "stato": "Watermarked",
        "bit accuracy": round(bit_acc_before, 4),
        "wm presence (logit c0)": round(logit_before, 4)
    })

    results.append({
        "nomeImg": clean_name,
        "modello usato": "pixelseal",
        "categoria": "img_dirette",
        "stato": "Attaccata",
        "bit accuracy": round(bit_acc_after, 4),
        "wm presence (logit c0)": round(logit_after, 4)
    })

pd.DataFrame(results).to_csv(CSV_FILE, index=False, sep=';')
print(f"\nTest completato! CSV salvato in {CSV_FILE}")