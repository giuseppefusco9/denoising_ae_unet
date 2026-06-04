import torch
from torchinfo import summary

# Importiamo la tua U-Net RGB con detector incapsulato
from unet_attack_model import UNetDenoiseAttack

# ==========================================
# 1. CONFIGURAZIONE E INIZIALIZZAZIONE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inizializziamo direttamente il modello reale
model_to_analyze = UNetDenoiseAttack(in_channels=3, out_channels=3).to(device)

# ==========================================
# 2. GENERAZIONE DEL REPORT 
# ==========================================
# Manteniamo le impostazioni avanzate:
# - row_settings=["depth", "var_names"] ci permette di tracciare le skip connection 
#   tramite i nomi delle variabili (x1..x5, t1..t5)
# - depth=4 scende in profondità sia nei sotto-blocchi DoubleConv che nel grafo di PixelSeal
model_summary = summary(
    model_to_analyze, 
    input_size=(32, 3, 256, 256), 
    col_names=["input_size", "output_size", "num_params", "kernel_size"],
    row_settings=["depth", "var_names"],
    depth=4,
    verbose=0
)

# ==========================================
# 3. SCRITTURA DEL FILE SUMMARY.TXT
# ==========================================
with open("summary.txt", "w", encoding="utf-8") as f:
    f.write(str(model_summary))

print("✅ File summary.txt generato con successo direttamente dal modello integrato!")