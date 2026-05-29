import torch
from torchinfo import summary
from unet_attack_model import UNetDenoiseAttack

# 1. Configurazione hardware minima
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Inizializzazione del modello
model = UNetDenoiseAttack(in_channels=3, out_channels=3).to(device)

# 3. Generazione del report
model_summary = summary(
    model, 
    input_size=(32, 3, 256, 256), 
    col_names=["input_size", "output_size", "num_params", "kernel_size"],
    depth=3,
    verbose=0
)

# 4. Scrittura del file summary.txt
with open("summary.txt", "w", encoding="utf-8") as f:
    f.write(str(model_summary))

print("✅ File summary.txt generato con successo in 'summary.txt'!")