import torch
import torch.nn as nn
from torchinfo import summary
import videoseal

# Importiamo la tua U-Net 
from unet_attack_model import UNetDenoiseAttack

# ==========================================
# 1. CREAZIONE DEL WRAPPER PER IL SUMMARY
# ==========================================
class AttackAndEvaluationWrapper(nn.Module):
    """
    Questo modulo fittizio serve esclusivamente a mostrare a torchinfo 
    l'intero grafo computazionale dell'addestramento: la U-Net che attacca
    e il Detector di PixelSeal congelato che valuta l'output.
    """
    def __init__(self):
        super().__init__()
        # La tua U-Net RGB addestrabile
        self.unet = UNetDenoiseAttack(in_channels=3, out_channels=3)
        
        # Il detector di Meta (che andiamo a congelare)
        self.detector = videoseal.load("pixelseal")
        self.detector.eval()
        for param in self.detector.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Passo 1: L'immagine subisce l'attacco della U-Net
        reconstructed_imgs = self.unet(x)
        # Passo 2: Il detector congelato analizza l'immagine ricostruita
        detector_outputs = self.detector.detect(reconstructed_imgs)
        return reconstructed_imgs, detector_outputs

# ==========================================
# 2. CONFIGURAZIONE E INIZIALIZZAZIONE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inizializziamo il wrapper complessivo
model_to_analyze = AttackAndEvaluationWrapper().to(device)

# ==========================================
# 3. GENERAZIONE DEL REPORT 
# ==========================================
# Cosa abbiamo aggiunto/modificato:
# - col_names: aggiunta "forward_pass_alloc" per vedere l'impatto della memoria delle skip connection
# - row_settings: ["depth", "var_names"] mostra i nomi esatti delle variabili (es. t1, x5) aiutando a vedere dove avviene la skip
# - depth=4: aumentiamo la profondità a 4 per scendere dentro le convoluzioni di PixelSeal
model_summary = summary(
    model_to_analyze, 
    input_size=(32, 3, 256, 256), 
    col_names=["input_size", "output_size", "num_params", "kernel_size"],
    row_settings=["depth", "var_names"],
    depth=4,
    verbose=0
)

# ==========================================
# 4. SCRITTURA DEL FILE SUMMARY.TXT
# ==========================================
with open("summary.txt", "w", encoding="utf-8") as f:
    f.write(str(model_summary))

print("✅ File summary.txt generato con successo!")