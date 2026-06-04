import torch
import torch.nn as nn
from torchinfo import summary
import videoseal

from unet_attack_model import UNetDenoiseAttack

# Wrapper strutturato ad uso esclusivo di torchinfo per mappare l'albero gerarchico completo
class AttackAndEvaluationWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNetDenoiseAttack(in_channels=3, out_channels=3)
        self.detector = videoseal.load("pixelseal")
        self.detector.eval()
        for param in self.detector.parameters():
            param.requires_grad = False

    def forward(self, x):
        # CORRETTO: Spacchettiamo tre argomenti per allinearci al nuovo metodo residuale della U-Net
        reconstructed_imgs, detected_bit_logits, detection_score = self.unet(x, detector=self.detector)
        return reconstructed_imgs, detected_bit_logits, detection_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_to_analyze = AttackAndEvaluationWrapper().to(device)

# Creazione del report tabellare txt
model_summary = summary(
    model_to_analyze, 
    input_size=(32, 3, 256, 256), 
    col_names=["input_size", "output_size", "num_params", "kernel_size"],
    row_settings=["depth", "var_names"],
    depth=4,
    verbose=0
)

with open("summary.txt", "w", encoding="utf-8") as f:
    f.write(str(model_summary))

print("✅ Tabella strutturale esportata correttamente nel file 'summary.txt'!")