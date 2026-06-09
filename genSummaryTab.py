import torch
import torch.nn as nn
from torchinfo import summary
import videoseal

from unet_attack_model import UNetDenoiseAttack


class AttackAndEvaluationWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.detector = videoseal.load("pixelseal")
        self.detector.eval()
        for param in self.detector.parameters():
            param.requires_grad = False

        self.unet = UNetDenoiseAttack(in_channels=3, out_channels=3, detector=self.detector)

    def forward(self, x):
        reconstructed_imgs, detected_bit_logits = self.unet(x)
        return reconstructed_imgs, detected_bit_logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_to_analyze = AttackAndEvaluationWrapper().to(device)

model_summary = summary(
    model_to_analyze,
    input_size=(12, 3, 256, 256),
    col_names=["input_size", "output_size", "num_params", "kernel_size"],
    row_settings=["depth", "var_names"],
    depth=4,
    verbose=0
)

with open("summary.txt", "w", encoding="utf-8") as f:
    f.write(str(model_summary))

print("✅ Tabella strutturale esportata correttamente nel file 'summary.txt'!")
