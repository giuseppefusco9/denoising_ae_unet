import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F

class WatermarkDenoisingDataset(Dataset):
    def __init__(self, root_dir):
        self.clean_dir = os.path.join(root_dir, 'clean_img')
        self.wm_dir = os.path.join(root_dir, 'wm_img')
        self.valid_pairs = []

        if not os.path.exists(self.clean_dir) or not os.path.exists(self.wm_dir):
            print(f"⚠️ Cartelle non trovate in {root_dir}, dataset vuoto.")
            return

        clean_files = os.listdir(self.clean_dir)
        wm_files = os.listdir(self.wm_dir)

        for c_file in clean_files:
            if c_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                matching_wm = [w for w in wm_files if c_file in w]
                if matching_wm:
                    self.valid_pairs.append((c_file, matching_wm[0]))

        self.valid_pairs.sort()
        print(f"Trovate {len(self.valid_pairs)} coppie di immagini valide nel dataset ({root_dir}).")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        clean_name, wm_name = self.valid_pairs[idx]

        clean_path = os.path.join(self.clean_dir, clean_name)
        wm_path = os.path.join(self.wm_dir, wm_name)

        clean_img = Image.open(clean_path).convert("RGB")
        wm_img = Image.open(wm_path).convert("RGB")

        # Le immagini sul disco sono già 512x512 centrate dalla nuova pipeline.
        # Rimuoviamo il RandomCrop dinamico per mantenere i volumi intatti e coerenti.
        clean_tensor = F.to_tensor(clean_img)
        wm_tensor = F.to_tensor(wm_img)

        return wm_tensor, clean_tensor

# ==========================================
# CORREZIONE DEL BLOCCO DI TEST EXTRA
# ==========================================
if __name__ == "__main__":
    # Aggiornato il path puntando a dataset_minSize/train che esiste sul server
    mio_dataset = WatermarkDenoisingDataset(root_dir="dataset_minSize/train")
    
    if len(mio_dataset) > 0:
        mio_dataloader = DataLoader(mio_dataset, batch_size=12, shuffle=True)
        for batch_wm, batch_clean in mio_dataloader:
            print(f"✅ Verifica Superata!")
            print(f"  -> Formato Tensore Input (Watermarked): {batch_wm.shape}")
            print(f"  -> Formato Tensore Target (Clean):       {batch_clean.shape}")
            print(f"  -> Range valori: [{batch_wm.min().item():.2f}, {batch_wm.max().item():.2f}]")
            break
