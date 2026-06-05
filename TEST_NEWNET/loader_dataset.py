import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import torchvision.transforms as T

class WatermarkDenoisingDataset(Dataset):
    def __init__(self, root_dir, crop_size=256):
        self.clean_dir = os.path.join(root_dir, 'clean_img')
        self.wm_dir = os.path.join(root_dir, 'wm_img')
        self.crop_size = crop_size
        self.valid_pairs = []

        # Otteniamo le liste dei file
        clean_files = os.listdir(self.clean_dir)
        wm_files = os.listdir(self.wm_dir)

        for c_file in clean_files:
            # Cerca il corrispondente watermarked
            matching_wm = [w for w in wm_files if c_file in w]
            
            if matching_wm:
                # Salva la tupla con (nome_pulita, nome_watermarked)
                self.valid_pairs.append((c_file, matching_wm[0]))

        self.valid_pairs.sort()
        print(f"Trovate {len(self.valid_pairs)} coppie di immagini valide nel dataset.")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        clean_name, wm_name = self.valid_pairs[idx]

        clean_path = os.path.join(self.clean_dir, clean_name)
        wm_path = os.path.join(self.wm_dir, wm_name)

        clean_img = Image.open(clean_path).convert("RGB")
        wm_img = Image.open(wm_path).convert("RGB")

        # Ritaglio sincronizzato
        i, j, h, w = T.RandomCrop.get_params(clean_img, output_size=(self.crop_size, self.crop_size))
        clean_cropped = F.crop(clean_img, i, j, h, w)
        wm_cropped = F.crop(wm_img, i, j, h, w)

        clean_tensor = F.to_tensor(clean_cropped)
        wm_tensor = F.to_tensor(wm_cropped)

        return wm_tensor, clean_tensor

# ==========================================
# TEST
# ==========================================
if __name__ == "__main__":
    mio_dataset = WatermarkDenoisingDataset(root_dir="dataset", crop_size=256)
    
    if len(mio_dataset) > 0:
        mio_dataloader = DataLoader(mio_dataset, batch_size=4, shuffle=True)
        for batch_wm, batch_clean in mio_dataloader:
            print(f"Formato Tensore Input (Watermarked): {batch_wm.shape}")
            print(f"Formato Tensore Target (Pulito): {batch_clean.shape}")
            print("Tutto funziona perfettamente! Pronto per l'addestramento.")
            break
    else:
        print("ATTENZIONE: Ancora 0 file trovati. Verifica che la cartella 'dataset' sia dentro 'denoising_ae'.")