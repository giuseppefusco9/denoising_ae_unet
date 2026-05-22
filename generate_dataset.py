import os
import random
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image
import videoseal

# ==========================================
# CONFIGURAZIONE PATH E PARAMETRI
# ==========================================
SOURCE_DIR = "dataset/clean_img"
OUT_ROOT = "dataset_minSize"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)

def main():
    print("🔍 Ricerca della dimensione minima nel dataset...")
    files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(files) == 0:
        print("Errore: Nessuna immagine trovata nella cartella di origine.")
        return
        
    min_size = float('inf')
    for f in files:
        img_path = os.path.join(SOURCE_DIR, f)
        with Image.open(img_path) as img:
            w, h = img.size
            min_size = min(min_size, w, h)
            
    print(f"✅ Dimensione minima trovata: {min_size}x{min_size}")
    random.shuffle(files)
    
    splits = {
        "train": files[:50],
        "val": files[50:75],
        "test": files[75:]
    }

    print(f"🛡️ Caricamento modello PixelSeal su {DEVICE}...")
    try:
        pixelseal = videoseal.load("pixelseal").to(DEVICE).eval()
    except Exception as e:
        print(f"Errore nel caricamento di videoseal: {e}")
        return
    
    for split_name, split_files in splits.items():
        print(f"\n⚙️ Generazione set: {split_name.upper()} ({len(split_files)} immagini base -> {len(split_files)*4} ritagli)")
        
        clean_out_dir = os.path.join(OUT_ROOT, split_name, "clean_img")
        wm_out_dir = os.path.join(OUT_ROOT, split_name, "wm_img")
        os.makedirs(clean_out_dir, exist_ok=True)
        os.makedirs(wm_out_dir, exist_ok=True)
        
        for f in split_files:
            img_path = os.path.join(SOURCE_DIR, f)
            img = Image.open(img_path).convert("RGB")
            base_name, ext = os.path.splitext(f)
            
            for i in range(4):
                top, left, h, w = T.RandomCrop.get_params(img, output_size=(min_size, min_size))
                cropped_img = TF.crop(img, top, left, h, w)
                
                crop_filename = f"{base_name}_crop{i}{ext}"
                clean_save_path = os.path.join(clean_out_dir, crop_filename)
                cropped_img.save(clean_save_path)
                
                # Tensore in ingresso [1, 3, min_size, min_size]
                img_tensor = TF.to_tensor(cropped_img).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    embed_result = pixelseal.embed(img_tensor)
                    
                    wm_tensor = None
                    
                    # ----------------------------------------------------
                    # ESTRAZIONE DELL'OUTPUT
                    # ----------------------------------------------------
                    if isinstance(embed_result, dict):
                        # Cerca dinamicamente un tensore con le stesse dimensioni dell'input
                        for key, value in embed_result.items():
                            if isinstance(value, torch.Tensor) and value.shape == img_tensor.shape:
                                wm_tensor = value
                                break
                            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                                wm_tensor = value[0]
                                break
                        
                        if wm_tensor is None:
                            raise ValueError(f"Tensore non trovato! Chiavi generate da Meta: {list(embed_result.keys())}")
                            
                    elif isinstance(embed_result, torch.Tensor):
                        wm_tensor = embed_result
                    elif isinstance(embed_result, list):
                        wm_tensor = embed_result[0]
                    else:
                        raise TypeError(f"Formato output inatteso: {type(embed_result)}")

                # Assicuriamoci di rimuovere la dimensione batch [1, C, H, W] -> [C, H, W]
                if wm_tensor.dim() == 4 and wm_tensor.shape[0] == 1:
                     wm_tensor = wm_tensor.squeeze(0)

                wm_img_pil = TF.to_pil_image(wm_tensor.cpu())
                wm_save_path = os.path.join(wm_out_dir, crop_filename)
                wm_img_pil.save(wm_save_path)
                
    print(f"\n🎉 Generazione completata! Dataset salvato nella cartella: {OUT_ROOT}")

if __name__ == "__main__":
    main()