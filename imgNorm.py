import torch
from torch.utils.data import DataLoader
# Importiamo la tua classe reale dal tuo file di caricamento
from loader_dataset import WatermarkDenoisingDataset

def check_ranges():
    print("="*50)
    print("🔍 VERIFICA RAPIDA RANGE PIXEL (SENZA TRAINING)")
    print("="*50)
    
    try:
        # Inizializziamo il dataset puntando alla cartella generata
        dataset = WatermarkDenoisingDataset(root_dir="dataset_minSize/train", crop_size=256)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Estraiamo un solo batch reale
        wm_imgs, clean_imgs = next(iter(loader))
        
        print(f"Tensore Watermark (wm_imgs):")
        print(f"  -> Shape: {wm_imgs.shape}")
        print(f"  -> Tipo:  {wm_imgs.dtype}")
        print(f"  -> Min:   {wm_imgs.min().item():.5f}")
        print(f"  -> Max:   {wm_imgs.max().item():.5f}")
        
        print(f"\nTensore Pulito (clean_imgs):")
        print(f"  -> Shape: {clean_imgs.shape}")
        print(f"  -> Tipo:  {clean_imgs.dtype}")
        print(f"  -> Min:   {clean_imgs.min().item():.5f}")
        print(f"  -> Max:   {clean_imgs.max().item():.5f}")
        
        # Sbarramento logico di controllo
        if wm_imgs.max().item() > 1.0 or clean_imgs.max().item() > 1.0:
            print("\n⚠️ ATTENZIONE: I pixel sono nel range 0-255 o fuori scala!")
        elif wm_imgs.min().item() < 0.0 or clean_imgs.min().item() < 0.0:
            print("\n⚠️ ATTENZIONE: Rilevati valori negativi imprevisti!")
        else:
            print("\n✅ TUTTO CORRETTO: Le immagini entrano nel modello già normalizzate in [0, 1].")
            
    except Exception as e:
        print(f"Errore durante la verifica: {e}")
        
    print("="*50)

if __name__ == "__main__":
    check_ranges()