import torch
from torchinfo import summary
from unet_attack_model import ConvAutoencoderDenoise

# 1. Inizializza il tuo modello
model = ConvAutoencoderDenoise(in_channels=3, out_channels=3)

# 2. Definisci la grandezza di un'immagine in input finta (Batch, Canali, Altezza, Larghezza)
dimensione_input = (1, 3, 256, 256)

print("\n" + "="*60)
print(" SUMMARY DELL'AUTOENCODER ")
print("="*60)

# 3. Genera e stampa la tabella esatta
# Passiamo col_names per forzare le colonne identiche a quelle della tua foto
statistiche = summary(
    model, 
    input_size=dimensione_input,
    col_names=["output_size", "num_params"],
    col_width=20,
    row_settings=["var_names"]
)

print(statistiche)
