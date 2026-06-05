"""
test_attack.py
==============
Inferenza e valutazione di UNetDenoiseAttack (v2) sul test set.

Miglioramenti rispetto all'originale:
  1. Tile-based inference : elabora immagini ad alta risoluzione a tile
                            sovrapposti (overlap-tile strategy) per evitare
                            artefatti ai bordi e problemi di memoria GPU.
  2. Metriche estese      : aggiunge PSNR e SSIM (oltre a bit accuracy e logit)
                            per una valutazione quantitativa completa.
  3. Salvataggio side-by-side : griglia visiva PNG per ogni immagine
                                (clean | watermarked | attacked).
  4. Report CSV arricchito : una riga per stato (Pulita / WM / Attaccata)
                             con tutte le metriche calcolate.
  5. Normalizzazione robusta: pre/post-processing coerente con il training.
"""

import os
import math

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from PIL import Image
import pandas as pd
import videoseal

from unet_attack_model import UNetDenoiseAttack


# ============================================================================
# 1. CONFIGURAZIONE
# ============================================================================

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIR = "dataset_minSize/test"
OUTPUT_DIR = "results_unet"
CSV_FILE   = "risultati_unet_v2.csv"
MSG_SIZE   = 256

# Parametri tile-based inference
TILE_SIZE    = 512    # dimensione di ogni tile (pixel)
TILE_OVERLAP = 64     # sovrapposizione tra tile adiacenti (per fondere i bordi)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "grids"), exist_ok=True)


# ============================================================================
# 2. UTILITÀ: METRICHE
# ============================================================================

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse)


def ssim_simple(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """SSIM scalare (singola immagine [1, C, H, W])."""
    C = pred.shape[1]
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
    pad = window_size // 2

    def _conv(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel, padding=pad, groups=C)

    mu_p  = _conv(pred);    mu_t  = _conv(target)
    mu_p2 = mu_p * mu_p;   mu_t2 = mu_t * mu_t; mu_pt = mu_p * mu_t
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    sig_p2 = _conv(pred * pred)   - mu_p2
    sig_t2 = _conv(target * target) - mu_t2
    sig_pt = _conv(pred * target) - mu_pt
    num   = (2 * mu_pt + C1) * (2 * sig_pt + C2)
    denom = (mu_p2 + mu_t2 + C1) * (sig_p2 + sig_t2 + C2)
    return (num / denom).mean().item()


# ============================================================================
# 3. TILE-BASED INFERENCE
# ============================================================================

def run_tile_inference(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    tile_size: int = TILE_SIZE,
    overlap: int = TILE_OVERLAP,
) -> torch.Tensor:
    """
    Suddivide l'immagine in tile sovrapposti, esegue il modello su ognuno,
    poi ricompone con blending lineare nelle zone di sovrapposizione.

    img_tensor : [1, C, H, W] in [0, 1].
    Returns    : [1, C, H, W] in [0, 1].
    """
    _, C, H, W = img_tensor.shape
    stride     = tile_size - overlap

    # Canvas di output e mappa di peso per il blending
    out     = torch.zeros_like(img_tensor)
    weights = torch.zeros(1, 1, H, W, device=img_tensor.device)

    # Finestra di blending: ramp lineare ai bordi del tile
    blend_1d   = torch.ones(tile_size, device=img_tensor.device)
    for i in range(overlap):
        blend_1d[i]              = (i + 1) / (overlap + 1)
        blend_1d[tile_size - 1 - i] = (i + 1) / (overlap + 1)
    blend_2d = (blend_1d[:, None] * blend_1d[None, :]).unsqueeze(0).unsqueeze(0)  # [1,1,T,T]

    ys = list(range(0, H - tile_size + 1, stride))
    xs = list(range(0, W - tile_size + 1, stride))
    # Assicuriamo di coprire il bordo destro/inferiore
    if ys[-1] + tile_size < H:
        ys.append(H - tile_size)
    if xs[-1] + tile_size < W:
        xs.append(W - tile_size)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = img_tensor[:, :, y:y + tile_size, x:x + tile_size]
                result = model(patch)               # [1, C, T, T]
                out[:, :, y:y + tile_size, x:x + tile_size]     += result * blend_2d
                weights[:, :, y:y + tile_size, x:x + tile_size] += blend_2d

    out = out / weights.clamp(min=1e-6)
    return torch.clamp(out, 0.0, 1.0)


# ============================================================================
# 4. CARICAMENTO MODELLO
# ============================================================================

print("Caricamento UNet (v2)...")
model = UNetDenoiseAttack(in_channels=3, out_channels=3, base_ch=32).to(device)

state_dict = torch.load(
    "checkpoints/unet_best.pth", map_location=device, weights_only=True
)
# Pulizia prefisso DataParallel
from collections import OrderedDict
clean_sd = OrderedDict()
for k, v in state_dict.items():
    clean_sd[k[7:] if k.startswith("module.") else k] = v
model.load_state_dict(clean_sd)
model.eval()

print("Caricamento PixelSeal...")
pixelseal = videoseal.load("pixelseal").eval()


# ============================================================================
# 5. PREPARAZIONE COPPIE (clean / watermarked)
# ============================================================================

clean_dir = os.path.join(INPUT_DIR, "clean_img")
wm_dir    = os.path.join(INPUT_DIR, "wm_img")

clean_files = sorted(f for f in os.listdir(clean_dir) if not f.startswith("."))
wm_files    = sorted(f for f in os.listdir(wm_dir)    if not f.startswith("."))

valid_pairs: list[tuple[str, str]] = []
for c_file in clean_files:
    matching = [w for w in wm_files if c_file in w]
    if matching:
        valid_pairs.append((c_file, matching[0]))

print(f"\nCoppie valide: {len(valid_pairs)}")


# ============================================================================
# 6. INFERENZA
# ============================================================================

def load_tensor(path: str, device: torch.device) -> torch.Tensor:
    """Carica un'immagine PNG/JPEG come tensore [1, 3, H, W] in [0,1]."""
    img = Image.open(path).convert("RGB")
    return TF.to_tensor(img).unsqueeze(0).to(device)


def extract_detection(
    pix: torch.nn.Module, tensor: torch.Tensor, msg_size: int
) -> dict:
    """Estrae logit e bit dal detector PixelSeal (CPU)."""
    out     = pix.detect(tensor.cpu())
    logit_c0 = out["preds"][:, 0].item()
    bits     = (out["preds"][:, 1:] > 0).float()
    return {"logit": logit_c0, "bits": bits}


results: list[dict] = []

for idx, (clean_name, wm_name) in enumerate(valid_pairs):
    print(f"[{idx + 1:4d}/{len(valid_pairs)}] {clean_name}")

    t_clean = load_tensor(os.path.join(clean_dir, clean_name), device)
    t_wm    = load_tensor(os.path.join(wm_dir,    wm_name),    device)
    _, _, H, W = t_wm.shape

    # ── A. Analisi immagine pulita ──────────────────────────────────────────
    d_clean = extract_detection(pixelseal, t_clean, MSG_SIZE)

    # ── B. Analisi immagine watermarked ────────────────────────────────────
    d_wm = extract_detection(pixelseal, t_wm, MSG_SIZE)

    # ── C. Attacco (tile-based) ─────────────────────────────────────────────
    t_attacked = run_tile_inference(model, t_wm)

    # ── D. Analisi immagine attaccata ───────────────────────────────────────
    d_att = extract_detection(pixelseal, t_attacked, MSG_SIZE)

    # ── E. Metriche ─────────────────────────────────────────────────────────
    gt_bits = d_wm["bits"]   # ground truth = bit estratti dall'immagine WM

    ba_clean = (d_clean["bits"] == gt_bits).sum().item() / MSG_SIZE
    ba_wm    = 1.0                                          # per costruzione
    ba_att   = (d_att["bits"] == gt_bits).sum().item() / MSG_SIZE

    # PSNR e SSIM tra attaccata e pulita (stesso device)
    t_clean_cpu  = t_clean.cpu()
    t_wm_cpu     = t_wm.cpu()
    t_att_cpu    = t_attacked.cpu()

    psnr_att  = psnr(t_att_cpu,  t_clean_cpu)
    ssim_att  = ssim_simple(t_att_cpu,  t_clean_cpu)
    psnr_wm   = psnr(t_wm_cpu,   t_clean_cpu)
    ssim_wm   = ssim_simple(t_wm_cpu,   t_clean_cpu)

    # ── F. Salvataggio immagine ─────────────────────────────────────────────
    # Crop al formato originale (nessun padding aggiunto in v2)
    Image.fromarray(
        (t_att_cpu[0].permute(1, 2, 0).numpy() * 255).astype("uint8")
    ).save(os.path.join(OUTPUT_DIR, f"cleaned_{clean_name}"))

    # Griglia side-by-side: [clean | wm | attacked] — resize a 256 per la grid
    def _thumb(t: torch.Tensor, size: int = 256) -> torch.Tensor:
        return TF.resize(t.squeeze(0), [size, size]).unsqueeze(0)

    grid = vutils.make_grid(
        torch.cat([_thumb(t_clean_cpu), _thumb(t_wm_cpu), _thumb(t_att_cpu)], dim=0),
        nrow=3, normalize=False, padding=4, pad_value=1.0,
    )
    TF.to_pil_image(grid).save(
        os.path.join(OUTPUT_DIR, "grids", f"grid_{clean_name}")
    )

    # ── G. Raccolta risultati ───────────────────────────────────────────────
    for stato, ba, logit, psnr_val, ssim_val in [
        ("Pulita",     ba_clean, d_clean["logit"], None,      None),
        ("Watermarked", ba_wm,   d_wm["logit"],    psnr_wm,   ssim_wm),
        ("Attaccata",  ba_att,   d_att["logit"],   psnr_att,  ssim_att),
    ]:
        results.append({
            "nomeImg":            clean_name,
            "modello":            "pixelseal",
            "stato":              stato,
            "bit_accuracy":       round(ba,       4),
            "wm_logit_c0":        round(logit,    4),
            "PSNR_vs_clean_dB":   round(psnr_val, 3) if psnr_val is not None else "",
            "SSIM_vs_clean":      round(ssim_val, 4) if ssim_val is not None else "",
        })

# ============================================================================
# 7. SALVATAGGIO CSV + SUMMARY
# ============================================================================

df = pd.DataFrame(results)
df.to_csv(CSV_FILE, index=False, sep=";")

# Stampa sommario
df_att = df[df["stato"] == "Attaccata"]
df_wm  = df[df["stato"] == "Watermarked"]

print(f"\n{'='*55}")
print(f"  Immagini processate : {len(valid_pairs)}")
print(f"  Bit-Acc WM (pre-att): {df_wm['bit_accuracy'].mean():.4f}")
print(f"  Bit-Acc post-attacco: {df_att['bit_accuracy'].mean():.4f}")
print(f"  PSNR medio (att vs clean): {df_att['PSNR_vs_clean_dB'].apply(pd.to_numeric, errors='coerce').mean():.2f} dB")
print(f"  SSIM medio (att vs clean): {df_att['SSIM_vs_clean'].apply(pd.to_numeric, errors='coerce').mean():.4f}")
print(f"  Logit C0 medio (pre): {df_wm['wm_logit_c0'].mean():.4f}")
print(f"  Logit C0 medio (post): {df_att['wm_logit_c0'].mean():.4f}")
print(f"{'='*55}")
print(f"\nCSV salvato in: {CSV_FILE}")
print(f"Immagini salvate in: {OUTPUT_DIR}/")