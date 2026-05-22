import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================
CSV_FILE = "risultati_convae_mse_raptor.csv" 
COLONNA_SCORE = 'bit accuracy' 

# Impostiamo lo stile bianco con griglia, come nella tua immagine
sns.set_theme(style="whitegrid")

# ==========================================
# 2. LETTURA DATI
# ==========================================
print("📊 Lettura del file CSV in corso...")
df = pd.read_csv(CSV_FILE, sep=';')

# Dividiamo i dati nei tre stati
df_pulite = df[df['stato'] == 'Pulita'] 
df_wm = df[df['stato'] == 'Watermarked']
df_att = df[df['stato'] == 'Attaccata'] 

# ==========================================
# 3. GRAFICO 1: ISTOGRAMMA (STILE IMMAGINE ALLEGATA)
# ==========================================
print("📈 Generazione Istogramma...")
plt.figure(figsize=(10, 6), dpi=300)

sns.histplot(data=df_pulite, x=COLONNA_SCORE, color='limegreen', label='Pulita (Nessun WM)', 
             bins=60, binrange=(0.4, 1.0), edgecolor='black', alpha=0.9)

sns.histplot(data=df_att, x=COLONNA_SCORE, color='crimson', label='Attaccata (Danneggiata)', 
             bins=60, binrange=(0.4, 1.0), edgecolor='black', alpha=0.9)

sns.histplot(data=df_wm, x=COLONNA_SCORE, color='dodgerblue', label='Watermarked (Intatta)', 
             bins=60, binrange=(0.4, 1.0), edgecolor='black', alpha=0.9)

plt.xlabel("Bit Accuracy (0.0 = 0%, 1.0 = 100%)", fontsize=11)
plt.ylabel("Numero di Immagini", fontsize=11)
plt.title("Istogramma Bit Accuracy: Immagini Pulite, Watermarked e Attaccate - Attacco ConvAE (MSE)", fontsize=13, pad=15)

legend = plt.legend(title="Legenda Stati", loc="upper left", frameon=True, shadow=True, facecolor='white')
legend.get_frame().set_edgecolor('gray')

plt.tight_layout()
NOME_HIST = 'Istogramma_BitAccuracy_ConvAE_MSE.png'
plt.savefig(NOME_HIST)
plt.close()

# ==========================================
# 4. GRAFICO 2: CURVA ROC
# ==========================================
print("📈 Generazione Curva ROC...")
plt.figure(figsize=(9, 7), dpi=300)

# --- A. Baseline (Pulite vs Intatte) ---
y_true_base = [0] * len(df_pulite) + [1] * len(df_wm)
y_score_base = list(df_pulite[COLONNA_SCORE]) + list(df_wm[COLONNA_SCORE])

fpr_base, tpr_base, _ = roc_curve(y_true_base, y_score_base)
roc_auc_base = auc(fpr_base, tpr_base)

plt.plot(fpr_base, tpr_base, color='dodgerblue', lw=3, 
         label=f'Baseline (Pulite vs Intatte) - AUC: {roc_auc_base:.4f}')

# --- B. Attacco (Pulite vs Attaccate ConvAE) ---
y_true_att = [0] * len(df_pulite) + [1] * len(df_att)
y_score_att = list(df_pulite[COLONNA_SCORE]) + list(df_att[COLONNA_SCORE])

fpr_att, tpr_att, _ = roc_curve(y_true_att, y_score_att)
roc_auc_att = auc(fpr_att, tpr_att)

plt.plot(fpr_att, tpr_att, color='crimson', lw=3, linestyle='--',
         label=f'Attacco ConvAE (Loss MSE) - AUC: {roc_auc_att:.4f}')


# Impostazioni Assi
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR) - Falsi Allarmi', fontsize=12)
plt.ylabel('True Positive Rate (TPR) - Rilevamenti Corretti', fontsize=12)
plt.title('Curva ROC: Robustezza di PixelSeal contro Attacco ConvAE (MSE)', fontsize=14, pad=15)
plt.legend(loc="lower right", fontsize=11, frameon=True, shadow=True)

plt.tight_layout()
NOME_ROC = 'Curva_ROC_ConvAE_MSE.png'
plt.savefig(NOME_ROC)
plt.close()

print(f"✅ Finito! Grafici salvati come:\n1. {NOME_HIST}\n2. {NOME_ROC}")