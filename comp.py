"""
wsl_benchmark.py
Native Linux Benchmarking: Biological Gatekeeper v2 vs DNABERT-2
Runs cleanly in WSL2 using full ALiBi Attention and Triton.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from Bio import Entrez, SeqIO
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, mean_absolute_error
)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. GLOBAL CONFIGURATION
# =============================================================================
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME   = "zhihan1996/DNABERT-2-117M"
NCBI_EMAIL   = "example@example.com"
NCBI_ID      = "NT_011109.10" 
SEQ_START    = 1        
SEQ_STOP     = 1000000  # 1 Million BP 
MODEL_FILE   = "model.pkl"
TEST_SIZE    = 200      # Increase to 1000+ for the final paper
# =============================================================================

print(f"--- Starting WSL2 Native Benchmark on {DEVICE.upper()} ---")

# --- 2. LOAD BIOLOGICAL GATEKEEPER V2 ---
print("\n1. Loading Biological Gatekeeper v2...")
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Could not find {MODEL_FILE}. Run pwm 3.py first.")

with open(MODEL_FILE, 'rb') as f:
    v2_data = pickle.load(f)

pwm_logo, codon_vocab = v2_data['pwm_logo'], v2_data['codon_vocab']
nuc2idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def score_promoter(window):
    if len(window) != 50: return -999.0
    return sum(pwm_logo[nuc2idx.get(c, 0), i] for i, c in enumerate(window))

def encode(seq_str):
    return [codon_vocab.get(seq_str[i:i+3], codon_vocab["UNK"]) for i in range(0, len(seq_str)-2, 3)]

def decoder_v2(seq_str, tokens, pwm_threshold=-3.0, min_cds_len_bp=135):
    candidates = []
    for t, tok in enumerate(tokens):
        if tok == codon_vocab["ATG"]:
            nuc_idx = t * 3
            upstream = seq_str[max(0, nuc_idx-50):nuc_idx]
            if len(upstream) == 50:
                p_score = score_promoter(upstream)
                if p_score >= pwm_threshold: candidates.append((p_score, t))
    if not candidates: return [0] * len(tokens)
    _, best_start = max(candidates)
    if (len(tokens) - best_start - 1) * 3 < min_cds_len_bp: return [0] * len(tokens)
    path = [0] * len(tokens)
    path[best_start] = 1
    return path

# --- 3. LOAD FULL DNABERT-2 (NATIVE LINUX) ---
print(f"\n2. Loading DNABERT-2 (With ALiBi & Triton enabled)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
config.pad_token_id = 0

# Since we are in Linux, this will load the custom architecture flawlessly
dnabert = AutoModel.from_pretrained(
    MODEL_NAME, config=config, trust_remote_code=True, low_cpu_mem_usage=False
)
if DEVICE == "cuda":
    dnabert = dnabert.to(DEVICE).half()
dnabert.eval()

def get_dnabert_embedding(seq):
    inputs = tokenizer(seq, return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = dnabert(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]

# --- 4. FETCH NCBI DATA ---
Entrez.email = NCBI_EMAIL
print(f"\n3. Fetching {NCBI_ID} from {SEQ_START} to {SEQ_STOP}...")
with Entrez.efetch(db="nucleotide", id=NCBI_ID, seq_start=SEQ_START, seq_stop=SEQ_STOP, rettype="gbwithparts", retmode="text") as h:
    rec_test = SeqIO.read(h, "genbank")

test_cds_wins, test_int_wins, intergenics = [], [], []
last_end = 0

for f in sorted([f for f in rec_test.features if f.type == "CDS"], key=lambda x: x.location.start):
    if f.location.start > last_end + 300:
        intergenics.append((int(last_end), int(f.location.start)))
    last_end = max(last_end, f.location.end)

for f in rec_test.features:
    if f.type == "CDS" and f.location.strand == 1:
        s = int(f.location.start)
        ws, we = s - 150, s + 150
        if ws >= 0 and we <= len(rec_test.seq):
            seq_str = str(rec_test.seq[ws:we]).upper()
            if all(c in "ACGT" for c in seq_str):
                test_cds_wins.append((seq_str, [0]*50 + [1] + [2]*48 + [3]))

for start, end in intergenics:
    for i in range(start, end-300, 300):
        seq_str = str(rec_test.seq[i:i+300]).upper()
        if all(c in "ACGT" for c in seq_str):
            test_int_wins.append((seq_str, [0]*100))

np.random.seed(42)
np.random.shuffle(test_cds_wins)
np.random.shuffle(test_int_wins)

train_data = test_cds_wins[:TEST_SIZE] + test_int_wins[:TEST_SIZE]
test_data = test_cds_wins[TEST_SIZE:TEST_SIZE*2] + test_int_wins[TEST_SIZE:TEST_SIZE*2]

# --- 5. TRAIN DNABERT CLASSIFIER HEAD ---
print("\n4. Extracting Embeddings for Classification Head...")
X_train, y_train = [], []
for seq, lbls in tqdm(train_data, desc="DNABERT Train Extract"):
    X_train.append(get_dnabert_embedding(seq))
    y_train.append(1 if 1 in lbls else 0)

clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train, y_train)

# --- 6. EVALUATION PARADIGM ---
print("\n5. Running Head-to-Head Inference...")
v2_t, v2_p, db_p = [], [], []
true_starts, v2_starts, db_starts = [], [], []
v2_exact, db_exact = 0, 0
total_cds = 0

# Less aggressive threshold for the benchmark
PWM_THRESH = -10.0
MIN_LEN = 30

for seq, lbls in tqdm(test_data, desc="Benchmarking (Native Linux)"):
    bin_t = 1 if 1 in lbls else 0
    v2_t.append(bin_t)
    
    toks = encode(seq)
    preds_v2 = decoder_v2(seq, toks, pwm_threshold=PWM_THRESH, min_cds_len_bp=MIN_LEN)
    bin_v2 = 1 if 1 in preds_v2 else 0
    v2_p.append(bin_v2)

    emb = get_dnabert_embedding(seq)
    prob_db = clf.predict_proba([emb])[0][1]
    bin_db = 1 if prob_db > 0.5 else 0
    db_p.append(bin_db)

    if bin_t == 1:
        total_cds += 1
        t_s = lbls.index(1) * 3
        true_starts.append(t_s)
        
        if bin_v2 == 1:
            p_s = preds_v2.index(1) * 3
            v2_starts.append(p_s)
            if p_s == t_s: v2_exact += 1
        else:
            v2_starts.append(len(seq))
            
        if bin_db == 1:
            best_atg, min_dist = -1, float('inf')
            for i in range(0, len(seq)-2, 3):
                if seq[i:i+3] == "ATG":
                    if abs(i - 150) < min_dist:
                        min_dist, best_atg = abs(i - 150), i
            db_starts.append(best_atg if best_atg != -1 else len(seq))
            if best_atg == t_s: db_exact += 1
        else:
            db_starts.append(len(seq))

# --- SANITY CHECK ---
print(f"\nDebug: True Labels 1s: {sum(v2_t)}")
print(f"Debug: HMM Predicted 1s: {sum(v2_p)}")
print(f"Debug: DNABERT Predicted 1s: {sum(db_p)}")

if sum(db_p) == 0: print("[!] WARNING: DNABERT is predicting 0 for everything.")
if sum(v2_p) == 0: print("[!] WARNING: HMM is predicting 0 for everything. Lower thresholds!")

def calc(preds, starts, exact):
    return {
        'mcc': matthews_corrcoef(v2_t, preds),
        'precision': precision_score(v2_t, preds, zero_division=0),
        'recall': recall_score(v2_t, preds, zero_division=0),
        'f1': f1_score(v2_t, preds, zero_division=0),
        'mae': mean_absolute_error(true_starts, starts) if true_starts else 0,
        'exact_rate': (exact / total_cds * 100) if total_cds else 0
    }

v2_res = calc(v2_p, v2_starts, v2_exact)
db_res = calc(db_p, db_starts, db_exact)

# --- 7. VISUALIZATION DASHBOARD ---
print("\n6. Generating Comparison Dashboard...")
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(17, 10))
fig.suptitle("DNABERT-2 vs Biological Gatekeeper v2 — TIS Benchmark", fontsize=18, fontweight='bold', y=0.99)
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

metric_names = ["MCC", "Precision", "Recall", "F1-Score"]
v2_vals = [v2_res["mcc"], v2_res["precision"], v2_res["recall"], v2_res["f1"]]
db_vals = [db_res["mcc"], db_res["precision"], db_res["recall"], db_res["f1"]]

x = np.arange(len(metric_names))
width = 0.35

ax1 = fig.add_subplot(gs[0, 0])
bars_v2 = ax1.bar(x - width/2, v2_vals, width, label="v2 (PWM + Length)", color="#55A868", alpha=0.85)
bars_db = ax1.bar(x + width/2, db_vals, width, label="DNABERT-2", color="#9B59B6", alpha=0.85)
ax1.set_xticks(x)
ax1.set_xticklabels(metric_names)
ax1.set_ylim(0, 1.0)
ax1.set_title("Classification Metrics", fontsize=13)
ax1.legend()
for b in list(bars_v2) + list(bars_db):
    ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.015, f"{b.get_height():.3f}", ha='center', fontsize=9)

ax2 = fig.add_subplot(gs[0, 1])
models = ["v2 (PWM+Len)", "DNABERT-2"]
maes = [v2_res["mae"], db_res["mae"]]
ax2.bar(models, maes, color=['#55A868', '#9B59B6'], width=0.5, alpha=0.8)
ax2.set_title("Mean Absolute Error (Lower is Better)", fontsize=13)
ax2.set_ylabel("Distance from True Start (bp)")
for i, m in enumerate(maes):
    ax2.text(i, m + (max(maes)*0.02), f"{m:.1f} bp", ha='center', fontsize=11)

ax3 = fig.add_subplot(gs[0, 2])
exacts = [v2_res["exact_rate"], db_res["exact_rate"]]
ax3.bar(models, exacts, color=['#55A868', '#9B59B6'], width=0.5, alpha=0.8)
ax3.set_title("Exact Start Codon Match (%)", fontsize=13)
ax3.set_ylim(0, 100)
for i, e in enumerate(exacts):
    ax3.text(i, e + 2, f"{e:.1f}%", ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig("wsl2_dnabert_vs_v2_dashboard.png", dpi=300, bbox_inches='tight')
print("\nBenchmark Complete! Dashboard saved as 'wsl2_dnabert_vs_v2_dashboard.png'.")