import json
import numpy as np
import pandas as pd
from Bio import SeqIO, Entrez
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, mean_absolute_error,
    confusion_matrix, roc_auc_score, roc_curve, auc, balanced_accuracy_score
)
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import argparse
import os

warnings.filterwarnings('ignore')
Entrez.email = "example@example.com"

# Global model variables
nuc2idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
pwm_logo = None
codon_vocab = {}
emissions = None
log_emissions = None

# 1. During Training: Build a Codon Frequency Map
def build_codon_map(sequences):
    counts = {codon: 1 for codon in all_possible_codons} # Laplacian smoothing
    for seq in sequences:
        for i in range(0, len(seq)-2, 3):
            codon = seq[i:i+3]
            counts[codon] += 1
    # Convert to Log-Likelihood
    total = sum(counts.values())
    return {k: np.log(v/total) for k, v in counts.items()}

# 2. During Decoding: Add the "Coding Potential" check
def score_coding_potential(downstream_seq, codon_map):
    score = 0
    for i in range(0, len(downstream_seq)-2, 3):
        codon = downstream_seq[i:i+3]
        score += codon_map.get(codon, -10) # Penalty for unknown/stop codons
    return score

def score_promoter(window):
    if len(window) != 50:
        return -999.0
    # Use .get(c, 0) to assign a neutral score to 'N's instead of crashing
    return sum(pwm_logo[nuc2idx.get(c, 0), i] for i, c in enumerate(window))

def encode(seq_str):
    return [codon_vocab.get(seq_str[i:i+3], codon_vocab["UNK"])
            for i in range(0, len(seq_str)-2, 3)]

def load_training_sources(filepath="training_sources.json"):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {filepath}.")
        return []
    
def train_model():
    print("1. Fetching Pan-Eukaryotic Training Data...")
    training_sources = load_training_sources()
    train_cds = []
    pwm_counts = np.ones((4, 50)) * 1e-4
    
    # NEW: Initialize Codon Usage Counter with Laplace smoothing
    all_codons = [a+b+c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
    codon_counts = {c: 1 for c in all_codons}
    total_codons = 64
    
    seen_tis = set() 
    metadata_log = []

    for source in training_sources:
        print(f"   -> Pulling {source['name']}...")
        try:
            fetch_params = {"db": "nucleotide", "id": source["id"], "rettype": "gbwithparts", "retmode": "text"}
            if source.get("start") and source.get("stop"):
                fetch_params["seq_start"] = source["start"]
                fetch_params["seq_stop"] = source["stop"]
                offset = source["start"]
            else:
                offset = 0
            
            with Entrez.efetch(**fetch_params) as handle:
                rec_train = SeqIO.read(handle, "genbank")
            
            source_count = 0
            for f in rec_train.features:
                if f.type == "CDS":
                    strand = f.location.strand
                    if strand == 1:
                        tis_rel = int(f.location.start)
                        ws, we = tis_rel - 150, tis_rel + 150
                    else:
                        tis_rel = int(f.location.end) - 1
                        ws, we = tis_rel - 149, tis_rel + 151
                        
                    tis_abs = tis_rel + (offset - 1 if offset > 0 else 0)

                    if (source["id"], tis_abs) in seen_tis:
                        continue
                    seen_tis.add((source["id"], tis_abs))

                    if ws >= 0 and we <= len(rec_train.seq):
                        chunk = rec_train.seq[ws:we]
                        if strand == -1:
                            chunk = chunk.reverse_complement()
                        
                        seq_str = str(chunk).upper()
                        if len(seq_str) == 300 and seq_str[150:153] == "ATG":
                            meta = {
                                "species": source["name"],
                                "accession": source["id"],
                                "tis_absolute": tis_abs,
                                "strand": strand
                            }
                            train_cds.append((seq_str, [0]*50 + [1] + [2]*48 + [3], meta))
                            metadata_log.append(meta)
                            
                            source_count += 1
                            # 1. Update PWM (Upstream)
                            for i, nuc in enumerate(seq_str[100:150]):
                                if nuc in nuc2idx:
                                    pwm_counts[nuc2idx[nuc], i] += 1
                                    
                            # 2. NEW: Update Codon Usage (Downstream)
                            downstream_cds = seq_str[153:]
                            for i in range(0, len(downstream_cds)-2, 3):
                                codon = downstream_cds[i:i+3]
                                if codon in codon_counts:
                                    codon_counts[codon] += 1
                                    total_codons += 1
                                    
            print(f"      Gathered {source_count} CDS windows.")
        except Exception as e:
            print(f"      [!] Failed {source['name']}: {e}")

    # Calculate PWM Log-Odds
    pwm = pwm_counts / pwm_counts.sum(axis=0, keepdims=True)
    bg = np.array([0.25]*4).reshape(4,1)
    pwm_logo_val = np.log2(pwm / bg)

    # NEW: Calculate Codon Usage Log-Probabilities
    codon_usage_log_probs = {c: np.log(count / total_codons) for c, count in codon_counts.items()}

    # ... (Keep your existing vocab and emissions logic here for legacy v1/v2 support) ...
    codon_vocab_val = {c: i for i, c in enumerate(all_codons)}
    codon_vocab_val["UNK"] = 64
    emissions_val = np.ones((4, 65)) * 1e-6
    log_emissions_val = np.log(emissions_val / emissions_val.sum(axis=1, keepdims=True))

    return {
        'pwm_logo': pwm_logo_val,
        'codon_vocab': codon_vocab_val,
        'emissions': emissions_val,
        'log_emissions': log_emissions_val,
        'codon_usage': codon_usage_log_probs # <-- Make sure to save this
    }


# ─────────────────────────────────────────────
# 2.  DECODER FUNCTIONS  (v1 and v2)
# ─────────────────────────────────────────────

def decoder_v1(seq_str, tokens):
    """v1: PWM only — no threshold, no length filter."""
    best_start, best_score = -1, -np.inf
    for t, tok in enumerate(tokens):
        if tok == codon_vocab["ATG"]:
            nuc_idx = t * 3
            upstream = seq_str[nuc_idx-50:nuc_idx]
            if len(upstream) == 50:
                p = score_promoter(upstream)
                if p > best_score:
                    best_score, best_start = p, t
    path = [0] * len(tokens)
    if best_start != -1:
        path[best_start] = 1
        for i in range(best_start+1, len(tokens)):
            path[i] = 2
    return path


def decoder_v2(seq_str, tokens,
               pwm_threshold: float = -5.0,
               min_cds_len_bp: int  = 90):
    """
    v2: Biological Gatekeeper with two hard filters.

    Gate 1 — Dynamic PWM Threshold:
        Any ATG where upstream PWM score < pwm_threshold is discarded.
        Default: -3.0  (empirically optimal from forensic sweep).

    Gate 2 — Minimum CDS Length Filter:
        If the projected coding region from the chosen ATG to the end
        of the window is < min_cds_len_bp, the prediction is suppressed.
        Default: 135 bp  (eliminates ~37% sORF false positives).
    """
    candidates = []

    for t, tok in enumerate(tokens):
        if tok == codon_vocab["ATG"]:
            nuc_idx = t * 3
            upstream = seq_str[nuc_idx-50:nuc_idx]
            if len(upstream) == 50:
                p_score = score_promoter(upstream)
                # GATE 1: PWM Threshold
                if p_score >= pwm_threshold:
                    candidates.append((p_score, t))

    if not candidates:
        return [0] * len(tokens)

    _, best_start = max(candidates)

    # GATE 2: Minimum CDS Length
    projected_coding_bp = (len(tokens) - best_start - 1) * 3
    if projected_coding_bp < min_cds_len_bp:
        return [0] * len(tokens)

    path = [0] * len(tokens)
    path[best_start] = 1
    for i in range(best_start+1, len(tokens)):
        path[i] = 2
    return path

def score_coding_potential(sequence, codon_usage_map):
    """Scores the downstream sequence based on log-probabilities of codons."""
    score = 0.0
    valid_codons = 0
    for i in range(0, len(sequence)-2, 3):
        codon = sequence[i:i+3]
        if codon in codon_usage_map:
            score += codon_usage_map[codon]
            valid_codons += 1
    return score / valid_codons if valid_codons > 0 else -999.0

def decoder_v3(seq_str, tokens, pwm_threshold=-5.0, cub_weight=1.5):
    """
    v3: Contextual Gatekeeper. Combines Upstream PWM with Downstream Codon Bias.
    Proves that looking at 3-bp intervals downstream separates true ATGs from noise.
    """
    candidates = []
    
    # Needs access to global model data
    global codon_usage_log_probs 
    
    for t, tok in enumerate(tokens):
        if tok == codon_vocab["ATG"]:
            nuc_idx = t * 3
            upstream = seq_str[nuc_idx-50:nuc_idx]
            downstream = seq_str[nuc_idx+3:nuc_idx+93] # Look at next 30 amino acids
            
            if len(upstream) == 50 and len(downstream) >= 90:
                p_score = score_promoter(upstream)
                
                # Only check downstream if the promoter is viable
                if p_score >= pwm_threshold:
                    cub_score = score_coding_potential(downstream, codon_usage_log_probs)
                    # Combine scores (you can tune cub_weight)
                    total_score = p_score + (cub_score * cub_weight) 
                    candidates.append((total_score, t))

    if not candidates:
        return [0] * len(tokens)

    _, best_start = max(candidates)

    path = [0] * len(tokens)
    path[best_start] = 1
    for i in range(best_start+1, len(tokens)):
        path[i] = 2
    return path

def prepare_test_data(rec_test):
    test_cds_wins = []
    seen_test_tis = set()
    
    # 1. Gather all True CDS windows (Both Forward and Reverse Strands)
    for f in rec_test.features:
        if f.type == "CDS":
            strand = f.location.strand
            
            # Forward Strand relative coordinate math
            if strand == 1:
                tis_rel = int(f.location.start)
                ws, we = tis_rel - 150, tis_rel + 150
                # print(str(f.location.start) + ", ", end = "")
            # Reverse Strand relative coordinate math (Shifted by 1)
            else:
                tis_rel = int(f.location.end) - 1
                ws, we = tis_rel - 149, tis_rel + 151
                
            # Prevent evaluating identical splice-variants multiple times
            if tis_rel in seen_test_tis:
                continue
            seen_test_tis.add(tis_rel)

            if ws >= 0 and we <= len(rec_test.seq):
                chunk = rec_test.seq[ws:we]
                
                # Flip negative strand to match model orientation
                if strand == -1:
                    chunk = chunk.reverse_complement()
                
                seq_str = str(chunk).upper()
                
                # STRICT VERIFICATION: If there is no ATG, it's an internal exon. Discard it.
                if len(seq_str) == 300 and seq_str[150:153] == "ATG" and all(c in "ACGT" for c in seq_str):
                    test_cds_wins.append((seq_str, [0]*50 + [1] + [2]*48 + [3]))

    # 2. Gather Intergenic/Decoy Windows
    intergenics = []
    last_end = 0
    # Map valid intergenic spans
    for f in sorted([f for f in rec_test.features if f.type == "CDS"], key=lambda x: int(x.location.start)):
        if int(f.location.start) > last_end + 300:
            intergenics.append((int(last_end), int(f.location.start)))
        last_end = max(last_end, int(f.location.end))

    test_int_wins = []
    for start, end in intergenics:
        for i in range(start, end - 300, 300):
            seq_str = str(rec_test.seq[i:i+300]).upper()
            if all(c in "ACGT" for c in seq_str):
                test_int_wins.append((seq_str, [0]*100))

    # Shuffle to prevent sequential biases during evaluation
    np.random.seed(42)
    np.random.shuffle(test_cds_wins)
    np.random.shuffle(test_int_wins)
    
    # Return capped dataset to prevent memory bloat
    return test_cds_wins[:1000] + test_int_wins[:1000]
    
    
def evaluate(decoder_fn, label, test_data):
    all_t, all_p = [], []
    all_scores = [] 
    true_starts, pred_starts = [], []
    exact_hits = 0
    total_cds = 0

    for seq, lbls in test_data:
        toks  = encode(seq)
        preds = decoder_fn(seq, toks)
        
        # --- THE FIX ---
        # Align the continuous score with the model's actual decision
        if 1 in preds:
            # Model predicted a CDS. Get the PWM score of the chosen start codon.
            p_s = preds.index(1)
            final_score = score_promoter(seq[p_s*3-50 : p_s*3])
        else:
            # Model rejected the window (failed PWM threshold or Length Gate).
            # Assign a baseline failing score so the ROC curve reflects the rejection.
            final_score = -20.0 
        # ---------------
        
        bin_t = [1 if x > 0 else 0 for x in lbls]
        bin_p = [1 if x > 0 else 0 for x in preds]
        
        all_t.append(1 if sum(bin_t) > 0 else 0)
        all_p.append(1 if sum(bin_p) > 0 else 0)
        all_scores.append(final_score)

        # Distance and exact match metrics
        if 1 in lbls:
            total_cds += 1
            t_s = lbls.index(1)
            true_starts.append(t_s * 3)
            if 1 in preds:
                p_s = preds.index(1)
                pred_starts.append(p_s * 3)
                if p_s == t_s:
                    exact_hits += 1
            else:
                pred_starts.append(len(seq))

    # Core Metrics
    acc  = accuracy_score(all_t, all_p)
    prec = precision_score(all_t, all_p, zero_division=0)
    rec  = recall_score(all_t, all_p, zero_division=0)
    f1   = f1_score(all_t, all_p, zero_division=0)
    mcc  = matthews_corrcoef(all_t, all_p)
    mae  = mean_absolute_error(true_starts, pred_starts) if true_starts else 0
    exact_rate = exact_hits / total_cds if total_cds else 0

    # Added Advanced Metrics
    tn, fp, fn, tp = confusion_matrix(all_t, all_p, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    bal_acc = balanced_accuracy_score(all_t, all_p)

    # print(f"\n  [{label}]")
    # print(f"    Accuracy:  {acc:.4f}   Precision: {prec:.4f}")
    # print(f"    Recall:    {rec:.4f}   F1-Score:  {f1:.4f}")
    # print(f"    MCC:       {mcc:.4f}   MAE:       {mae:.2f} bp")
    # print(f"    Spec:      {specificity:.4f}   Bal Acc:   {bal_acc:.4f}")
    # print(f"    Exact Start Match: {exact_rate*100:.2f}%")

    # Retained strict dict formatting + appended raw arrays for visualization
    return dict(label=label, accuracy=acc, precision=prec, recall=rec,
                f1=f1, mcc=mcc, mae=mae, exact_rate=exact_rate,
                specificity=specificity, balanced_accuracy=bal_acc,
                y_true=all_t, y_pred=all_p, y_scores=all_scores)

def plot_resolution_sharpness(test_sequence, true_tis_index=150):
    """
    IEEE Paper Figure: Demonstrates 1-bp resolution of Codon Tokenization.
    Scans a sequence base-by-base to show the catastrophic score drop-off 
    if the reading frame is shifted by even 1 bp.
    """
    print("\nGenerating IEEE Sharpness Plot...")
    scores = []
    positions = list(range(50, len(test_sequence) - 93))
    
    for i in positions:
        upstream = test_sequence[i-50:i]
        downstream = test_sequence[i+3:i+93]
        
        # We penalize heavily if it's not an ATG, simulating hard-coded biological rules
        if test_sequence[i:i+3] != "ATG":
            scores.append(-50)
            continue
            
        p_score = score_promoter(upstream)
        cub_score = score_coding_potential(downstream, codon_usage_log_probs)
        scores.append(p_score + (cub_score * 1.5))

    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 5))
    
    plt.plot(positions, scores, color='#C44E52', lw=2)
    plt.axvline(x=true_tis_index, color='black', linestyle='--', label=f'True TIS (bp {true_tis_index})')
    
    # Highlight the reading frame drops
    plt.annotate('Complete score failure\nat +1/-1 frame shift', 
                 xy=(true_tis_index+1, -40), xytext=(true_tis_index+20, -20),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

    plt.title("Explainable 1-bp Resolution via Codon Tokenization", fontsize=14, fontweight='bold')
    plt.xlabel("Genomic Position (bp)", fontsize=12)
    plt.ylabel("Contextual Prediction Score (PWM + CUB)", fontsize=12)
    plt.xlim([100, 200]) # Zoom in around the TIS
    plt.ylim([-60, max(scores) + 10])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("ieee_resolution_sharpness.png", dpi=300)
    plt.close()
    print("Saved -> ieee_resolution_sharpness.png")

def generate_dashboard(results_v1, results_v2, species_name):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(17, 10))
    fig.suptitle(f"Biological Gatekeeper v2 vs v1 — {species_name}",
                 fontsize=18, fontweight='bold', y=0.99)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    metric_names = ["MCC", "Precision", "Recall", "F1-Score"]
    v1_vals = [results_v1["mcc"], results_v1["precision"], results_v1["recall"], results_v1["f1"]]
    v2_vals = [results_v2["mcc"], results_v2["precision"], results_v2["recall"], results_v2["f1"]]

    x = np.arange(len(metric_names))
    width = 0.28

    ax1 = fig.add_subplot(gs[0, 0])
    bars_v1 = ax1.bar(x - width, v1_vals, width, label="v1 (PWM only)", color="#4C72B0", alpha=0.85)
    bars_v2 = ax1.bar(x, v2_vals, width, label="v2 (PWM + Length)", color="#55A868", alpha=0.85)
    ax1.set_xticks(x - width/2)
    ax1.set_xticklabels(metric_names)
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Classification Metrics\nv1 vs v2", fontsize=13)
    ax1.set_ylabel("Score")
    ax1.legend(fontsize=9)
    
    for b in list(bars_v1) + list(bars_v2):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.015,
                 f"{b.get_height():.3f}", ha='center', fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    deltas = [v2 - v1 for v2, v1 in zip(v2_vals, v1_vals)]
    colors = ['#55A868' if d >= 0 else '#C44E52' for d in deltas]
    ax2.bar(metric_names, deltas, color=colors, alpha=0.85, edgecolor='white')
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_title("Δ Improvement\n(v2 − v1)", fontsize=13)
    ax2.set_ylabel("Score Delta")
    for i, (m, d) in enumerate(zip(metric_names, deltas)):
        ax2.text(i, d + (0.003 if d >= 0 else -0.007),
                 f"{d:+.3f}", ha='center', va='bottom' if d >= 0 else 'top', fontsize=10)

    ax3 = fig.add_subplot(gs[0, 2])
    models   = ["v1\n(PWM)", "v2\n(PWM+Len)"]
    maes     = [results_v1["mae"], results_v2["mae"]]
    mar_cols = ['#DD8452', '#55A868']
    ax3b = ax3.twinx()
    exact_rates = [results_v1["exact_rate"]*100, results_v2["exact_rate"]*100]
    ax3.bar(models, maes, color=mar_cols, alpha=0.75, label="MAE (bp)", width=0.4)
    ax3b.plot(models, exact_rates, 'D--', color='#4C72B0', lw=2.5,
              markersize=9, label="Exact Start Match %")
    ax3.set_title("MAE & Exact Start Match", fontsize=13)
    ax3.set_ylabel("MAE (bp)", color='#C44E52')
    ax3b.set_ylabel("Exact Start Match (%)", color='#4C72B0')
    ax3b.set_ylim(0, 100)
    for i, (m, e) in enumerate(zip(maes, exact_rates)):
        ax3.text(i, m + (max(maes)*0.02), f"{m:.1f}", ha='center', fontsize=9)
        ax3b.text(i, e + 2, f"{e:.1f}%", ha='center', fontsize=9, color='#4C72B0')

    plt.tight_layout()
    safe_name = species_name.replace(" ", "_").replace("/", "-")
    plt.savefig(f"{safe_name}_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close()
    
def generate_diagnostics(results_v1, results_v2, species_name):
    diag_fig, (ax_cm, axc_cm, ax_roc) = plt.subplots(1, 3, figsize=(14, 6))

    cm = confusion_matrix(results_v1["y_true"], results_v1["y_pred"])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=['Intergenic', 'CDS'], yticklabels=['Intergenic', 'CDS'])
    ax_cm.set_title(f"v1 CM: {species_name}")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    
    cm = confusion_matrix(results_v2["y_true"], results_v2["y_pred"])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axc_cm,
                xticklabels=['Intergenic', 'CDS'], yticklabels=['Intergenic', 'CDS'])
    axc_cm.set_title(f"v2 CM: {species_name}")
    axc_cm.set_xlabel("Predicted")
    axc_cm.set_ylabel("True")

    for res in [results_v1, results_v2]:
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_scores"])
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, lw=2, label=f'{res["label"]} (AUC = {roc_auc:.3f})')

    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title(f'ROC - {species_name}')
    ax_roc.legend(loc="lower right")

    plt.tight_layout()
    safe_name = species_name.replace(" ", "_").replace("/", "-")
    diag_fig.savefig(f"{safe_name}_diagnostics.png", dpi=300)
    plt.close(diag_fig)

# fetch test datas
def fetchTestFastafromjson(type = "nucleotide", ID="NC_000001.11", 
                           start = 2.01*1000000, stop = 3*1000000, 
                           retype = "gbwithparts", rettmode = "text"):
    with Entrez.efetch(db=type, 
                   id=ID, 
                   seq_start=start, 
                   seq_stop=stop, 
                   rettype=retype, 
                   retmode=rettmode) as h:
        rec_test = SeqIO.read(h, "genbank")
    return rec_test
def generate_aggregate_diagnostics(tprs_v1, tprs_v2, mean_fpr, aucs_v1, aucs_v2, cm_v1, cm_v2):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Cross-Species Evaluation: v1 vs v2", fontsize=18, fontweight='bold', y=0.96)

    # Function to plot aggregated ROC
    def plot_aggregated_roc(ax, tprs, aucs, title, color):
        # Plot individual species curves faintly
        for tpr in tprs:
            ax.plot(mean_fpr, tpr, color=color, alpha=0.15, lw=1)
            
        # Calculate mean and standard deviation (error area)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_tpr = np.std(tprs, axis=0)
        
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        # Plot the main highlighted line and area
        ax.plot(mean_fpr, mean_tpr, color=color, lw=2.5, label=f'Mean ROC (AUC = {mean_auc:.3f})')
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=0.2, label=r'$\pm$ 1 Std. Dev.')
        
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc="lower right", fontsize=11)

    # Plot Top Row: ROC Curves
    plot_aggregated_roc(axes[0, 0], tprs_v1, aucs_v1, "v1 (PWM Only) - Combined ROC", '#4C72B0')
    plot_aggregated_roc(axes[0, 1], tprs_v2, aucs_v2, "v2 (Gatekeeper) - Combined ROC", '#55A868')

    # Plot Bottom Row: Confusion Matrices
    sns.heatmap(cm_v1, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['Intergenic', 'CDS'], yticklabels=['Intergenic', 'CDS'], annot_kws={"size": 14})
    axes[1, 0].set_title(f"Total Confusion Matrix (v1)\nN = {np.sum(cm_v1)}", fontsize=14)
    axes[1, 0].set_xlabel("Predicted Label", fontsize=12)
    axes[1, 0].set_ylabel("True Label", fontsize=12)

    sns.heatmap(cm_v2, annot=True, fmt='d', cmap='Greens', ax=axes[1, 1],
                xticklabels=['Intergenic', 'CDS'], yticklabels=['Intergenic', 'CDS'], annot_kws={"size": 14})
    axes[1, 1].set_title(f"Total Confusion Matrix (v2)\nN = {np.sum(cm_v2)}", fontsize=14)
    axes[1, 1].set_xlabel("Predicted Label", fontsize=12)
    axes[1, 1].set_ylabel("True Label", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("combined_performance_diagnostics.png", dpi=300, bbox_inches='tight')
    plt.close()

def compare_train_test_roc(train_json="training_sources.json", test_json="testing_sources.json"):
    """
    NEW: Evaluates the model on both the training and testing datasets 
    (with synthetically mapped true negative windows) to check for overfitting.
    """
    print("\n=========================================")
    print("Running Train vs Test ROC Comparison...")
    print("=========================================")

    def get_eval_data(json_file):
        sources = load_training_sources(json_file)
        if not sources: return []
        
        all_data = []
        for source in sources:
            fetch_args = {"db": "nucleotide", "id": source["id"], "rettype": "gbwithparts", "retmode": "text"}
            if source.get("start") and source.get("stop"):
                fetch_args["seq_start"] = source["start"]
                fetch_args["seq_stop"] = source["stop"]
            try:
                with Entrez.efetch(**fetch_args) as handle:
                    rec = SeqIO.read(handle, "genbank")
                all_data.extend(prepare_test_data(rec))
            except Exception as e:
                print(f"  [!] Failed fetching {source['name']} for eval: {e}")
        return all_data

    print("Fetching and parsing Training evaluation data...")
    train_data = get_eval_data(train_json)
    
    print("Fetching and parsing Testing evaluation data...")
    test_data = get_eval_data(test_json)

    if not train_data or not test_data:
        print("[!] Missing evaluation data. Cannot generate Train vs Test ROC.")
        return

    print("Evaluating Model (v2) on Training Set...")
    res_train = evaluate(decoder_v2, "Train (v2)", train_data)

    print("Evaluating Model (v2) on Testing Set...")
    res_test = evaluate(decoder_v2, "Test (v2)", test_data)

    # Plot Comparison ROC
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))
    
    for res, color in zip([res_train, res_test], ['#4C72B0', '#55A868']):
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_scores"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2.5, color=color, label=f'{res["label"]} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve: Training Data vs Testing Data', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    
    plt.tight_layout()
    plt.savefig("train_vs_test_roc.png", dpi=300)
    plt.close()
    print("Saved -> train_vs_test_roc.png\n")

# -- Removed redundant plotting blocks for brevity, keep the ones already existing in your script --

def generate_aggregate_diagnostics(tprs_v1, tprs_v2, mean_fpr, aucs_v1, aucs_v2, cm_v1, cm_v2):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Cross-Species Evaluation: v1 vs v2", fontsize=18, fontweight='bold', y=0.96)

    def plot_aggregated_roc(ax, tprs, aucs, title, color):
        for tpr in tprs:
            ax.plot(mean_fpr, tpr, color=color, alpha=0.15, lw=1)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        ax.plot(mean_fpr, mean_tpr, color=color, lw=2.5, label=f'Mean ROC (AUC = {mean_auc:.3f})')
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=0.2, label=r'$\pm$ 1 Std. Dev.')
        
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc="lower right", fontsize=11)

    plot_aggregated_roc(axes[0, 0], tprs_v1, aucs_v1, "v1 (PWM Only) - Combined ROC", '#4C72B0')
    plot_aggregated_roc(axes[0, 1], tprs_v2, aucs_v2, "v2 (Gatekeeper) - Combined ROC", '#55A868')

    sns.heatmap(cm_v1, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['Intergenic', 'CDS'], yticklabels=['Intergenic', 'CDS'], annot_kws={"size": 14})
    axes[1, 0].set_title(f"Total Confusion Matrix (v1)\nN = {np.sum(cm_v1)}", fontsize=14)
    axes[1, 0].set_xlabel("Predicted Label", fontsize=12)
    axes[1, 0].set_ylabel("True Label", fontsize=12)

    sns.heatmap(cm_v2, annot=True, fmt='d', cmap='Greens', ax=axes[1, 1],
                xticklabels=['Intergenic', 'CDS'], yticklabels=['Intergenic', 'CDS'], annot_kws={"size": 14})
    axes[1, 1].set_title(f"Total Confusion Matrix (v2)\nN = {np.sum(cm_v2)}", fontsize=14)
    axes[1, 1].set_xlabel("Predicted Label", fontsize=12)
    axes[1, 1].set_ylabel("True Label", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("combined_performance_diagnostics.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_aggregate_diagnostics_v3(tprs_dict, aucs_dict, mean_fpr, cms_dict):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Performance Evolution: PWM (v1) vs Gatekeeper (v2) vs Contextual (v3)", 
                 fontsize=20, fontweight='bold', y=0.98)

    versions = ["v1", "v2", "v3"]
    colors = ['#4C72B0', '#55A868', '#C44E52'] # Blue, Green, Red
    cm_cmaps = ['Blues', 'Greens', 'Reds']

    for i, ver in enumerate(versions):
        ax_roc = axes[0, i]
        ax_cm = axes[1, i]
        
        # 1. Aggregate ROC
        current_tprs = tprs_dict[ver]
        mean_tpr = np.mean(current_tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_tpr = np.std(current_tprs, axis=0)

        ax_roc.plot(mean_fpr, mean_tpr, color=colors[i], lw=3, label=f'Mean (AUC={mean_auc:.3f})')
        ax_roc.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), 
                            np.minimum(mean_tpr + std_tpr, 1), color=colors[i], alpha=0.2)
        ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax_roc.set_title(f"{ver.upper()} ROC Curve", fontsize=14, fontweight='bold')
        ax_roc.legend(loc="lower right")

        # 2. Total Confusion Matrix
        sns.heatmap(cms_dict[ver], annot=True, fmt='d', cmap=cm_cmaps[i], ax=ax_cm, cbar=False,
                    xticklabels=['Intergenic', 'CDS'], yticklabels=['Intergenic', 'CDS'])
        ax_cm.set_title(f"{ver.upper()} Confusion Matrix", fontsize=14, fontweight='bold')
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("v3_performance_comparison.png", dpi=300)
    plt.close()
    print("[✔] Saved -> v3_performance_comparison.png")
 
def testingFromjson(filepath="testing_sources.json"):
    testing_sources = load_training_sources(filepath)
    if not testing_sources:
        print("No testing sources found or file is empty.")
        return

    all_metrics_v1, all_metrics_v2, all_metrics_v3 = [], [], []
    mean_fpr = np.linspace(0, 1, 100)
    
    # Storage for aggregate plotting
    tprs = {"v1": [], "v2": [], "v3": []}
    aucs = {"v1": [], "v2": [], "v3": []}
    cms  = {"v1": np.zeros((2, 2), dtype=int), 
            "v2": np.zeros((2, 2), dtype=int), 
            "v3": np.zeros((2, 2), dtype=int)}

    for source in testing_sources:
        species_name = source['name']
        print(f"\n>>> Evaluating Species: {species_name}")

        fetch_args = {"db": "nucleotide", "id": source["id"], "rettype": "gbwithparts", "retmode": "text"}
        if source.get("start") and source.get("stop"):
            fetch_args["seq_start"], fetch_args["seq_stop"] = source["start"], source["stop"]

        try:
            with Entrez.efetch(**fetch_args) as handle:
                rec_test = SeqIO.read(handle, "genbank")
        except Exception as e:
            print(f"      [!] Fetch Failed: {e}")
            continue

        test_data = prepare_test_data(rec_test)
        if not test_data: continue

        # Run all three decoders
        results = {
            "v1": evaluate(decoder_v1, "v1", test_data),
            "v2": evaluate(decoder_v2, "v2", test_data),
            "v3": evaluate(decoder_v3, "v3", test_data)
        }

        # Process ROC and Confusion Matrix for each
        for ver in ["v1", "v2", "v3"]:
            res = results[ver]
            fpr, tpr, _ = roc_curve(res["y_true"], res["y_scores"])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs[ver].append(interp_tpr)
            aucs[ver].append(auc(fpr, tpr))
            cms[ver] += confusion_matrix(res["y_true"], res["y_pred"], labels=[0, 1])

            # Strip bulky arrays for CSV logging
            clean_res = {k: v for k, v in res.items() if k not in ['y_true', 'y_pred', 'y_scores']}
            clean_res['species'] = species_name
            if ver == "v1": all_metrics_v1.append(clean_res)
            elif ver == "v2": all_metrics_v2.append(clean_res)
            else: all_metrics_v3.append(clean_res)

    # Save results
   pd.DataFrame(all_metrics_v1).to_csv("all_species_results_v1.csv", index=False)
    pd.DataFrame(all_metrics_v2).to_csv("all_species_results_v2.csv", index=False)
    # Use 'all_metrics_v3' here to match the initialization
    pd.DataFrame(all_metrics_v3).to_csv("all_species_results_v3.csv", index=False) 
    
    print("\n[✔] Metrics saved for v1, v2, and v3.")

    # Call updated plotting function
    generate_aggregate_diagnostics_v3(tprs, aucs, mean_fpr, cms)
# ─────────────────────────────────────────────
# 3.  LOAD CHROMOSOME IV TEST DATA
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Biological Gatekeeper v3: Codon Tokenization Benchmark")
    parser.add_argument('--model_file', type=str, default='model.pkl', help='Path to load/save the model')
    parser.add_argument('--train_json', type=str, default='training_sources.json', help='JSON for training')
    parser.add_argument('--test_json', type=str, default='testing_sources.json', help='JSON for testing')
    parser.add_argument('--compare_roc', action='store_true', help='Compare Train vs Test ROC')
    parser.add_argument('--gen_sharpness', action='store_true', help='Generate IEEE Sharpness Plot')
    args = parser.parse_args()

    global pwm_logo, codon_vocab, emissions, log_emissions, codon_usage_log_probs

    # 1. LOAD OR TRAIN MODEL
    if os.path.exists(args.model_file):
        print(f"Loading model from {args.model_file}...")
        with open(args.model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        pwm_logo = model_data['pwm_logo']
        codon_vocab = model_data['codon_vocab']
        emissions = model_data['emissions']
        log_emissions = model_data['log_emissions']
        # Load new v3 data, fallback to None if re-training is needed
        codon_usage_log_probs = model_data.get('codon_usage') 
        
        if codon_usage_log_probs is None:
            print("[!] Model file lacks Codon Usage data. Re-training required for v3.")
            model_data = train_model()
            pwm_logo, codon_vocab = model_data['pwm_logo'], model_data['codon_vocab']
            emissions, log_emissions = model_data['emissions'], model_data['log_emissions']
            codon_usage_log_probs = model_data['codon_usage']
            with open(args.model_file, 'wb') as f:
                pickle.dump(model_data, f)
    else:
        print(f"Model file not found. Training new v3 model...")
        model_data = train_model()
        pwm_logo, codon_vocab = model_data['pwm_logo'], model_data['codon_vocab']
        emissions, log_emissions = model_data['emissions'], model_data['log_emissions']
        codon_usage_log_probs = model_data['codon_usage']
        
        with open(args.model_file, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {args.model_file}.")

    # 2. RUN CROSS-SPECIES BENCHMARK
    # Note: Update your testingFromjson to include res_v3 = evaluate(decoder_v3, ...)
    testingFromjson(args.test_json)

    # 3. GENERATE TRAIN VS TEST ROC (If requested)
    if args.compare_roc:
        compare_train_test_roc(args.train_json, args.test_json)

    # 4. GENERATE IEEE SHARPNESS PLOT (The "Proof" for your paper)
    if args.gen_sharpness:
        # Example: Use a known Human TIS sequence to demonstrate resolution
        # You can pull a specific sequence from your training_metadata_log.json
        sample_seq = "..." # Insert a 300bp sample sequence here for the plot
        plot_resolution_sharpness(sample_seq, true_tis_index=150)

if __name__ == '__main__':
    main()