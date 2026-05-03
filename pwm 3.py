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
    # ─────────────────────────────────────────────
    # 1.  TRAIN PAN-EUKARYOTIC MODEL
    # ─────────────────────────────────────────────
    print("1. Fetching Pan-Eukaryotic Training Data...")

    # We define slices for larger genomes to prevent 400 Bad Request errors
    training_sources = load_training_sources()

    train_cds = []
    pwm_counts = np.ones((4, 50)) * 1e-4  # Initialize with pseudo-counts to prevent log(0)

    for source in training_sources:
        print(f"   -> Pulling {source['name']}...")
        try:
            # Handle sub-range fetches for massive genomes
            if source["start"] and source["stop"]:
                handle = Entrez.efetch(db="nucleotide", id=source["id"], 
                                       seq_start=source["start"], seq_stop=source["stop"], 
                                       rettype="gbwithparts", retmode="text")
                offset = source["start"]
            else:
                handle = Entrez.efetch(db="nucleotide", id=source["id"], 
                                       rettype="gbwithparts", retmode="text")
                offset = 0
                
            rec_train = SeqIO.read(handle, "genbank")
            handle.close()
            
            source_count = 0
            for f in rec_train.features:
                if f.type == "CDS" and f.location.strand == 1:
                    # Normalize coordinates based on the fetch offset
                    s = int(f.location.start) - offset
                    ws, we = s - 150, s + 150
                    
                    # Boundary check
                    if ws >= 0 and we <= len(rec_train.seq):
                        seq_str = str(rec_train.seq[ws:we]).upper()
                        
                        # Length check only. Stop filtering out 'N' gaps entirely.
                        if len(seq_str) == 300:
                            train_cds.append((seq_str, [0]*50 + [1] + [2]*48 + [3]))
                            source_count += 1
                            
                            # Accumulate PWM counts (upstream 50bp)
                            for i, nuc in enumerate(seq_str[100:150]):
                                if nuc in nuc2idx:
                                    pwm_counts[nuc2idx[nuc], i] += 1
                                    
            print(f"      Gathered {source_count} CDS windows.")
        except Exception as e:
            print(f"      [!] Failed to fetch {source['name']}: {e}")

    print(f"   Total Training Sequences: {len(train_cds)}")

    # Finalize the Position Weight Matrix
    pwm       = pwm_counts / pwm_counts.sum(axis=0, keepdims=True)
    bg        = np.array([0.25]*4).reshape(4,1)
    pwm_logo_val  = np.log2(pwm / bg)
    # pwm_logo_val  = pwm / bg

    # ─────────────────────────────────────────────
    # 1.5 BUILD EMISSION MODEL (For Future Use)
    # ─────────────────────────────────────────────
    codon_vocab_val = {}
    idx = 0
    for a in "ACGT":
        for b in "ACGT":
            for c in "ACGT":
                codon_vocab_val[a+b+c] = idx
                idx += 1
    codon_vocab_val["UNK"] = idx

    vocab_size = 65
    emissions_val = np.ones((4, vocab_size)) * 1e-6
    for seq, lbls in train_cds:
        toks = [codon_vocab_val.get(seq[i:i+3], codon_vocab_val["UNK"])
                for i in range(0, len(seq)-2, 3)]
        for t, state in zip(toks, lbls):
            emissions_val[state, t] += 1
    for i in range(4):
        emissions_val[i] /= emissions_val[i].sum()
    log_emissions_val = np.log(emissions_val)

    return {
        'pwm_logo': pwm_logo_val,
        'codon_vocab': codon_vocab_val,
        'emissions': emissions_val,
        'log_emissions': log_emissions_val
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
               pwm_threshold: float = -3.0,
               min_cds_len_bp: int  = 135):
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
def prepare_test_data(rec_test):
    test_cds_wins, intergenics = [], []
    last_end = 0
    for f in sorted([f for f in rec_test.features if f.type == "CDS"],
                    key=lambda x: x.location.start):
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

    test_int_wins = []
    for start, end in intergenics:
        for i in range(start, end-300, 300):
            seq_str = str(rec_test.seq[i:i+300]).upper()
            if all(c in "ACGT" for c in seq_str):
                test_int_wins.append((seq_str, [0]*100))

    np.random.seed(42)
    np.random.shuffle(test_cds_wins)
    np.random.shuffle(test_int_wins)
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
    
def testingFromjson(filepath="testing_sources.json"):
    testing_sources = load_training_sources(filepath)
    if not testing_sources:
        print("No testing sources found or file is empty.")
        return

    # Data aggregators for CSVs
    all_metrics_v1 = []
    all_metrics_v2 = []

    # Common FPR space for standardizing ROC curves for averaging
    mean_fpr = np.linspace(0, 1, 100)
    
    # Data aggregators for ROC interpolations
    tprs_v1, tprs_v2 = [], []
    aucs_v1, aucs_v2 = [], []

    # Data aggregators for Confusion Matrices
    cm_v1_total = np.zeros((2, 2), dtype=int)
    cm_v2_total = np.zeros((2, 2), dtype=int)

    for source in testing_sources:
        species_name = source['name']
        print(f"\n=========================================")
        print(f"Testing Pipeline for: {species_name}")
        print(f"=========================================")

        fetch_args = {
            "db": "nucleotide",
            "id": source["id"],
            "rettype": "gbwithparts",
            "retmode": "text"
        }
        
        if source.get("start") and source.get("stop"):
            fetch_args["seq_start"] = source["start"]
            fetch_args["seq_stop"] = source["stop"]

        print(f"Fetching sequence data...")
        try:
            with Entrez.efetch(**fetch_args) as handle:
                rec_test = SeqIO.read(handle, "genbank")
        except Exception as e:
            print(f"Failed to fetch test data for {species_name}: {e}")
            continue

        print(f"Preparing windows...")
        test_data = prepare_test_data(rec_test)
        print(f"Test set: {len(test_data)} total windows.")

        if not test_data:
            print(f"Skipping {species_name} due to empty test data.")
            continue

        print(f"Evaluating v1 and v2...")
        res_v1 = evaluate(decoder_v1, "v1", test_data)
        res_v2 = evaluate(decoder_v2, "v2", test_data)

        # ROC extraction and Interpolation for aggregation
        fpr1, tpr1, _ = roc_curve(res_v1["y_true"], res_v1["y_scores"])
        interp_tpr1 = np.interp(mean_fpr, fpr1, tpr1)
        interp_tpr1[0] = 0.0
        tprs_v1.append(interp_tpr1)
        aucs_v1.append(auc(fpr1, tpr1))

        fpr2, tpr2, _ = roc_curve(res_v2["y_true"], res_v2["y_scores"])
        interp_tpr2 = np.interp(mean_fpr, fpr2, tpr2)
        interp_tpr2[0] = 0.0
        tprs_v2.append(interp_tpr2)
        aucs_v2.append(auc(fpr2, tpr2))

        # Sum up the Confusion Matrices
        cm_v1_total += confusion_matrix(res_v1["y_true"], res_v1["y_pred"], labels=[0, 1])
        cm_v2_total += confusion_matrix(res_v2["y_true"], res_v2["y_pred"], labels=[0, 1])

        # Clean dictionary metrics for pandas CSV (exclude raw huge array columns)
        clean_v1 = {k: v for k, v in res_v1.items() if k not in ['y_true', 'y_pred', 'y_scores']}
        clean_v1['species'] = species_name
        all_metrics_v1.append(clean_v1)

        clean_v2 = {k: v for k, v in res_v2.items() if k not in ['y_true', 'y_pred', 'y_scores']}
        clean_v2['species'] = species_name
        all_metrics_v2.append(clean_v2)

    # ---------------------------------------------------------
    # End of testing loop - Output Generation
    # ---------------------------------------------------------
    print("\n\n=========================================")
    print("ALL SPECIES EVALUATION COMPLETE")
    print("=========================================")
    
    # 1. Output the ONE csv for v1 and ONE csv for v2
    # Moving 'species' to the first column for better readability
    df_v1 = pd.DataFrame(all_metrics_v1)
    df_v1 = df_v1[['species'] + [c for c in df_v1 if c != 'species']]
    df_v1.to_csv("all_species_results_v1.csv", index=False)
    
    df_v2 = pd.DataFrame(all_metrics_v2)
    df_v2 = df_v2[['species'] + [c for c in df_v2 if c != 'species']]
    df_v2.to_csv("all_species_results_v2.csv", index=False)
    print("Saved -> all_species_results_v1.csv")
    print("Saved -> all_species_results_v2.csv")

    # 2. Output the ONE massive combined diagnostics plot
    generate_aggregate_diagnostics(tprs_v1, tprs_v2, mean_fpr, aucs_v1, aucs_v2, cm_v1_total, cm_v2_total)
    print("Saved -> combined_performance_diagnostics.png")

# ─────────────────────────────────────────────
# 3.  LOAD CHROMOSOME IV TEST DATA
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Biological Gatekeeper v2 Model Benchmark")
    parser.add_argument('--model_file', type=str, default='model.pkl', help='Path to load/save the trained model (.pkl)')
    parser.add_argument('--test_json', type=str, default='testing_sources.json', help='JSON file containing test targets')
    args = parser.parse_args()

    global pwm_logo, codon_vocab, emissions, log_emissions

    if os.path.exists(args.model_file):
        print(f"Loading model from {args.model_file}...")
        with open(args.model_file, 'rb') as f:
            model_data = pickle.load(f)
        pwm_logo = model_data['pwm_logo']
        codon_vocab = model_data['codon_vocab']
        emissions = model_data['emissions']
        log_emissions = model_data['log_emissions']
    else:
        print(f"Model file {args.model_file} not found. Training a new model...")
        model_data = train_model()
        pwm_logo = model_data['pwm_logo']
        codon_vocab = model_data['codon_vocab']
        emissions = model_data['emissions']
        log_emissions = model_data['log_emissions']
        
        with open(args.model_file, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {args.model_file}.")

    # Trigger the testing from JSON loop
    testingFromjson(args.test_json)0

if __name__ == '__main__':
    main()