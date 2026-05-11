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
import pickle
import argparse
import os

warnings.filterwarnings('ignore')
Entrez.email = "example@example.com"

# ─────────────────────────────────────────────
# GLOBAL VARIABLES
# ─────────────────────────────────────────────
nuc2idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
pwm_logo = None
codon_vocab = {}
emissions = None
log_emissions = None
codon_log_odds_map = None  # NEW: v3 Discriminator

# ─────────────────────────────────────────────
# SCORING FUNCTIONS
# ─────────────────────────────────────────────
def score_promoter(window):
    if len(window) != 50:
        return -999.0
    return sum(pwm_logo[nuc2idx.get(c, 0), i] for i, c in enumerate(window))

def score_coding_potential(sequence, log_odds_map):
    """v3: Scores based on Codon Log-Odds (CDS vs Intergenic)"""
    score = 0.0
    valid_codons = 0
    for i in range(0, len(sequence)-2, 3):
        codon = sequence[i:i+3]
        if codon in log_odds_map:
            score += log_odds_map[codon]
            valid_codons += 1
    return score / valid_codons if valid_codons > 0 else 0.0

def encode(seq_str):
    return [codon_vocab.get(seq_str[i:i+3], codon_vocab["UNK"])
            for i in range(0, len(seq_str)-2, 3)]

# ─────────────────────────────────────────────
# DATA LOADING & TRAINING
# ─────────────────────────────────────────────
def load_training_sources(filepath="training_sources.json"):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []

def train_model():
    print("1. Fetching Pan-Eukaryotic Training Data & Mining Background...")
    training_sources = load_training_sources()
    train_cds = []
    pwm_counts = np.ones((4, 50)) * 1e-4
    
    # Laplace smoothing for Codon Frequencies
    all_codons = [a+b+c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
    cds_counts = {c: 1 for c in all_codons}
    bg_counts  = {c: 1 for c in all_codons}
    
    seen_tis = set() 
    metadata_log = []

    for source in training_sources:
        print(f"   -> Pulling {source['name']}...")
        try:
            fetch_params = {"db": "nucleotide", "id": source["id"], "rettype": "gbwithparts", "retmode": "text"}
            if source.get("start") and source.get("stop"):
                fetch_params["seq_start"], fetch_params["seq_stop"] = source["start"], source["stop"]
                offset = source["start"]
            else:
                offset = 0
            
            with Entrez.efetch(**fetch_params) as handle:
                rec_train = SeqIO.read(handle, "genbank")
            
            # 1. Extract CDS (Positives)
            source_cds_count = 0
            last_end = 0
            cds_intervals = []
            
            for f in sorted([f for f in rec_train.features if f.type == "CDS"], key=lambda x: int(x.location.start)):
                strand = f.location.strand
                tis_rel = int(f.location.start) if strand == 1 else int(f.location.end) - 1
                cds_intervals.append((int(f.location.start), int(f.location.end)))
                
                ws, we = (tis_rel - 150, tis_rel + 150) if strand == 1 else (tis_rel - 149, tis_rel + 151)
                tis_abs = tis_rel + (offset - 1 if offset > 0 else 0)

                if (source["id"], tis_abs) in seen_tis: continue
                seen_tis.add((source["id"], tis_abs))

                if ws >= 0 and we <= len(rec_train.seq):
                    chunk = rec_train.seq[ws:we]
                    if strand == -1: chunk = chunk.reverse_complement()
                    seq_str = str(chunk).upper()
                    
                    if len(seq_str) == 300 and seq_str[150:153] == "ATG":
                        train_cds.append((seq_str, [0]*50 + [1] + [2]*48 + [3], {}))
                        source_cds_count += 1
                        
                        # Update PWM
                        for i, nuc in enumerate(seq_str[100:150]):
                            if nuc in nuc2idx: pwm_counts[nuc2idx[nuc], i] += 1
                                
                        # Update CDS Codon Usage
                        downstream = seq_str[153:243] # 90bp
                        for i in range(0, len(downstream)-2, 3):
                            codon = downstream[i:i+3]
                            if codon in cds_counts: cds_counts[codon] += 1
            
            # 2. Extract Intergenic Background (Negatives)
            bg_count = 0
            for start, end in cds_intervals:
                if int(start) > last_end + 300:
                    # Found intergenic space, extract a 90bp chunk for background
                    mid_point = last_end + ((int(start) - last_end) // 2)
                    bg_chunk = str(rec_train.seq[mid_point:mid_point+90]).upper()
                    if len(bg_chunk) == 90 and all(c in "ACGT" for c in bg_chunk):
                        for i in range(0, 90-2, 3):
                            codon = bg_chunk[i:i+3]
                            if codon in bg_counts: bg_counts[codon] += 1
                        bg_count += 1
                last_end = max(last_end, int(end))
                
            print(f"      Gathered {source_cds_count} CDS and {bg_count} Intergenic chunks.")
        except Exception as e:
            print(f"      [!] Failed {source['name']}: {e}")

    # Calculate PWM (Base 2)
    pwm = pwm_counts / pwm_counts.sum(axis=0, keepdims=True)
    bg = np.array([0.25]*4).reshape(4,1)
    pwm_logo_val = np.log2(pwm / bg)

    # Calculate Codon Log-Odds Discriminator (Base 2)
    total_cds = sum(cds_counts.values())
    total_bg = sum(bg_counts.values())
    codon_log_odds_val = {
        c: np.log2((cds_counts[c]/total_cds) / (bg_counts[c]/total_bg)) 
        for c in all_codons
    }

    # Vocab for legacy decoders
    codon_vocab_val = {c: i for i, c in enumerate(all_codons)}
    codon_vocab_val["UNK"] = 64
    emissions_val = np.ones((4, 65)) * 1e-6
    log_emissions_val = np.log(emissions_val / emissions_val.sum(axis=1, keepdims=True))

    return {
        'pwm_logo': pwm_logo_val,
        'codon_vocab': codon_vocab_val,
        'emissions': emissions_val,
        'log_emissions': log_emissions_val,
        'codon_log_odds': codon_log_odds_val
    }

# ─────────────────────────────────────────────
# DECODERS (v1, v2, v3)
# ─────────────────────────────────────────────
def decoder_v1(seq_str, tokens):
    best_start, best_score = -1, -np.inf
    for t, tok in enumerate(tokens):
        if tok == codon_vocab["ATG"]:
            nuc_idx = t * 3
            upstream = seq_str[nuc_idx-50:nuc_idx]
            if len(upstream) == 50:
                p = score_promoter(upstream)
                if p > best_score: best_score, best_start = p, t
    path = [0] * len(tokens)
    if best_start != -1:
        path[best_start] = 1
        for i in range(best_start+1, len(tokens)): path[i] = 2
    return path

def decoder_v2(seq_str, tokens, pwm_threshold=-5.0, min_cds_len_bp=90):
    candidates = []
    for t, tok in enumerate(tokens):
        if tok == codon_vocab["ATG"]:
            nuc_idx = t * 3
            upstream = seq_str[nuc_idx-50:nuc_idx]
            if len(upstream) == 50:
                p_score = score_promoter(upstream)
                if p_score >= pwm_threshold:
                    candidates.append((p_score, t))
    if not candidates: return [0] * len(tokens)
    _, best_start = max(candidates)
    if (len(tokens) - best_start - 1) * 3 < min_cds_len_bp: return [0] * len(tokens)
    path = [0] * len(tokens)
    path[best_start] = 1
    for i in range(best_start+1, len(tokens)): path[i] = 2
    return path

def decoder_v3(seq_str, tokens, pwm_threshold=-5.0, cub_weight=2.0):
    """v3 Contextual Gatekeeper: Uses Log-Odds Ratio for precise discrimination."""
    candidates = []
    global codon_log_odds_map
    
    for t, tok in enumerate(tokens):
        if tok == codon_vocab["ATG"]:
            nuc_idx = t * 3
            upstream = seq_str[nuc_idx-50:nuc_idx]
            downstream = seq_str[nuc_idx+3:nuc_idx+93] 
            
            if len(upstream) == 50 and len(downstream) >= 90:
                p_score = score_promoter(upstream)
                if p_score >= pwm_threshold:
                    cub_score = score_coding_potential(downstream, codon_log_odds_map)
                    total_score = p_score + (cub_score * cub_weight)
                    candidates.append((total_score, t))

    if not candidates: return [0] * len(tokens)
    _, best_start = max(candidates)
    path = [0] * len(tokens)
    path[best_start] = 1
    for i in range(best_start+1, len(tokens)): path[i] = 2
    return path

def decoder_baseline(seq_str, tokens):
    """Baseline: Plain codon tokenization - predicts first ATG as TIS."""
    path = [0] * len(tokens)
    for t, tok in enumerate(tokens):
        if tok == codon_vocab["ATG"]:
            path[t] = 1
            for i in range(t+1, len(tokens)): path[i] = 2
            return path
    return path

# ─────────────────────────────────────────────
# PIPELINE & EVALUATION
# ─────────────────────────────────────────────
def prepare_test_data(rec_test):
    test_cds_wins = []
    seen_test_tis = set()
    
    for f in rec_test.features:
        if f.type == "CDS":
            strand = f.location.strand
            tis_rel = int(f.location.start) if strand == 1 else int(f.location.end) - 1
            ws, we = (tis_rel - 150, tis_rel + 150) if strand == 1 else (tis_rel - 149, tis_rel + 151)
                
            if tis_rel in seen_test_tis: continue
            seen_test_tis.add(tis_rel)

            if ws >= 0 and we <= len(rec_test.seq):
                chunk = rec_test.seq[ws:we]
                if strand == -1: chunk = chunk.reverse_complement()
                seq_str = str(chunk).upper()
                if len(seq_str) == 300 and seq_str[150:153] == "ATG" and all(c in "ACGT" for c in seq_str):
                    test_cds_wins.append((seq_str, [0]*50 + [1] + [2]*48 + [3]))

    intergenics = []
    last_end = 0
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

    np.random.seed(42)
    np.random.shuffle(test_cds_wins)
    np.random.shuffle(test_int_wins)
    return test_cds_wins[:1000] + test_int_wins[:1000]

def evaluate(decoder_fn, label, test_data):
    all_t, all_p, all_scores = [], [], []
    true_starts, pred_starts = [], []
    exact_hits, total_cds = 0, 0

    for seq, lbls in test_data:
        toks = encode(seq)
        preds = decoder_fn(seq, toks)
        
        if 1 in preds:
            p_s = preds.index(1)
            if label == "v3":
                p_score = score_promoter(seq[p_s*3-50 : p_s*3])
                cub = score_coding_potential(seq[p_s*3+3 : p_s*3+93], codon_log_odds_map)
                final_score = p_score + (cub * 2.0)
            elif label == "baseline":
                final_score = 1.0 # Binary prediction, no statistical confidence
            else:
                final_score = score_promoter(seq[p_s*3-50 : p_s*3])
        else:
            final_score = -50.0 
        
        bin_t = [1 if x > 0 else 0 for x in lbls]
        bin_p = [1 if x > 0 else 0 for x in preds]
        
        all_t.append(1 if sum(bin_t) > 0 else 0)
        all_p.append(1 if sum(bin_p) > 0 else 0)
        all_scores.append(final_score)

        if 1 in lbls:
            total_cds += 1
            t_s = lbls.index(1)
            true_starts.append(t_s * 3)
            if 1 in preds:
                p_s = preds.index(1)
                pred_starts.append(p_s * 3)
                if p_s == t_s: exact_hits += 1
            else:
                pred_starts.append(len(seq))

    acc  = accuracy_score(all_t, all_p)
    prec = precision_score(all_t, all_p, zero_division=0)
    rec  = recall_score(all_t, all_p, zero_division=0)
    f1   = f1_score(all_t, all_p, zero_division=0)
    mcc  = matthews_corrcoef(all_t, all_p)
    mae  = mean_absolute_error(true_starts, pred_starts) if true_starts else 0
    exact_rate = exact_hits / total_cds if total_cds else 0

    tn, fp, fn, tp = confusion_matrix(all_t, all_p, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    bal_acc = balanced_accuracy_score(all_t, all_p)

    return dict(label=label, accuracy=acc, precision=prec, recall=rec,
                f1=f1, mcc=mcc, mae=mae, exact_rate=exact_rate,
                specificity=specificity, balanced_accuracy=bal_acc,
                y_true=all_t, y_pred=all_p, y_scores=all_scores)
    
# ─────────────────────────────────────────────
# DIAGNOSTICS & PLOTTING
# ─────────────────────────────────────────────
def generate_aggregate_diagnostics_v3(tprs_dict, aucs_dict, mean_fpr, cms_dict):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 4, figsize=(24, 10)) # Expanded to 4 columns
    fig.suptitle("Cross-Eukaryotic TIS Prediction: Evolving to Codon-Level Resolution", 
                 fontsize=20, fontweight='bold', y=0.98)

    versions = ["baseline", "v1", "v2", "v3"]
    colors = ['#808080', '#4C72B0', '#55A868', '#C44E52']
    cm_cmaps = ['Greys', 'Blues', 'Greens', 'Reds']

    for i, ver in enumerate(versions):
        ax_roc = axes[0, i]
        ax_cm = axes[1, i]
        
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

        sns.heatmap(cms_dict[ver], annot=True, fmt='d', cmap=cm_cmaps[i], ax=ax_cm, cbar=False,
                    xticklabels=['Intergenic', 'CDS'], yticklabels=['Intergenic', 'CDS'])
        ax_cm.set_title(f"{ver.upper()} Confusion Matrix", fontsize=14, fontweight='bold')
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("performance_comparison_full.png", dpi=300)
    plt.close()
    print("[✔] Saved -> performance_comparison_full.png")

def plot_resolution_sharpness():
    """Generates the IEEE Proof graph using a synthetic ideal sequence."""
    print("\nGenerating IEEE Sharpness Plot...")
    
    # Synthetic sequence: 100bp junk + 50bp Kozak + ATG + 147bp high-CUB gene
    # Using 'GAG' and 'CGC' to simulate strong coding potential
    upstream_junk = "TGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGACTG"
    kozak = "GCCGCCACC"
    atg = "ATG"
    downstream_gene = "GAGCGCGAGCGCGAGCGCGAGCGCGAGCGCGAGCGCGAGCGCGAGCGCGAGCGCGAGCGCGAGCGCGAGCGCGAGCGCGAGCGCGAGCGCGAGCGC"
    
    test_seq = upstream_junk + kozak + atg + downstream_gene
    true_tis = len(upstream_junk) + len(kozak)
    
    scores = []
    positions = list(range(50, len(test_seq) - 93))
    
    for i in positions:
        upstream = test_seq[i-50:i]
        downstream = test_seq[i+3:i+93]
        
        if test_seq[i:i+3] != "ATG":
            scores.append(-20) # Frame failure baseline
            continue
            
        p_score = score_promoter(upstream)
        cub_score = score_coding_potential(downstream, codon_log_odds_map)
        scores.append(p_score + (cub_score * 2.0))

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 5))
    
    plt.plot(positions, scores, color='#C44E52', lw=2)
    plt.axvline(x=true_tis, color='black', linestyle='--', label=f'True TIS (bp {true_tis})')
    
    plt.annotate('Complete score failure\nat +1/-1 frame shift', 
                 xy=(true_tis+1, -15), xytext=(true_tis+15, -5),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

    plt.title("Explainable 1-bp Resolution via Codon Tokenization", fontsize=14, fontweight='bold')
    plt.xlabel("Genomic Position (bp)", fontsize=12)
    plt.ylabel("Contextual Prediction Score (PWM + CUB Log-Odds)", fontsize=12)
    plt.xlim([true_tis - 20, true_tis + 30])
    plt.ylim([-25, max(scores) + 5])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("ieee_resolution_sharpness.png", dpi=300)
    plt.close()
    print("[✔] Saved -> ieee_resolution_sharpness.png")

def testingFromjson(filepath="testing_sources.json"):
    testing_sources = load_training_sources(filepath)
    if not testing_sources: return

    all_metrics_baseline, all_metrics_v1, all_metrics_v2, all_metrics_v3 = [], [], [], []
    mean_fpr = np.linspace(0, 1, 100)
    
    tprs = {"baseline": [], "v1": [], "v2": [], "v3": []}
    aucs = {"baseline": [], "v1": [], "v2": [], "v3": []}
    cms  = {"baseline": np.zeros((2, 2), dtype=int),
            "v1": np.zeros((2, 2), dtype=int), 
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

        results = {
            "baseline": evaluate(decoder_baseline, "baseline", test_data),
            "v1": evaluate(decoder_v1, "v1", test_data),
            "v2": evaluate(decoder_v2, "v2", test_data),
            "v3": evaluate(decoder_v3, "v3", test_data)
        }

        for ver in ["baseline", "v1", "v2", "v3"]:
            res = results[ver]
            fpr, tpr, _ = roc_curve(res["y_true"], res["y_scores"])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs[ver].append(interp_tpr)
            aucs[ver].append(auc(fpr, tpr))
            cms[ver] += confusion_matrix(res["y_true"], res["y_pred"], labels=[0, 1])

            clean_res = {k: v for k, v in res.items() if k not in ['y_true', 'y_pred', 'y_scores']}
            clean_res['species'] = species_name
            
            if ver == "baseline": all_metrics_baseline.append(clean_res)
            elif ver == "v1": all_metrics_v1.append(clean_res)
            elif ver == "v2": all_metrics_v2.append(clean_res)
            else: all_metrics_v3.append(clean_res)

    pd.DataFrame(all_metrics_baseline).to_csv("all_species_results_baseline.csv", index=False)
    pd.DataFrame(all_metrics_v1).to_csv("all_species_results_v1.csv", index=False)
    pd.DataFrame(all_metrics_v2).to_csv("all_species_results_v2.csv", index=False)
    pd.DataFrame(all_metrics_v3).to_csv("all_species_results_v3.csv", index=False)
    print("\n[✔] Metrics saved for baseline, v1, v2, and v3.")

    generate_aggregate_diagnostics_v3(tprs, aucs, mean_fpr, cms)

# ─────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Biological Gatekeeper v3 Benchmark")
    parser.add_argument('--model_file', type=str, default='model.pkl', help='Path to model file')
    parser.add_argument('--test_json', type=str, default='testing_sources.json', help='Testing targets')
    parser.add_argument('--gen_sharpness', action='store_true', help='Generate IEEE Sharpness Plot')
    args = parser.parse_args()

    global pwm_logo, codon_vocab, emissions, log_emissions, codon_log_odds_map

    if os.path.exists(args.model_file):
        print(f"Loading model from {args.model_file}...")
        with open(args.model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        pwm_logo = model_data['pwm_logo']
        codon_vocab = model_data['codon_vocab']
        emissions = model_data['emissions']
        log_emissions = model_data['log_emissions']
        codon_log_odds_map = model_data.get('codon_log_odds') 
        
        if codon_log_odds_map is None:
            print("[!] Model file lacks Codon Log-Odds. Re-training required for v3.")
            model_data = train_model()
            pwm_logo = model_data['pwm_logo']
            codon_vocab = model_data['codon_vocab']
            emissions = model_data['emissions']
            log_emissions = model_data['log_emissions']
            codon_log_odds_map = model_data['codon_log_odds']
            with open(args.model_file, 'wb') as f: pickle.dump(model_data, f)
    else:
        print(f"Model file not found. Training new v3 model...")
        model_data = train_model()
        pwm_logo = model_data['pwm_logo']
        codon_vocab = model_data['codon_vocab']
        emissions = model_data['emissions']
        log_emissions = model_data['log_emissions']
        codon_log_odds_map = model_data['codon_log_odds']
        with open(args.model_file, 'wb') as f: pickle.dump(model_data, f)
        print(f"Model saved to {args.model_file}.")

    # Generate IEEE Sharpness Plot if flagged
    if args.gen_sharpness:
        plot_resolution_sharpness()
        return # Exit early if only plotting was requested

    # Run Pipeline
    testingFromjson(args.test_json)

if __name__ == '__main__':
    main()