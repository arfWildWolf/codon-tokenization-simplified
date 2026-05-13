import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score, auc,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
UPSTREAM = 150
DOWNSTREAM = 150
PWM_WINDOW = 150     # Evaluate 50bp immediately preceding the start codon
CODON_WINDOW = 150   # Evaluate 90bp (30 codons) downstream of the start codon
nuc2idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# ─────────────────────────────────────────────
# SCORING & TOKENIZATION
# ─────────────────────────────────────────────
def generate_vocab():
    bases = ['A', 'C', 'G', 'T']
    codons = [a+b+c for a in bases for b in bases for c in bases]
    vocab = {codon: i for i, codon in enumerate(codons)}
    vocab['UNK'] = 64
    return vocab

def encode(seq_str, codon_vocab):
    return [codon_vocab.get(seq_str[i:i+3], codon_vocab["UNK"])
            for i in range(0, len(seq_str)-2, 3)]

def score_promoter(window, pwm_logo):
    if len(window) != PWM_WINDOW:
        return -999.0
    return sum(pwm_logo[nuc2idx.get(c, 0), i] for i, c in enumerate(window))

def score_codons(tokens, log_emissions):
    return sum(log_emissions[token] for token in tokens)

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train_model(csv_path):
    print(f"Training ablation components from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    train_true = df[df['Label'] == True]['Sequence'].tolist()
    train_false = df[df['Label'] == False]['Sequence'].tolist()
    
    # 1. Train Upstream PWM
    pwm_counts = np.ones((4, PWM_WINDOW)) * 1e-4
    for seq in train_true:
        upstream_chunk = seq[UPSTREAM - PWM_WINDOW : UPSTREAM]
        for i, nuc in enumerate(upstream_chunk):
            if nuc in nuc2idx:
                pwm_counts[nuc2idx[nuc], i] += 1
                
    pwm = pwm_counts / pwm_counts.sum(axis=0, keepdims=True)
    bg = np.array([0.25]*4).reshape(4,1)
    pwm_logo = np.log2(pwm / bg)
    
    # 2. Train Downstream Codon Emissions (CUB)
    codon_vocab = generate_vocab()
    true_codon_counts = np.ones(65) * 1e-4
    false_codon_counts = np.ones(65) * 1e-4
    
    downstream_start = UPSTREAM + 3
    downstream_end = downstream_start + CODON_WINDOW
    
    for seq in train_true:
        cds_chunk = seq[downstream_start : downstream_end]
        tokens = encode(cds_chunk, codon_vocab)
        for t in tokens: true_codon_counts[t] += 1
            
    for seq in train_false:
        non_cds_chunk = seq[downstream_start : downstream_end]
        tokens = encode(non_cds_chunk, codon_vocab)
        for t in tokens: false_codon_counts[t] += 1
            
    true_emissions = true_codon_counts / true_codon_counts.sum()
    false_emissions = false_codon_counts / false_codon_counts.sum()
    log_emissions = np.log2(true_emissions / false_emissions)
    
    return {
        'pwm_logo': pwm_logo, 
        'codon_vocab': codon_vocab,
        'log_emissions': log_emissions
    }

# ─────────────────────────────────────────────
# ABLATION BENCHMARK (Technique B)
# ─────────────────────────────────────────────
def evaluate_comparative(csv_path, model_data):
    print(f"\n>>> Running Ablation Benchmark on {csv_path}...")
    df = pd.read_csv(csv_path)
    
    pwm_logo = model_data['pwm_logo']
    codon_vocab = model_data['codon_vocab']
    log_emissions = model_data['log_emissions']
    
    y_true = df['Label'].astype(int).values
    results = {"baseline": [], "v1_pwm": [], "v2_codon": [], "v3_hybrid": []}
    
    downstream_start = UPSTREAM + 3
    downstream_end = downstream_start + CODON_WINDOW
    
    for seq in df['Sequence']:
        # Baseline: Ribosome Scanning Model (First ATG encountered)
        first_atg_idx = seq.find("ATG")
        b_score = 1.0 if first_atg_idx == UPSTREAM else 0.0
        results["baseline"].append(b_score)
        
        # Component Scoring
        p_score = score_promoter(seq[UPSTREAM - PWM_WINDOW : UPSTREAM], pwm_logo)
        tokens = encode(seq[downstream_start : downstream_end], codon_vocab)
        c_score = score_codons(tokens, log_emissions)
        
        results["v1_pwm"].append(p_score)
        results["v2_codon"].append(c_score)
        results["v3_hybrid"].append(p_score + c_score)
        
    # Visualization setup
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Feature Ablation in Binary TIS Classification (Fixed-Window)", 
                 fontsize=18, fontweight='bold', y=0.98)
    
    colors = {'baseline': '#000000', 'v1_pwm': '#4C72B0', 'v2_codon': '#55A868', 'v3_hybrid': '#C44E52'}
    labels = {'baseline': 'Naive Scanning (First ATG)', 'v1_pwm': 'v1 (PWM Kozak Only)', 
              'v2_codon': 'v2 (Codon Bias Only)', 'v3_hybrid': 'v3 (Hybrid Model)'}
    
    metrics_list = []

    for model_name, y_scores in results.items():
        y_scores = np.array(y_scores)
        
        # ROC Data
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=colors[model_name], lw=2.5, label=f"{labels[model_name]} (AUC = {roc_auc:.3f})")
        
        # PR Data
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        ax_pr.plot(recall, precision, color=colors[model_name], lw=2.5, label=f"{labels[model_name]} (AUC = {pr_auc:.3f})")
        
        # Dynamic Threshold Optimization
        if model_name == 'baseline':
            y_pred = y_scores
            ideal_thresh = 0.5
        else:
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_idx = np.argmax(f1_scores)
            ideal_thresh = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
            y_pred = (y_scores >= ideal_thresh).astype(int)
        
        metrics_list.append({
            "Model": labels[model_name],
            "Threshold": f"{ideal_thresh:.4f}",
            "Accuracy": f"{accuracy_score(y_true, y_pred):.4f}",
            "F1-Score": f"{f1_score(y_true, y_pred, zero_division=0):.4f}",
            "MCC": f"{matthews_corrcoef(y_true, y_pred):.4f}",
            "ROC AUC": f"{roc_auc:.4f}",
            "PR AUC": f"{pr_auc:.4f}"
        })

    # Format Plots
    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax_roc.set_title("ROC Curve: Feature Discrimination", fontsize=14, fontweight='bold')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")

    ax_pr.set_title("Precision-Recall Curve", fontsize=14, fontweight='bold')
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="lower left")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    output_img = f"ablation_metrics_{os.path.basename(csv_path).split('.')[0]}.png"
    plt.savefig(output_img, dpi=300)
    plt.close()
    
    print(pd.DataFrame(metrics_list).to_string(index=False))
    print(f"\n[✔] High-res comparative plot saved to {output_img}")

# ─────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CSV-based Hybrid TIS Classifier Ablation")
    parser.add_argument('--train_csv', type=str, default='train.csv', help='CSV containing training data')
    parser.add_argument('--test_csv', type=str, default='test.csv', help='CSV containing testing data')
    parser.add_argument('--model_file', type=str, default='hybrid_model.pkl', help='Model save path')
    args = parser.parse_args()

    if os.path.exists(args.model_file):
        print(f"Loading existing ablation model from {args.model_file}...")
        with open(args.model_file, 'rb') as f:
            model_data = pickle.load(f)
    else:
        model_data = train_model(args.train_csv)
        with open(args.model_file, 'wb') as f:
            pickle.dump(model_data, f)
        print("Model generated and saved.")

    # Run the consolidated comparative evaluation on the test set
    evaluate_comparative(args.test_csv, model_data)

if __name__ == '__main__':
    main()