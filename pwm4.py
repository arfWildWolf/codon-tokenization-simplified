import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score
)
import argparse
import pickle
import os

# Configuration matching the dataset generator
UPSTREAM = 150
DOWNSTREAM = 150
PWM_WINDOW = 50     # Evaluate 50bp immediately preceding the start codon
CODON_WINDOW = 90   # Evaluate 90bp (30 codons) downstream of the start codon

nuc2idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

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

def train_model(csv_path):
    print(f"Training hybrid model from {csv_path}...")
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
    # all_seqs = "".join(df['Sequence'].tolist())
    # bg_counts = np.array([all_seqs.count(n) for n in ['A', 'C', 'G', 'T']])
    # bg = (bg_counts / bg_counts.sum()).reshape(4, 1)
    pwm_logo = np.log2(pwm / bg)
    
    # 2. Train Downstream Codon Emissions
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
            
    # Calculate Log-Odds Ratio for Emissions
    true_emissions = true_codon_counts / true_codon_counts.sum()
    false_emissions = false_codon_counts / false_codon_counts.sum()
    log_emissions = np.log2(true_emissions / false_emissions)
    
    return {
        'pwm_logo': pwm_logo, 
        'codon_vocab': codon_vocab,
        'log_emissions': log_emissions
    }
    
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

def evaluate_and_optimize(csv_path, model_data):
    print(f"Optimizing Threshold for {csv_path}...")
    df = pd.read_csv(csv_path)
    
    pwm_logo = model_data['pwm_logo']
    codon_vocab = model_data['codon_vocab']
    log_emissions = model_data['log_emissions']
    
    y_true = df['Label'].astype(int).values
    y_scores = []
    
    # Generate raw scores for every sequence
    downstream_start = UPSTREAM + 3
    downstream_end = downstream_start + CODON_WINDOW
    
    for seq in df['Sequence']:
        p_score = score_promoter(seq[UPSTREAM - PWM_WINDOW : UPSTREAM], pwm_logo)
        c_score = score_codons(encode(seq[downstream_start : downstream_end], codon_vocab), log_emissions)
        y_scores.append(p_score + c_score)
    
    y_scores = np.array(y_scores)

    # 1. Calculate Precision-Recall Curve data
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # 2. Find the Ideal Threshold (Maximizing F1-score)
    # Avoid division by zero
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    ideal_threshold = thresholds[best_idx]
    
    print("\n" + "="*30)
    print(f"IDEAL THRESHOLD: {ideal_threshold:.4f}")
    print(f"Projected F1-Score: {f1_scores[best_idx]:.4f}")
    print(f"Projected Precision: {precision[best_idx]:.4f}")
    print(f"Projected Recall:    {recall[best_idx]:.4f}")
    print("="*30 + "\n")

    # 3. Visualization with Seaborn
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Precision-Recall Curve
    sns.lineplot(x=recall, y=precision, ax=ax1, color='teal', lw=2)
    ax1.fill_between(recall, precision, alpha=0.2, color='teal')
    ax1.scatter(recall[best_idx], precision[best_idx], color='red', s=100, label=f'Ideal (T={ideal_threshold:.2f})')
    ax1.set_title("Precision-Recall Curve (Bioinformatics Focus)")
    ax1.set_xlabel("Recall (Sensitivity)")
    ax1.set_ylabel("Precision (Positive Predictive Value)")
    ax1.legend()

    # Plot 2: Threshold vs F1-Score
    # 'thresholds' has one fewer element than precision/recall
    sns.lineplot(x=thresholds, y=f1_scores[:-1], ax=ax2, color='darkorange', lw=2)
    ax2.axvline(ideal_threshold, color='red', linestyle='--', label=f'Threshold: {ideal_threshold:.2f}')
    ax2.set_title("Threshold Sweep vs F1-Score")
    ax2.set_xlabel("Decision Threshold")
    ax2.set_ylabel("F1-Score")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # 4. Final Performance at Ideal Threshold
    y_pred = (y_scores >= ideal_threshold).astype(int)
    print("FINAL CONFUSION MATRIX AT IDEAL THRESHOLD:")
    print(confusion_matrix(y_true, y_pred))

def evaluate(csv_path, model_data):
    print(f"Evaluating {csv_path}...")
    df = pd.read_csv(csv_path)
    
    pwm_logo = model_data['pwm_logo']
    codon_vocab = model_data['codon_vocab']
    log_emissions = model_data['log_emissions']
    
    y_true = df['Label'].astype(int).tolist()
    y_pred = []
    y_scores = []
    
    downstream_start = UPSTREAM + 3
    downstream_end = downstream_start + CODON_WINDOW
    
    for seq in df['Sequence']:
        # Upstream PWM Score
        upstream_chunk = seq[UPSTREAM - PWM_WINDOW : UPSTREAM]
        p_score = score_promoter(upstream_chunk, pwm_logo)
        
        # Downstream Codon Score
        cds_chunk = seq[downstream_start : downstream_end]
        tokens = encode(cds_chunk, codon_vocab)
        c_score = score_codons(tokens, log_emissions)
        
        # Combined probability representing the True state
        total_score = p_score + c_score
        y_scores.append(total_score)
        
        # Argmax evaluation: 
        # State 0 (False/Decoy) is represented by baseline 0.0
        # State 1 (True TIS) is represented by the log-odds total_score
        # prediction = np.argmax([0.0, total_score])
        custom_threshold = 3.2863
        prediction = 1 if total_score >= custom_threshold else 0
        y_pred.append(prediction)
        
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else 0.0

    print("\n=== PERFORMANCE METRICS ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    
    print("\nConfusion Matrix [TN, FP | FN, TP]:")
    print(confusion_matrix(y_true, y_pred))

def main():
    parser = argparse.ArgumentParser(description="CSV-based Hybrid TIS Classifier")
    parser.add_argument('--train_csv', type=str, default='train.csv', help='CSV containing training data')
    parser.add_argument('--test_csv', type=str, default='test.csv', help='CSV containing testing data')
    parser.add_argument('--model_file', type=str, default='hybrid_model.pkl', help='Model save path')
    args = parser.parse_args()

    if os.path.exists(args.model_file):
        print("Loading existing hybrid model...")
        with open(args.model_file, 'rb') as f:
            model_data = pickle.load(f)
    else:
        model_data = train_model(args.train_csv)
        with open(args.model_file, 'wb') as f:
            pickle.dump(model_data, f)
        print("Model generated and saved.")

    evaluate(args.test_csv, model_data)
    evaluate_and_optimize(args.train_csv, model_data)

if __name__ == '__main__':
    main()