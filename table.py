import pandas as pd
import os

# List of result files generated from pwm 3.py
files = [
    'all_species_results_naive.csv',
    'all_species_results_baseline.csv',
    'all_species_results_v1.csv',
    'all_species_results_v2.csv',
    'all_species_results_v3.csv'
]

summary_data = []

for f in files:
    if os.path.exists(f):
        df = pd.read_csv(f)
        # Identify the model label
        label = df['label'].iloc[0]
        # Calculate the mean of all numeric metrics across all species
        numeric_means = df.drop(columns=['species', 'label']).mean()
        numeric_means['Model'] = label
        summary_data.append(numeric_means)

# Combine into a single Summary DataFrame
summary_df = pd.DataFrame(summary_data)

# Reorder columns for academic presentation
cols = ['Model', 'accuracy', 'precision', 'recall', 'f1', 'mcc', 'exact_rate', 'specificity', 'balanced_accuracy', 'mae']
summary_df = summary_df[cols]

# Rename columns to match IEEE/Conference standards
summary_df.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'Exact Rate', 'Specificity', 'Bal. Acc', 'MAE']

# Sort models by logical progression
model_order = {'naive': 0, 'baseline': 1, 'v1': 2, 'v2': 3, 'v3': 4}
summary_df['sort_val'] = summary_df['Model'].map(model_order)
summary_df = summary_df.sort_values('sort_val').drop(columns='sort_val')

# Export to CSV and print the table
summary_df.to_csv('bab4_summary_table.csv', index=False)
print(summary_df.round(4).to_string(index=False))