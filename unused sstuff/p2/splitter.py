import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import os

def generate_splits(input_csv="tis_dataset.csv", train_csv="train.csv", test_csv="test.csv", explicit_test_ids=None):
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    df = pd.read_csv(input_csv)
    
    # Validate the previously mentioned format is intact (150 + 3 + 150)
    required_cols = ['Source', 'Sequence', 'Label']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input CSV is corrupted. Must contain exactly: {required_cols}")

    if explicit_test_ids:
        print(f"Performing strict hold-out for test sequences: {explicit_test_ids}")
        # Match the base Accession ID within the generated Source string
        test_mask = df['Source'].apply(lambda x: any(test_id in x for test_id in explicit_test_ids))
        
        test_df = df[test_mask]
        train_df = df[~test_mask]
    else:
        print("Performing GroupShuffleSplit to prevent transcript leakage.")
        # Keeps sequences from the same Source string together if no explicit IDs are given
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(df, groups=df['Source']))
        
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

    # Shuffle training data to prevent sequential bias during batching
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print("\n=== Split Complete ===")
    print(f"Train Dataset: {len(train_df)} candidate sites -> {train_csv}")
    print(f"Test Dataset:  {len(test_df)} candidate sites -> {test_csv}")

if __name__ == "__main__":
    # Sourced from your previously defined JSON targets.
    # Hardcoding these Accession IDs guarantees they are never seen during training.
    json_targets_to_hold_out = [
        'NC_001136.10',  # Yeast (Fungi - Chr IV)
        'NC_003279.8'    # Worm (Nematode - Chr I)
    ]
    
    generate_splits(
        input_csv="tis_dataset.csv", 
        train_csv="train.csv", 
        test_csv="test.csv",
        explicit_test_ids=json_targets_to_hold_out
    )