import pandas as pd
from Bio import SeqIO, Entrez
import random
import re
import json
from Bio.Seq import Seq

# Update with your actual email
Entrez.email = "example@example.com"

def generate_tis_dataset_from_json(json_path, upstream=150, downstream=150, max_negatives=10, output_csv="tis_dataset.csv"):
    # Load targets from the JSON file
    with open(json_path, 'r') as f:
        targets = json.load(f)
        
    records = []
    
    for target in targets:
        name = target.get('name', 'Unknown')
        acc = target['id']
        seq_start = target['start']
        seq_stop = target['stop']
        
        print(f"Fetching {name} ({acc}) from {seq_start} to {seq_stop}...")
        
        try:
            # Using seq_start and seq_stop forces the server to do the heavy lifting
            handle = Entrez.efetch(
                db="nucleotide", 
                id=acc, 
                rettype="gbwithparts", 
                retmode="text",
                seq_start=seq_start,
                seq_stop=seq_stop
            )
            rec = SeqIO.read(handle, "genbank")
        except Exception as e:
            print(f"Failed fetching {acc}: {e}")
            continue
            
        full_seq = str(rec.seq).upper()
        seq_len = len(full_seq)
        
        # Tag the source string with coordinates so you can trace the data back
        source_label = f"{acc}_{seq_start}-{seq_stop}"
        
        for f in rec.features:
            if f.type == "CDS":
                strand = f.location.strand
                
                # Biopython automatically offsets coordinates to be relative to the sliced sequence
                start = int(f.location.start)
                end = int(f.location.end)
                
                # Forward Strand Processing
                if strand == 1:
                    tis_idx = start
                    if tis_idx - upstream >= 0 and tis_idx + 3 + downstream <= seq_len:
                        true_seq = full_seq[tis_idx - upstream : tis_idx + 3 + downstream]
                        codon = true_seq[upstream:upstream+3]
                        records.append({'Source': source_label, 'Sequence': true_seq, 'Label': True})
                        
                        # Find Negatives (same codon, different position in the same slice)
                        matches = [m.start() for m in re.finditer(codon, full_seq)]
                        valid_negatives = [
                            full_seq[m - upstream : m + 3 + downstream] for m in matches 
                            if m != tis_idx and (m - upstream >= 0) and (m + 3 + downstream <= seq_len)
                        ]
                        
                        sampled = random.sample(valid_negatives, min(max_negatives, len(valid_negatives)))
                        for neg_seq in sampled:
                            records.append({'Source': source_label, 'Sequence': neg_seq, 'Label': False})
                
                # Reverse Strand Processing
                elif strand == -1:
                    rc_full = str(Seq(full_seq).reverse_complement())
                    rc_tis_idx = seq_len - end
                    
                    if rc_tis_idx - upstream >= 0 and rc_tis_idx + 3 + downstream <= seq_len:
                        true_seq = rc_full[rc_tis_idx - upstream : rc_tis_idx + 3 + downstream]
                        codon = true_seq[upstream:upstream+3]
                        records.append({'Source': source_label, 'Sequence': true_seq, 'Label': True})
                        
                        # Find Negatives on the RC strand
                        matches = [m.start() for m in re.finditer(codon, rc_full)]
                        valid_negatives = [
                            rc_full[m - upstream : m + 3 + downstream] for m in matches 
                            if m != rc_tis_idx and (m - upstream >= 0) and (m + 3 + downstream <= seq_len)
                        ]
                        
                        sampled = random.sample(valid_negatives, min(max_negatives, len(valid_negatives)))
                        for neg_seq in sampled:
                            records.append({'Source': source_label, 'Sequence': neg_seq, 'Label': False})

    df = pd.DataFrame(records)
    
    if not df.empty:
        # Shuffle to eliminate sequential bias during model training
        df = df.sample(frac=1).reset_index(drop=True)
        df.to_csv(output_csv, index=False)
        print(f"\nDataset generated successfully: {output_csv} with {len(df)} total candidate sites.")
    else:
        print("\nFailed. No valid TIS sequences found in the provided ranges.")

if __name__ == "__main__":
    # Save your provided array as 'targets.json' in the same directory before running
    generate_tis_dataset_from_json("targets.json", upstream=150, downstream=150, max_negatives=10)