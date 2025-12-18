import os
import pandas as pd
from src.features import calculate_snapshot_features

def get_true_path(base, folder):
    """Recursively finds data folder handling nested structures."""
    curr = os.path.join(base, folder)
    # Check simple nested
    if os.path.exists(os.path.join(curr, folder)): return os.path.join(curr, folder)
    # Check 4th_test/txt quirk
    if os.path.exists(os.path.join(curr, "4th_test", "txt")): return os.path.join(curr, "4th_test", "txt")
    if os.path.exists(os.path.join(curr, "txt")): return os.path.join(curr, "txt")
    return curr

def load_dataset(base_path):
    """Iterates through folders and returns a combined DataFrame."""
    experiments = {'Set1': '1st_test', 'Set2': '2nd_test', 'Set3': '3rd_test'}
    all_records = []

    print(f"Loading data from {base_path}")

    for set_name, folder in experiments.items():
        full_path = get_true_path(base_path, folder)
        if not os.path.exists(full_path):
            print(f"Skipping {set_name} (Path not found)")
            continue

        files = sorted([f for f in os.listdir(full_path) if not os.path.isdir(os.path.join(full_path, f))])
        
        print(f"Processing {set_name} ({len(files)} files)")
        
        for filename in files:
            try:
                # Read Raw
                df_raw = pd.read_csv(os.path.join(full_path, filename), sep='\t', header=None)
                
                # Feature Engineering
                feats = calculate_snapshot_features(df_raw, set_name)
                feats['timestamp'] = filename
                feats['dataset'] = set_name
                
                all_records.append(feats)
            except Exception:
                continue

    # Final DataFrame Construction
    df = pd.DataFrame(all_records)
    
    # Try parsing timestamp
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y.%m.%d.%H.%M.%S')
    except:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    return df