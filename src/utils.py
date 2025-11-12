import os
import re

def ensure_dirs(*paths): # Creates directories if they don't exist.
    for p in paths:
        os.makedirs(p, exist_ok=True)

def save_csv(df, path): # Saves a DataFrame to CSV.
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows into {path}")

def extract_model_name_from_filename(path: str) -> str: # Extracts model name from results filename.
    base = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"_(\d{4})_predictions$", "", base)
