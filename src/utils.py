import os

def ensure_dirs(*paths): # Creates directories if they don't exist.
    for p in paths:
        os.makedirs(p, exist_ok=True)

def save_csv(df, path): # Saves a DataFrame to CSV.
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows into {path}")
