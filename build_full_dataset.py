import os
import pandas as pd

# 1. Define input/output paths
CLEAN_DIR = "reposage-assets"
OUT_DIR   = "data/h2o"
OUT_FILE  = os.path.join(OUT_DIR, "dataset.csv")

# 2. Gather records
records = []
for root, _, files in os.walk(CLEAN_DIR):
    label = os.path.basename(root) or "root"
    for fname in files:
        if not fname.lower().endswith(".txt"):
            continue
        path = os.path.join(root, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        records.append({"text": text, "label": label})

# 3. Build DataFrame and write CSV
df = pd.DataFrame(records)
os.makedirs(OUT_DIR, exist_ok=True)
df.to_csv(OUT_FILE, index=False)
print(f"Built dataset with {len(df)} rows → {OUT_FILE}")
