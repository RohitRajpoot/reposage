# scripts/balance_classes.py

import os
import pandas as pd

# 1. Load your fully labeled CSV
in_path  = "data/classification_labeled.csv"
out_path = "data/classification_balanced.csv"
df = pd.read_csv(in_path)

# 2. Merge small classes into 'reference'
df['category'] = df['category'].replace({
    'configuration': 'reference',
    'api_reference': 'reference'
})

# 3. Compute how many to sample for each class
counts    = df['category'].value_counts()
max_count = counts.max()

# 4. Oversample each class to match the largest
frames = []
for cat, cnt in counts.items():
    subset = df[df['category'] == cat]
    if cnt < max_count:
        extra = subset.sample(max_count - cnt, replace=True, random_state=42)
        subset = pd.concat([subset, extra])
    frames.append(subset)

balanced = pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True)

# 5. Save and report
os.makedirs(os.path.dirname(out_path), exist_ok=True)
balanced.to_csv(out_path, index=False)

print("New class distribution:")
print(balanced['category'].value_counts())
print(f"\nBalanced dataset saved to {out_path}")
