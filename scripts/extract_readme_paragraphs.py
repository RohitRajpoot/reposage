import os
import csv

# 1. Point this at the folder containing your .md files
ROOT_DIR = "."   # or "." if your README.md is at repo root
OUT_CSV  = "data/classification.csv"

# 2. Collect all paragraphs
rows = []
for dirpath, _, filenames in os.walk(ROOT_DIR):
    for fn in filenames:
        if not fn.lower().endswith(".md"):
            continue
        path = os.path.join(dirpath, fn)
        text = open(path, encoding="utf-8").read()
        # split on blank lines
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        for para in paras:
            rows.append({"text": para, "category": ""})

# 3. Write out a CSV with an empty category column
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["text","category"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"Extracted {len(rows)} paragraphs → {OUT_CSV}")
