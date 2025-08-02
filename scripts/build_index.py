#!/usr/bin/env python3
# scripts/build_index.py

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# 1. Gather all paragraphs from Markdown files in this repo
texts = []
meta  = []
for root, _, files in os.walk("."):
    # Skip virtualenv, data, and scripts folders
    if any(skip in root for skip in ["./.venv", "./data", "./scripts"]):
        continue
    for fname in files:
        if not fname.lower().endswith(".md"):
            continue
        path = os.path.join(root, fname)
        content = open(path, encoding="utf-8").read()
        for para in content.split("\n\n"):
            p = para.strip()
            if p:
                texts.append(p)
                meta.append(path)
print(f"Found {len(texts)} paragraphs across your .md files.")

# 2. Compute embeddings with Sentence-Transformers
model = SentenceTransformer("all-MiniLM-L6-v2")
embs  = model.encode(texts, show_progress_bar=True)
embs  = np.array(embs, dtype="float32")

# 3. Normalize & build FAISS index (inner-product)
faiss.normalize_L2(embs)
index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)

# 4. Save index and metadata
os.makedirs("data", exist_ok=True)
faiss.write_index(index, "data/deepseek.index")
with open("data/deepseek_meta.pkl", "wb") as f:
    pickle.dump(meta, f)
with open("data/deepseek_texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print(f"Built FAISS index with {len(texts)} entries.")
