import json
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load once at import time
WEIGHTS = torch.load("tensor.pt").numpy()   # shape: (V, D)
with open("vocab.json", "r") as f:
    TOKEN2IDX = json.load(f)
# Build reverse map: idx (as int) → token (str)
IDX2TOKEN = {int(i): w for w, i in TOKEN2IDX.items()}

def chat(question: str) -> str:
    """
    Embedding Q&A stub:
    - Tokenize by whitespace
    - Lookup embeddings
    - Average them
    - Find nearest token in vocab
    """
    # Simple whitespace tokenizer; you can improve this later
    tokens = question.lower().split()
    # Map to indices, drop unknowns
    idxs = [TOKEN2IDX[t] for t in tokens if t in TOKEN2IDX]
    if not idxs:
        return "🤔 I don't recognize any of those words."
    # Average embedding vector
    q_embed = np.mean(WEIGHTS[idxs], axis=0, keepdims=True)
    # Cosine‐similarity against all vocab embeddings
    sims = cosine_similarity(q_embed, WEIGHTS)[0]
    best = int(np.argmax(sims))
    best_word = IDX2TOKEN.get(best, "<unknown>")
    return f"🗣️ Nearest concept: **{best_word}**"
