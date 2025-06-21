import json
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the Bayesian embeddings & vocab at import time
WEIGHTS = torch.load("tensor_bayes.pt").detach().numpy()   # shape: (V, V)
with open("vocab_bayes.json", "r") as f:
    TOKEN2IDX = json.load(f)
IDX2TOKEN = {int(idx): tok for tok, idx in TOKEN2IDX.items()}

def bayes_chat(question: str) -> str:
    """
    Given a user question, tokenize → average Bayesian embeddings →
    find the nearest token in the vocab → return that as the "answer."
    """
    tokens = question.lower().split()
    idxs = [TOKEN2IDX[t] for t in tokens if t in TOKEN2IDX]
    if not idxs:
        return "🤔 I don’t recognize any of those words."
    # average the rows corresponding to each token
    qv = np.mean(WEIGHTS[idxs], axis=0, keepdims=True)
    # compute similarities against every token’s vector
    sims = cosine_similarity(qv, WEIGHTS)[0]
    best_idx = int(np.argmax(sims))
    best_tok = IDX2TOKEN.get(best_idx, "<unknown>")
    return f"🔬 Bayesian neighbor: **{best_tok}**"
