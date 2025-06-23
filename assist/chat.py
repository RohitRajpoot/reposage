# assist/chat.py

import json
import torch

# module-level placeholders
_WEIGHTS: torch.Tensor | None = None
_TOKEN2IDX: dict[str,int]    = {}
_IDX2TOKEN: dict[int,str]    = {}

def chat(question: str) -> str:
    """
    Embedding‐based Q&A stub that:
      1) lazy‐loads tensor.pt into a torch.Tensor,
      2) lazy‐loads vocab.json,
      3) tokenizes question by whitespace,
      4) averages the embeddings,
      5) finds the nearest vocab token via cosine similarity.
    """
    global _WEIGHTS, _TOKEN2IDX, _IDX2TOKEN

    # --- 1) lazy‐load model assets ---
    if _WEIGHTS is None:
        # load the embedding tensor (V × D)
        _WEIGHTS = torch.load("tensor.pt")
        # load token→index map
        with open("vocab.json", "r") as f:
            _TOKEN2IDX = json.load(f)
        # build reverse map
        _IDX2TOKEN = {int(i): w for w, i in _TOKEN2IDX.items()}

    # --- 2) tokenize & map to indices ---
    tokens = question.lower().split()
    idxs   = [ _TOKEN2IDX[t] for t in tokens if t in _TOKEN2IDX ]
    if not idxs:
        return "🤔 I don't recognize any of those words."

    # --- 3) compute average embedding (shape: D) ---
    q_embed = _WEIGHTS[idxs].mean(dim=0)  # torch.Tensor of shape (D,)

    # --- 4) compute cosine similarities against all V embeddings ---
    W     = _WEIGHTS                        # (V, D)
    dots  = W @ q_embed                     # (V,)
    W_norm = W.norm(dim=1)                  # (V,)
    q_norm = q_embed.norm()                 # scalar
    sims   = dots / (W_norm * q_norm + 1e-8)

    # --- 5) pick highest index & map back to word ---
    best_idx  = int(torch.argmax(sims))
    best_word = _IDX2TOKEN.get(best_idx, "<unknown>")

    return f"🗣️ Nearest concept: **{best_word}**"
