# assist/bayes_chat.py

import json
import torch

# placeholders for lazy loading
_WEIGHTS: torch.Tensor | None = None
_TOKEN2IDX: dict[str, int]   = {}
_IDX2TOKEN: dict[int, str]   = {}

def bayes_chat(question: str) -> str:
    """
    Bayesian‐style Q&A stub:
      1) lazy‐load co‐occurrence (tensor_bayes.pt) into a torch.Tensor,
      2) lazy‐load vocab_bayes.json,
      3) tokenize question by whitespace,
      4) average the corresponding rows of the matrix,
      5) compute cosine similarity with all rows,
      6) return the token with highest similarity.
    """
    global _WEIGHTS, _TOKEN2IDX, _IDX2TOKEN

    # 1) lazy‐load data if needed
    if _WEIGHTS is None:
        # load your V×V or V×D Bayesian‐style tensor
        tensor = torch.load("tensor_bayes.pt")
        _WEIGHTS = tensor.detach().cpu()    # still a torch.Tensor
        # load vocab mapping
        with open("vocab_bayes.json", "r") as f:
            _TOKEN2IDX = json.load(f)
        _IDX2TOKEN = {int(idx): tok for tok, idx in _TOKEN2IDX.items()}

    # 2) tokenize & map to indices
    tokens = question.lower().split()
    idxs   = [ _TOKEN2IDX[t] for t in tokens if t in _TOKEN2IDX ]
    if not idxs:
        return "🤔 I don't recognize any of those words."

    # 3) average the selected rows → qv (shape: D)
    qv = _WEIGHTS[idxs].mean(dim=0)

    # 4) cosine similarity: (W @ qv) / (||W|| * ||qv||)
    W      = _WEIGHTS                   # shape (V, D)
    dots   = W @ qv                     # shape (V,)
    W_norm = W.norm(dim=1)              # shape (V,)
    q_norm = qv.norm()                  # scalar
    sims   = dots / (W_norm * q_norm + 1e-8)

    # 5) pick best index & lookup token
    best_idx  = int(torch.argmax(sims).item())
    best_tok  = _IDX2TOKEN.get(best_idx, "<unknown>")

    return f"🔬 Bayesian neighbor: **{best_tok}**"
