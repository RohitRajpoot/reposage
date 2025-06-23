# assist/transformer_demo.py

import json
import torch
import torch.nn as nn

# placeholders for lazy loading
_WEIGHTS: torch.Tensor | None     = None
_TOKEN2IDX: dict[str,int] | None  = None
_IDX2TOKEN: dict[int,str] | None  = None
_BLOCK: nn.Module | None          = None

class SingleTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff    = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

def transformer_next(prompt: str) -> str:
    """
    Given a prompt string:
      1. lazy-load tensor.pt (V×D) and vocab.json,
      2. tokenize by whitespace → indices,
      3. embed → run through one transformer block,
      4. use last position’s output to pick the nearest vocab token.
    """
    global _WEIGHTS, _TOKEN2IDX, _IDX2TOKEN, _BLOCK

    # 1) Lazy-load assets
    if _WEIGHTS is None:
        _WEIGHTS = torch.load("tensor.pt").detach().cpu()  # shape: V×D
        with open("vocab.json", "r") as f:
            _TOKEN2IDX = json.load(f)
        _IDX2TOKEN = {int(i): w for w, i in _TOKEN2IDX.items()}
        # build the transformer block
        _BLOCK = SingleTransformerBlock(embed_dim=_WEIGHTS.size(1), num_heads=2)

    # 2) Tokenize + map to indices
    tokens = prompt.lower().split()
    idxs   = [ _TOKEN2IDX[t] for t in tokens if t in _TOKEN2IDX ]
    if not idxs:
        return "🤔 No known tokens to predict from."

    # 3) Prepare input (1 × seq_len × D)
    x = _WEIGHTS[idxs].unsqueeze(0)

    # 4) Forward pass through one transformer block
    out = _BLOCK(x)  # shape: 1 × seq_len × D

    # 5) Take the last position’s vector → 1×D
    last = out[0, -1].unsqueeze(0)

    # 6) Cosine similarity with all embeddings
    sims   = nn.functional.cosine_similarity(last, _WEIGHTS)
    best_i = int(torch.argmax(sims))

    # 7) Return the predicted next token
    return f"🔮 Next‐token prediction: **{_IDX2TOKEN[best_i]}**"
