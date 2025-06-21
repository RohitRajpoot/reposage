import torch
import torch.nn as nn
import numpy as np
from .chat import TOKEN2IDX, IDX2TOKEN  # reuse your vocab maps
from .chat import WEIGHTS              # reuse your embedding weights

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

# Instantiate once
_EMB = torch.tensor(WEIGHTS, dtype=torch.float32)              # V×D
_block = SingleTransformerBlock(embed_dim=_EMB.size(1), num_heads=2)

def transformer_next(prompt: str) -> str:
    """
    Given a prompt, tokenize it, embed each token, run through one
    transformer block, then use the last position’s output vector
    to pick the nearest vocab token as the “next token.”
    """
    tokens = prompt.lower().split()
    idxs = [TOKEN2IDX[t] for t in tokens if t in TOKEN2IDX]
    if not idxs:
        return "🤔 No known tokens to predict from."
    # Build batch: 1×seq_len×D
    x = _EMB[idxs].unsqueeze(0)
    # Forward pass
    out = _block(x)              # 1×seq_len×D
    last = out[0, -1].unsqueeze(0)  # 1×D
    # Cosine similarity against all embeddings
    sims = nn.functional.cosine_similarity(last, _EMB)
    best = int(torch.argmax(sims))
    return f"🔮 Next‐token prediction: **{IDX2TOKEN[best]}**"
