import torch
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def get_heatmap_figure(tensor_path="tensor.pt"):
    # Load embeddings
    weights = torch.load(tensor_path).detach().numpy()
    # Compute similarity
    sim = cosine_similarity(weights)
    # Build figure
    fig, ax = plt.subplots()
    cax = ax.imshow(sim, cmap="viridis")
    fig.colorbar(cax, ax=ax)
    ax.set_title("Token Similarity Heatmap")
    return fig
get_heatmap_figure()
