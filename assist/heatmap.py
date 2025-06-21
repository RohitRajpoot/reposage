import torch
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def show_heatmap(tensor_path="tensor.pt"):
    # Load embeddings
    weights = torch.load(tensor_path).detach().numpy()
    # Compute similarity
    sim = cosine_similarity(weights)
    # Plot
    plt.imshow(sim, cmap="viridis")
    plt.colorbar()
    plt.title("Token Similarity Heatmap")
    plt.show()
