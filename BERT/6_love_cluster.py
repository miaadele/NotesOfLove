import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["figure.dpi"] = 150

#Load love embeddings:
with open(Path("data") / "love_embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

love_embeddings = np.array(data["embeddings"])
love_metadata = data["metadata"]

print(f"Loaded {len(love_embeddings)} 'love' embeddings.")
print(f"Embedding shape: {love_embeddings.shape}")

# Find optimal k using Slihouette score
if len(love_embeddings) >= 10:
    k_range = range(2, min(8, len(love_embeddings) // 2))
    sil_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(love_embeddings)
        sil_scores.append(silhouette_score(love_embeddings, labels))
        print(f"  k={k}: silhouette={sil_scores[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), sil_scores, "ro-")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Optimal Clusters for 'Love' Embeddings")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("love_cluster_selection.png", dpi=150, bbox_inches="tight")
    plt.show()  
    
    best_k = list(k_range)[np.argmax(sil_scores)]
    print(f"\nBest k by silhouette: {best_k}")
else:
    best_k = 2
    print(f"Too few occurrences for silhouette analysis. Using k={best_k}.")

# Cluster
K_LOVE = best_k
print(f"the best k score is {K_LOVE}")
km_love = KMeans(n_clusters=K_LOVE, random_state=42, n_init=10)
love_cluster_labels = km_love.fit_predict(love_embeddings)

print(f"\nLove clusters (k={K_LOVE}):")
for c in range(K_LOVE):
    count = (love_cluster_labels == c).sum()
    print(f"  Cluster {c}: {count} occurrences")

# PCA visualization. 
pca_love = PCA(n_components=2)
love_2d = pca_love.fit_transform(love_embeddings)

colors = ["#b50b60", "#ff76bb", "#ff0015", "#ac59cd"]

fig, ax = plt.subplots(figsize=(10, 8))

for c in range(K_LOVE):
    mask = love_cluster_labels == c
    ax.scatter(
        love_2d[mask, 0], love_2d[mask, 1],
        c=colors[c], s=50, alpha=0.6, label=f"Cluster {c}",
        edgecolors="black", linewidths=0.3,
    )

ax.set_xlabel(f"PC1 ({pca_love.explained_variance_ratio_[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({pca_love.explained_variance_ratio_[1]:.1%} variance)")
ax.set_title("Contextual Embeddings of 'Love'")
ax.legend()
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("love_embeddings_pca.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSaved plots to love_cluster_selection.png and love_embeddings_pca.png")

# --- Save cluster labels for later steps ---
output = {
    "love_cluster_labels": love_cluster_labels.tolist(),
    "K_LOVE": K_LOVE,
}
with open(Path("data") / "love_cluster_labels.json", "w") as f:
    json.dump(output, f, indent=2)

print("Saved cluster labels to data/love_cluster_labels.json")