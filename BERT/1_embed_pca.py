#use PCA to project the vectors into 2D

import json
import re
from pathlib import Path

cwd = Path.cwd()
parent = cwd.parent
DATA_PATH = parent/"love_contexts.json" #the word 'love' with 12 context words on either side 

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

lyrics = []
for entry in data:
    lyrics.append(entry["Context"])

print(f"Loaded {len(lyrics)} contexts")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(lyrics)
print(f"Embedding shape: {embeddings.shape}")
print(f"Each sentence of the lyrics is now a vector of {embeddings.shape[1]} numbers.\n")

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#PCA dimensionality reduction

pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

print(f"Variance explained: PC1={pca.explained_variance_ratio_[0]:.1%}, "
      f"PC2={pca.explained_variance_ratio_[1]:.1%}")

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.4,  color = "#B81456")

ax.set_xlabel("PCA dimension 1")
ax.set_ylabel("PCA dimension 2")
ax.set_title("Sentence Embeddings (all-MiniLM-L6-v2) projected with PCA")

plt.tight_layout()
plt.savefig("PCA_love_contexts.png", dpi=150, bbox_inches="tight")
plt.show()
