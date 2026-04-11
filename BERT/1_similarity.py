# Produce a BERT cos sim matrix with the context windows around the word love
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

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(lyrics)
print(f"Embedding shape: {embeddings.shape}")
print(f"Each sentence of the lyrics is now a vector of {embeddings.shape[1]} numbers.\n")

# Compute BERT-based similarity
bert_sim = cosine_similarity(embeddings)
print("BERT similarity matrix has finished computing")

# print("BERT Cosine Similarity Matrix:\n")
# print(f"{'':>30s}", end="")
# for i in range(len(lyrics)):
#     print(f"  [{i}]", end="")
# print()

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

#cos dist matrix
dist_matrix = cosine_distances(embeddings)

#2D projection to preserve pairwise distances
mds = MDS(
    n_components=2,
    dissimilarity="precomputed",
    random_state=42,
    metric=True
)
reduced = mds.fit_transform(dist_matrix)

fig, ax = plt.subplots(figsize=(10, 7))

ax.set_xlabel("MDS dimension 1")
ax.set_ylabel("MDS dimension 2")
ax.set_title("Sentence Embeddings (all-MiniLM-L6-v2) projected with MDS")

plt.tight_layout()
plt.savefig("mds_toy_corpus.png", dpi=150, bbox_inches="tight")
plt.show()
