import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

#Load love embeddings:
with open(Path("data") / "love_embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

love_embeddings = np.array(data["embeddings"])
love_metadata = data["metadata"]

K_LOVE = 4
km = KMeans(n_clusters=K_LOVE, random_state=42, n_init=10)
km.fit(love_embeddings)
centroids = km.cluster_centers_
cluster_labels = km.labels_

#compute ambiguity
distances = np.array([
    [np.linalg.norm(emb - centroid) for centroid in centroids]
    for emb in love_embeddings
])

# Sort distances for each point: smallest first
sorted_distances = np.sort(distances, axis=1)

# Ambiguity = difference between closest and second-closest centroid
# Small value = the point is torn between two senses
ambiguity = sorted_distances[:, 1] - sorted_distances[:, 0]

# Identify the two clusters each point is torn between
closest_two = np.argsort(distances, axis=1)[:, :2]

def show_love_in_context(sentence, width = 100):
    lower = sentence.lower()
    if "love" in lower:
        pos = lower.index("love")
        start = max(0, pos - width)
        end = min(len(sentence), pos + width)
        return "..." + sentence[start:end] + "..."
    return sentence[:200]

most_ambiguous = np.argsort(ambiguity)[:15]
cluster_names = {0: "Cluster 0", 1: "Cluster 1", 2: "Cluster 2", 3: "Cluster 3"}

print("Most ambiguous uses of 'love' (k=4)")
for rank, idx in enumerate(most_ambiguous, 1):
    meta = love_metadata[idx]
    assigned = cluster_labels[idx]
    c1, c2 = closest_two[idx]
    d1, d2 = distances[idx][c1], distances[idx][c2]

    print(f"\n{rank}. Assigned to Cluster {assigned} ({cluster_names.get(assigned, '?')})")
    print(f"   Torn between: Cluster {c1} ({cluster_names.get(c1, '?')}) "
        f"and Cluster {c2} ({cluster_names.get(c2, '?')})")
    print(f"   Distances: {d1:.4f} vs {d2:.4f}  (gap: {ambiguity[idx]:.4f})")
    print(f"   [{meta['Song Title']}]")
    print(f"   {show_love_in_context(meta['Context'])}")

print("Borderline cases by cluster pair")
pairs = [
    (0,1),
    (0,2),
    (0,3),
    (1,2),
    (1,3),
    (2,3)
]
pair_labels = [
    ("Cluster 0", "Cluster 1"),
    ("Cluster 0", "Cluster 2"),
    ("Cluster 0", "Cluster 3"),
    ("Cluster 1", "Cluster 2"),
    ("Cluster 1", "Cluster 3"),
    ("Cluster 2", "Cluster 3")
]

for(ca, cb), (name_a, name_b) in zip(pairs, pair_labels):
    print(f"\n{'='*60}")
    print(f"BETWEEN Cluster {ca} ({name_a}) AND Cluster {cb} ({name_b})")
    print(f"{'='*60}")
    # Find points whose two closest clusters are this pair
    pair_mask = np.array([
        set(closest_two[i]) == {ca, cb} for i in range(len(love_embeddings))
    ])

    if pair_mask.sum() == 0:
        print("  No borderline cases between these clusters.\n")
        continue

    # Among those, find the most ambiguous
    pair_indices = np.where(pair_mask)[0]
    pair_ambiguity = ambiguity[pair_indices]
    most_ambig_in_pair = pair_indices[np.argsort(pair_ambiguity)[:5]]

    for idx in most_ambig_in_pair:
        meta = love_metadata[idx]
        assigned = cluster_labels[idx]
        print(f"\n  [Assigned: Cluster {assigned}] Gap: {ambiguity[idx]:.4f}")
        print(f"  [{meta['Song Title']}]")
        print(f"  {show_love_in_context(meta['Context'])}")


# MOST CLEAR-CUT: firmly in one sense

print("\n\n" + "=" * 80)
print("MOST CLEAR-CUT USES (for contrast)")
print("=" * 80)

most_clear = np.argsort(ambiguity)[-5:][::-1]

for rank, idx in enumerate(most_clear, 1):
    meta = love_metadata[idx]
    assigned = cluster_labels[idx]

    print(f"\n{rank}. Cluster {assigned} ({cluster_names.get(assigned, '?')})  "
          f"Gap: {ambiguity[idx]:.4f}")
    print(f"   [{meta['Song Title']}]")
    print(f"   {show_love_in_context(meta['Context'])}")