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
cluster_labels = km.labels_

# print(type(love_metadata))
# print(len(love_metadata))
# print(type(love_metadata[0]))
# print(love_metadata[0])

#filter for songs with Cluster 0 label
cluster_0_idx = np.where(cluster_labels == 3)[0]
years = [
    int(love_metadata[i]["Year"])
    for i in cluster_0_idx
    if love_metadata[i]["Year"] is not None
]

from collections import Counter
year_counts = Counter(years)
sorted_years = sorted(year_counts.keys())
counts = [year_counts[y] for y in sorted_years]

#plot by year
import matplotlib.pyplot as plt

from statsmodels.nonparametric.smoothers_lowess import lowess
smoothed = lowess(counts, sorted_years, frac = 0.2)
plt.scatter(sorted_years, counts, color = "#ac59cd")
plt.xlabel("Year")
plt.ylabel("Number of Songs")
plt.title("Songs in Cluster 3, by Year")
plt.plot(smoothed[:, 0], smoothed[:, 1], linestyle='--', color = "#B81456")
plt.show()