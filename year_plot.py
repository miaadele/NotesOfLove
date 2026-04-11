# Plot the occurrences of "love" in the lyrics of songs, by year

import json
import re
from collections import Counter

with open("love_contexts.json", "r", encoding="utf-8") as f:
    data = json.load(f)

year_counts = Counter()

for entry in data:
    year = entry.get("Year")
    if year is not None: 
        year_counts[int(year)] += 1

years = sorted(year_counts.keys())
counts = [year_counts[y] for y in years]

#make scatterplot of love occurrences in corpus
import matplotlib.pyplot as plt
import numpy as np

years = np.array(years)
counts = np.array(counts)
from statsmodels.nonparametric.smoothers_lowess import lowess
smoothed = lowess(counts, years, frac = 0.2) #for each point, LOWESS looks at 20% of nearby data points to fit a small regression
plt.figure(figsize= (10, 6))
plt.scatter(years, counts, color = "#B81456")
plt.plot(smoothed[:, 0], smoothed[:, 1], linestyle='--', color = "#B81456")


plt.title("Occurrences of the Word 'Love' in US Top 100 Song Lyrics From 1959 Through 2023")
plt.xlabel("Year of Song Ranking")
plt.ylabel("Occrrences of the Word 'Love'")
plt.savefig("love_occurrences_lowess_smoothing.png")