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

import matplotlib.pyplot as plt
plt.figure(figsize= (10, 6))
plt.plot(years, counts, marker = 'o')

plt.title("Occurrences of the Word 'Love' in US Top 100 Song Lyrics From 1959 Through 2023")
plt.savefig("love_occurerences_dist.png")