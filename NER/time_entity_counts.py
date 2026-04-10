import json
import re
from pathlib import Path
from collections import Counter

ENT_PATH = Path.cwd() / "lyrics_entities_raw_with_boundaries.json"

with open(ENT_PATH, "r", encoding="utf-8") as f:
    all_entities = json.load(f)

#remove whitespace
def normalize_ent(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()

entity_text_counts = Counter(
    normalize_ent(e["entity_text"]) for e in all_entities
)

print("Unique entity strings:", len(entity_text_counts))
# print("Top 20 entity strings:")
# for ent_text, n in entity_text_counts.most_common(20):
#     print(f"{n:>7}  {ent_text}")

#count entity strings for TIME labels only
KEEP_LABEL = {"TIME"}
filtered = [
    e for e in all_entities
    if e["entity_label"] in KEEP_LABEL
]
print("Filtered entity mentions:", len(filtered))

counts_time ={lab: Counter() for lab in KEEP_LABEL}
for e in filtered:
    lab = e["entity_label"]
    txt = normalize_ent(e["entity_text"])
    counts_time[lab][txt] += 1

import pandas as pd
rows = []

print(f"\nTop 15 {lab}: ")
for ent_text, n in counts_time[lab].most_common(15):
    rows.append({
        "entity_text": ent_text,
        "count": n
    })
    print(f"{n:7} {ent_text}")

df = pd.DataFrame(rows)
df.to_csv("top_time_entities.csv", index = False)