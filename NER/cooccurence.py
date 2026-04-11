#Filter DATE and TIME entities in context windows of target word "love"

import json
import re
from pathlib import Path
from collections import Counter

#load entity mentions from context windows
ENT_PATH = Path.cwd() / "context_all_entities.json"

with open(ENT_PATH, "r", encoding="utf-8") as f:
    all_entities = json.load(f)

print("Loaded entity mentions: ", len(all_entities))

KEEP_LABELS = {"DATE", "TIME"}
filtered = [
    e for e in all_entities
    if e["entity_label"] in KEEP_LABELS
]

print("Filtered entity mentions: ", len(filtered))

#Count by label and separate counts by entity type
counts_by_label = {lab: Counter() for lab in KEEP_LABELS}
for e in filtered:
    lab = e["entity_label"]
    txt = e["entity_text"]
    context_lyrics = e["Lyrics"]
    year = e["Year"]
    song = e["Song"]
    counts_by_label[lab][txt] += 1

for lab in ["DATE", "TIME"]:
    print(f"\nTop 15 {lab}:")
    for ent_text, n in counts_by_label[lab].most_common(15):
        print(f"{n:>7} {ent_text}")

#filter entries to keep only DATE and TIME
select_ents = [
    entry for entry in all_entities
    if entry.get("entity_label") in {"DATE", "TIME"}
]

#save new json
with open("time_date_cooccurrences.json", "w", encoding="utf-8") as f:
    json.dump(select_ents, f, indent = 2, ensure_ascii = False)