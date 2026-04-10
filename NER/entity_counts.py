import json
import re

with open("cleaned_lyrics.json", "r", encoding="utf-8") as f:
    data = json.load(f)

import spacy
nlp = spacy.load('en_core_web_sm')
print("spaCy model loaded")

all_entities = []
CHUNK_SIZE = 50000 #chars per chunk
LOOKAHEAD = 5000

boundary_re = re.compile(r"[.!?;:\n]")

def next_boundary(lyrics: str, start_idx: int) -> int:
    m = boundary_re.search(lyrics, pos = start_idx, endpos=min(len(lyrics), start_idx + LOOKAHEAD))
    if m:
        return m.end()
    return start_idx #no boundary found in lookahead window

for song in data:
    lyrics = song["Lyrics"]
    start = 0

    while start < len(lyrics):
        end = min(len(lyrics), start + CHUNK_SIZE)
        if end < len(lyrics):
            extended_end = next_boundary(lyrics, end)
            if extended_end > end:
                end = extended_end

        chunk = lyrics[start:end]
  
        doc = nlp(chunk)

        for ent in doc.ents:
            all_entities.append({
                "Artist": song["Artist"],
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "Song Title": song["Song Title"],
                "Year": song["Year"]
            })
        
        start = end

print("Total entities extracted:", len(all_entities))

from pathlib import Path
from collections import Counter

OUT_ENTS_RAW = Path.cwd() / "lyrics_entities_raw_with_boundaries.json"

with open(OUT_ENTS_RAW, "w", encoding="utf-8") as f:
    json.dump(all_entities, f, ensure_ascii=False, indent=2)

print("Saved raw entity mentions:", OUT_ENTS_RAW.resolve())

KEEP_LABELS = {"TIME"}

def normalize_ent(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()

counts_by_label = {lab: Counter() for lab in KEEP_LABELS}

for e in all_entities:
    lab = e["entity_label"]
    if lab in KEEP_LABELS:
        counts_by_label[lab][normalize_ent(e["entity_text"])] += 1

OUT_BASE_COUNTS = Path.cwd() / "lyrics_time_counts_base_with_boundaries.json"
counts_out = {lab: dict(c.most_common()) for lab, c in counts_by_label.items()}
with open(OUT_BASE_COUNTS, "w", encoding="utf-8") as f:
    json.dump(counts_out, f, ensure_ascii=False, indent=2)

print("Saved baseline counts:", OUT_BASE_COUNTS.resolve())