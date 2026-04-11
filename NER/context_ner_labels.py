#label entities for all words in the context windows of the word love

import spacy
import json
import re
from pathlib import Path
from collections import Counter
import pandas as pd

cwd = Path.cwd()
parent = cwd.parent
ENT_PATH = parent/"love_contexts.json"

with open(ENT_PATH, "r", encoding="utf-8") as f:
    love_with_context = json.load(f)

all_ents = []
rows = []
nlp = spacy.load("en_core_web_sm")

for love in love_with_context:
    doc = nlp(love["Context"])
    for ent in doc.ents:
        all_ents.append({
            "Song": love["Song Title"],
            "entity_text": ent.text,
            "entity_label": ent.label_,
            "Year": love["Year"],
            "Lyrics": love["Context"]
        })
#save raw entities into csv
rows = all_ents
df = pd.DataFrame(rows)
df.to_csv("context_all_entities.csv", index = False)

#save raw entities into json file
OUT_ENTS_RAW = Path.cwd() / "context_all_entities.json"
with open(OUT_ENTS_RAW, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent = 2)
