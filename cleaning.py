import json
import re

def normalize_lyrics(text):
    if not text:
        return ""
    
    #remove bracketed sections like [Chorus 1], [Verse 2]
    text = re.sub(r"\[.*?\]", "", text)

    #remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    return text

#load filtered data
with open("filtered.json", "r", encoding="utf-8") as f:
    data = json.load(f)

#apply normalization
for entry in data:
    entry["Lyrics"] = normalize_lyrics(entry.get("Lyrics"))

#save new file
with open("normalized.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent = 2, ensure_ascii= False)