# Step 1: Concept and Data preparation

import json
import re

with open("cleaned_lyrics.json", "r", encoding="utf-8") as f:
    data = json.load(f)

love_occurrences = []
term_ct = 0

#Regex to capture words with punctuation attached
word_pattern = re.compile(r"\S+")

for entry in data:
    lyrics = entry.get("Lyrics", "")
    words = word_pattern.findall(lyrics)

    for i, word in enumerate(words):
        #normalize word
        clean_word = re.sub(r"[^\w]", "", word).lower()
        if clean_word == "love":
            term_ct += 1
            window_start = max(0, i - 12)
            window_end = min(len(words), i + 13)
            window_unit = words[window_start:window_end]

            love_occurrences.append({
                "Artist": entry.get("Artist"),
                "Context": " ".join(window_unit),
                "Rank": entry.get("Rank"),
                "Song Title": entry.get("Song Title"),
                "Writers": entry.get("Writers"),
                "Year": entry.get("Year")
            })

with open ("love_contexts.json", "w", encoding="utf-8") as f:
    json.dump(love_occurrences, f, indent=2)

print(term_ct)