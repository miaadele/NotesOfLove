import json

#I want to keep select keys from the entries in the original dataset

SELECT_KEYS = [
    "Artist",
    "Lyrics",
    "Rank",
    "Song Title",
    "Year"
]

#load original dataset
with open("all_songs_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

#filter entries
filtered_data = [
    {k: entry.get(k) for k in SELECT_KEYS}
    for entry in data
]

#save new json
with open("filtered.json", "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, indent = 2, ensure_ascii = False)