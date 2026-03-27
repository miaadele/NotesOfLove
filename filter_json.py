import json
import ast

#I want to keep select keys from the entries in the original dataset

SELECT_KEYS = [
    "Artist",
    "Lyrics",
    "Rank",
    "Song Title",
    "Writers",
    "Year"
]
# In the original dataset, Writers is a string that looks like a list of dictionaries. 
# The string must be converted into a real Python object in order to extract the names.

#load original dataset
with open("all_songs_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

#filter entries
#transform "Writers" and copy the rest normally 
filtered_data = []

for entry in data: 
    new_entry = {}
    
    for k in SELECT_KEYS:
        if k == "Writers":
            writers_raw = entry.get("Writers")

            try:
                writers_list = ast.literal_eval(writers_raw) if writers_raw else []
                new_entry["Writers"] = [
                    w.get("name") for w in writers_list if isinstance(w, dict)
                ]
            except (ValueError, SyntaxError):
                new_entry["Writers"] = []

        else:
            new_entry[k] = entry.get(k)
    filtered_data.append(new_entry)

#save new json
with open("filtered.json", "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, indent = 2, ensure_ascii = False)