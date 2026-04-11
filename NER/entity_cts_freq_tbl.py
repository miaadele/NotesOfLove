import json
import pandas as pd

with open("lyrics_entities_raw_with_boundaries.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)


# Frequency by label
label_freq = df["entity_label"].value_counts()

print(label_freq)
label_freq.to_csv("label_freq.csv")

with open("context_all_entities.json", "r", encoding="utf-8") as f:
    context_data = json.load(f)

df2 = pd.DataFrame(context_data)


# Frequency by label
label_freq_context = df2["entity_label"].value_counts()

print(label_freq_context)
label_freq_context.to_csv("label_freq_context.csv")
