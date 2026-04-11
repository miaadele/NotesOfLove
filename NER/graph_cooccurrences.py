import json
import pandas as pd
import matplotlib.pyplot as plt

with open("time_date_cooccurrences.json", "r", encoding = "utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

#count by year
year_counts = df["Year"].value_counts().sort_index()

#plot date and time
grouped = df.groupby(["Year", "entity_label"]).size().unstack(fill_value=0)

grouped.plot()

plt.xlabel("Year that Song is On Top 100 List")
plt.ylabel("Count of DATE/TIME entities")
plt.title("DATE/TIME Entities Over Time, Coocurring with the Word 'Love'")
plt.show()