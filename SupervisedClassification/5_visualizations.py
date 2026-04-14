import pandas as pd
df = pd.read_csv("predictions.csv")

print(list(df.columns))
# ['Song', 'Text', 'Year', 'Chunk_ID', 'Label', 
# 'romance', 'longing', 'time', 'hope', 
# 'pred_romance', 'pred_longing', 'pred_time', 'pred_hope']

import matplotlib.pyplot as plt

#year trend: how themes change over time, aggregated by year
year_theme = df.groupby("Year")[["romance", "longing", "time", "hope"]].mean()
year_theme.plot(
    color = ["#ce2b29", "#2c91d4", "#9763cb", "#ff209b"], 
    linewidth = 2
)
plt.title("Concepts of 'Love' per Song")
plt.ylabel("Number of Chunks Per Song with This Label")
plt.xlabel("Year")
plt.show()

#songs with the most 'love,' normalized by song length
df["Song_Label"] = df["Song"] + " (" + df["Year"].astype(str) + ")"

top_love = (
    df.groupby("Song_Label")[["romance","longing","time","hope"]].mean()
)

top_love["love_score"] = (
    top_love["romance"] +
    top_love["longing"] +
    top_love["time"] +
    top_love["hope"]
)

top_love = top_love.sort_values("love_score", ascending=False)
print(top_love.head(20))