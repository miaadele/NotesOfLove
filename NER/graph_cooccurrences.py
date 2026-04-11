import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

with open("time_date_cooccurrences.json", "r", encoding = "utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

#Split df into DATE and TIME
df_date = df[df["entity_label"] == "DATE"]
df_time = df[df["entity_label"] == "TIME"]

#Count per year
date_cts = df_date["Year"].value_counts().sort_index()
time_cts = df_time["Year"].value_counts().sort_index()

#Align years
all_yrs = sorted(set(date_cts.index).union(set(time_cts.index)))

date_y = np.array([date_cts.get(y,0) for y in all_yrs])
time_y = np.array([time_cts.get(y,0) for y in all_yrs])
years = np.array(all_yrs)

#LOWESS smoothing
date_smooth = lowess(date_y, years, frac = 0.1)
time_smooth = lowess(time_y, years, frac = 0.1)

plt.figure()

#Raw points
plt.scatter(years, date_y, label = "DATE (raw)", alpha = 0.6, color = "#B81456")
plt.scatter(years, time_y, label = "TIME (raw)", alpha = 0.6, color = "#FF99C2")

#LOWESS smoothed lines
plt.plot(date_smooth[:, 0], date_smooth[:, 1], linestyle='--', label="DATE Trend", color = "#B81456")
plt.plot(time_smooth[:, 0], time_smooth[:, 1], linestyle='--', label="TIME Trend", color = "#FF99C2")

plt.xlabel("Year that Song is On Top 100 List")
plt.ylabel("Count of DATE/TIME entities")
plt.title("DATE/TIME Entities Over Time, Coocurring with the Word 'Love'")
plt.legend()
plt.show() 
