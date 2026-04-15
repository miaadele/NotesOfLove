#multilabel classifier
#train one classifier per label

from pathlib import Path
import nltk
from gensim.utils import simple_preprocess
import json

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

#CONTEXT 1: romantic love
ROM_A = {
    "lover", "loving", 
    "kiss", "kissing",
    "girl","baby", "honey",
    "sex", "heart"
}
ROM_B = {
    "madly", "deeply",
    "sweetness", "sweetly", "affection",
    "satisfy", "pleasing",
    "attracted", "dearest",
    "cuddle", "huggin",
    "desires", "temptation", 
    "wife", "husband",
    "lips", "bed", "mine", "hold"
}

#CONTEXT 2: longing and uncertainty
LONGING_A = {
    "bittersweet", "forgiving", "miss", "missing",
    "desired", "yearning", "grieve", "pleading",
    "fragile", "carefully", "helplessly"
}
LONGING_B = {
    "dies", "unreal",
    "disaster", 
    "deepest", "haunt",
    "deceived", "discouraged",
    "tragedy", "ashamed", "fail",
    "apologies"
}

#CONTEXT 3: time-based love
TIME_A = {
    "time",
    "tonight", "night",
    "forever", "always", "eternity", "eternally", "indefinitely", "endlessly",
    "lifetime", "lifetimes", "today","tomorrow",
    "year", "years", "month", "months", "day", "days", "decade", "decades"
}
TIME_B = {
    "life",
    "chapter",
    "vow", "marriage",
    "lasts", "devotion", "keeping"
}
#CONTEXT 4: hopeful
HOPE_A = {
    "conquer", "guide", "achieve",
    "blessings", "miracles", "miracle",
    "believes", "belongs",
    "fulfill", "heals"
}
HOPE_B = {
    "protection", "courage", "remedy",
    "sincere",
    "repair", "soothe", 
    "cheerleader",
    "sacred", "sunrise",
    "soar"
}

#load and process texts
cwd = Path.cwd()
MODEL_PATH = cwd/"models"
parent = cwd.parent
DATA_PATH = parent/"cleaned_lyrics.json" 
TARGET_WORDS = 120
MIN_WORDS = 5
MAX_WORDS = 200

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
    songs = []
    for entry in data:
        songs.append({
            "Lyrics": entry["Lyrics"],
            "Song": entry["Song Title"],
            "Year": entry["Year"]
        })

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

rows = []

def chunk_text(text, target_words=120):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current = []
    current_len = 0
    for sent in sentences:
        words = sent.split()
        if not words:
            continue
        if current_len + len(words) > target_words and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(sent)
        current_len += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks


#convert songs into a dataframe
for song in songs:
    chunks = chunk_text(song["Lyrics"], target_words=TARGET_WORDS)
    for i, chunk in enumerate(chunks):
        rows.append({
            "Song": song["Song"],
            "Text": chunk, #use chunk, not full lyrics
            "Year": song["Year"],
            "Chunk_ID": i,
            "Label": None
        })
df = pd.DataFrame(rows)
print("dataframe is made")

import re
#basic tokenization fn
def tokenize(text):
    return set(re.findall(r"\b\w+\b", text.lower()))

#labeling function:
    #give A words higher weight than B
    #allow multiple matches
    #pick the strongest category/context
def assign_multilabel(text):
    tokens = tokenize(text)
    
    return {
        "romance": int(bool(tokens & ROM_A or tokens & ROM_B)),
        "longing": int(bool(tokens & LONGING_A or tokens & LONGING_B)),
        "time": int(bool(tokens & TIME_A or tokens & TIME_B)),
        "hope": int(bool(tokens & HOPE_A or tokens & HOPE_B)),
    }

#apply labels to df
#each row has a label vector
labels = df["Text"].apply(assign_multilabel)
label_df = pd.DataFrame(list(labels))
df = pd.concat([df, label_df], axis = 1)

#check results and inspect examples
print(df["Label"].value_counts())
df[df["Label"] == "romance"]["Text"].head(10)
#RESULTS


print("Labels have been assigned")

#transform the text data into TF-IDF representation so the Scikit-Learn model can accept the training data
x = df["Text"] #select the lyrics column from the df. x is a pandas Series of the lyrics
y = df[["romance", "longing", "time", "hope"]]
vectorizer = TfidfVectorizer(max_df = 0.9) #ignore words that appear in more than 90% of documents
vectorizer.fit(x) #learn the vocab and IDF weights from the lyrics
print("dataset prepared")

#sanity checks
# print("songs:", len(songs))
# print("rows:", len(rows))
# print("df shape:", df.shape)
# print("Columns are:")
# print(list(df.columns))
# print("first 10 rows:")
# print(df.head(10))
# print("df head ran")

#split the dataset into training and test datasets
#80% training, 20% testing split. Same split every run
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 101)
print("data has been split into test and train sets")

#start training with the MultiOutputClassifier object
#train one classifier per label
x_train_tfidf = vectorizer.transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

#use the model to predict the test data
clf = MultiOutputClassifier(LogisticRegression()).fit(x_train_tfidf, y_train)
prediction = clf.predict(x_test_tfidf)


#evaluate the Multilabel Classifier model with accuracy metrics
from sklearn.metrics import accuracy_score
print("Accuracy Socre: ", round(accuracy_score(y_test, prediction),3))
#accuracy score: 0.6947928544279741
#the model predicts the exact label combination less less than 69.5% of the time

#evaluate the label prediction rather than label combination
#Hamming Loss evaluation metric:
    #loss function; the lower the score, the better:
    #0 indicates no wrong predictions
from sklearn.metrics import hamming_loss
print('Hamming Loss: ', round(hamming_loss(y_test, prediction),2))
#Hamming Loss: 0.09
#this model has a wrong prediction 9% of the time independently
#i.e. each label prediction might be wrong 9% of the time

#transform lyric chunks using the same vectorizer
x_all_tfidf = vectorizer.transform(df["Text"])
df_predictions = clf.predict(x_all_tfidf) #predict for all data
predictions_df = pd.DataFrame(df_predictions, columns=["romance", "longing", "time", "hope"]) #convert predictions into a df
df_predictions = pd.concat([df, predictions_df.add_prefix("pred_")], axis = 1)

df_predictions.to_csv("predictions.csv", index = False)

#save the model
import joblib
joblib.dump(clf, MODEL_PATH / "multilabel_model.joblib")
joblib.dump(vectorizer, MODEL_PATH / "tfidf_vectorizer.joblib")
print("Saved vectorizer and classifier")

#confusion matrix
from sklearn.metrics import multilabel_confusion_matrix
# label_names = ["romance", "longing", "time", "hope"]
# mcm = multilabel_confusion_matrix(y_test, prediction)
# for i, label in enumerate(label_names):
#     tn, fp, fn, tp = mcm[i].ravel()
#     print(f"{label}")
#     print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}\n")

#Cooccurrence matrix
#Ground truth co-occurence (rules)
import numpy as np
y_true = y.values
co_true = np.dot(y_true.T, y_true)

#predicted cooccurrence (model behavior)
y_pred_all = df_predictions[[
    "pred_romance", "pred_longing", "pred_time", "pred_hope"
]].values

co_pred = np.dot(y_pred_all.T, y_pred_all)

#rule-based v model-based cooccurrence
import seaborn as sns
import matplotlib.pyplot as plt

labels = ["romance", "longing", "time", "hope"]

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.heatmap(co_true, annot=True, fmt="d",
            xticklabels=labels, yticklabels=labels)
plt.title("Rule-based Co-occurrence")

plt.subplot(1,2,2)
sns.heatmap(co_pred, annot=True, fmt="d",
            xticklabels=labels, yticklabels=labels)
plt.title("Model Co-occurrence")

plt.show()

#error matrix / label confusion
error_matrix = np.dot(y_pred_all.T, (y_true - y_pred_all > 0))