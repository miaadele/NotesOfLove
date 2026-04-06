import json
import re

with open("cleaned_lyrics.json", "r", encoding="utf-8") as f:
    data = json.load(f)

segments = []

for entry in data:
    lyrics = entry.get("Lyrics", "")
    tokens = lyrics.split() #tokenize
    segments.append(tokens)

#Word2Vec training
from gensim.models import Word2Vec

w2v = Word2Vec(
    sentences = segments,
    vector_size = 100, #dimensionality
    window = 12,
    min_count = 10, #ignore very rare words
    workers = 4,
    sg = 1
)

# print("Vocabulary size:", len(w2v.wv.key_to_index)) #Vocab size: 9014

# nearest neighbors (cos sim)
def show_neighbors(word, topn = 10):
    if word not in w2v.wv:
        print(f"'{word}' not in vocabulary")
        return
    print(f"\nNearest neighbors for '{word}': ")
    for w, sim in w2v.wv.most_similar(word, topn = topn):
        print(f"{w:>12}  {sim:.3f}")

# for target in ["love"]:
#     show_neighbors(target, topn=10)

#Results:
# Nearest neighbors for 'love': 
#        love,  0.640
#        adorn  0.633
#     explain,  0.615
#       secure  0.606
#       behave  0.606
#      Really,  0.597
#        give,  0.588
#     leaving,  0.586
#       affect  0.586
#      rubbin'  0.580

#create a dictionary and a DTM
from gensim.corpora import Dictionary
dictionary = Dictionary(segments) #build a dictionary from the segments
#do some cleaning
dictionary.filter_extremes(
    no_below=100, #must appear in at least 100 of the 6396 segments
    no_above=0.5 #must appear in no more than 50% of segments
)

print("Vocabulary size after filtering: ", len(dictionary)) #1006

#Convert segments to BOW format
corpus = [dictionary.doc2bow(seg) for seg in segments]

#train LDA model
from gensim.models import LdaModel
lda = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=4,
    random_state=42,
    passes=10,
    alpha="auto",
    eta="auto"
)
print("LDA model trained")

for topic_id in range(lda.num_topics):
    print(f"\nTopic {topic_id}:")
    for word, weight in lda.show_topic(topic_id, topn=12):
        print(f"{word:>12}  {weight:.3f}")

# After training, inspect learned alpha values
print("Learned alpha values:", lda.alpha)

import numpy as np
import matplotlib.pyplot as plt

K = lda.num_topics
topic_mass = np.zeros(K)

# Sum topic probabilities over all segments
for bow in corpus:
    doc_topics = lda.get_document_topics(bow, minimum_probability=0)
    for k, p in doc_topics:
        topic_mass[k] += p

# Normalize to proportions (so bars sum to 1)
topic_share = topic_mass / topic_mass.sum()

# Plot
plt.figure()
plt.bar(range(K), topic_share)
plt.xticks(range(K), [f"T{k}" for k in range(K)])

plt.ylabel("Share of topic mass (across songs)")
plt.xlabel("Topic")
plt.title("LDA topic prevalence in U.S. Top 100 Songs from 1959-2023 (by song)")
plt.show()

# Print the numeric values too (useful for interpretation)
for k, s in enumerate(topic_share):
    print(f"Topic {k}: {s:.3f}")