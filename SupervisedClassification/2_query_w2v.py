#load word2vec model and ask for neighbors of 'love'from pathlib import Path
from gensim.models import Word2Vec
from pathlib import Path

model_path = Path("models") / "w2v_full.bin"
model = Word2Vec.load(str(model_path))

seed = "love"

import csv
output_file = "similar_words.csv"

if seed not in model.wv:
    print(f"'{seed}' not found in the model vocabulary.")
    print("This usually means min_count is too high or the corpus is too small.")
else:
    # print(f"Top 200 words similar to '{seed}':")
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "similarity"]) #header
        for word, score in model.wv.similar_by_word(seed, topn=200):
            writer.writerow([word, score])