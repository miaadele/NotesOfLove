from pathlib import Path
import nltk
import json
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# Load and preprocess all lyrics
cwd = Path.cwd()
parent = cwd.parent
DATA_PATH = parent/"cleaned_lyrics.json" 
TARGET_WORDS = 120
MIN_WORDS = 5
MAX_WORDS = 200

#read data
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
    songs = []
    for entry in data:
        songs.append({
            "Lyrics": entry["Lyrics"],
            "Song": entry["Song Title"],
            "Year": entry["Year"]
        })

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

token_lists = []

print("Building token lists...")

for song in songs:
    text = song["Lyrics"]

    chunks = chunk_text(text, TARGET_WORDS)

    for c in chunks:
        tokens = simple_preprocess(c, deacc=True) #lowercases, removes punctuation, removes accents, keeps only alphabetic tokens
        if MIN_WORDS <= len(tokens) <= MAX_WORDS:
            token_lists.append(tokens)

print("\nTotal tokenized chunks kept:", len(token_lists))


# Train Word2Vec 
print("\nTraining Word2Vec...")

model = Word2Vec(
    sentences=token_lists,
    vector_size=200,   # dimensionality of word vectors
    window=12,          # context window size
    min_count=5,       # ignore very rare words
    workers=4,         # adjust depending on your machine
    sg=1               # 1 = skip-gram; 0 = CBOW
)

# Save model

Path("models").mkdir(exist_ok=True)
model_path = Path("models") / "w2v_full.bin"
model.save(str(model_path))

print("\nModel saved to:", model_path)