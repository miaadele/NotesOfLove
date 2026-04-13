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
    "desired", "used to be",
    "yearning", "grieve", "pleading",
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
    "lifetime", "lifetimes"
    "today","tomorrow",
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

labeled = []

print(f"Processing {len(songs)} files...")

for song in songs:
    text = song["Lyrics"]

    chunks = chunk_text(text, TARGET_WORDS)

    for c in chunks:
        tokens = simple_preprocess(c, deacc=True)
        n_tokens = len(tokens)

        if not (MIN_WORDS <= n_tokens <= MAX_WORDS):
            continue

        token_set = set(tokens)

        if token_set & (ROM_A | ROM_B):
            label = 1   
        elif token_set & (LONGING_A | LONGING_B):
            label = 2   
        elif token_set & (TIME_A | TIME_B):
            label = 3
        elif token_set & (HOPE_A | HOPE_B):
            label = 4
        else:
            label = 0   # NEG

        labeled.append((c, label))

print("Total chunks labeled:", len(labeled))


# Save labeled data
Path("data").mkdir(exist_ok=True)

with open(Path("data") / "songs_labeled_chunks.json", "w", encoding="utf-8") as f:
    json.dump(labeled, f, ensure_ascii=False)

print("Saved labeled chunks to data/songs_labeled_chunks.json")