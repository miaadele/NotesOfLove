#use bert-base-uncased model to embed the cleaned lyrics
#turn each doc into a single vector

import json
import re
from pathlib import Path
import torch
import numpy as np
from transformers import BertModel, BertTokenizer


MAX_LENGTH = 512          # BERT maximum input length
STRIDE = 256              # overlap step
MIN_TOKENS = 10           # ignore small leftover chunks
MIN_CHARS = 50            # ignore extremely short documents

#configuration
cwd = Path.cwd()
parent = cwd.parent
DATA_PATH = parent/"cleaned_lyrics.json" 
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "embeddings_bertbase.json"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Create overlapping token chunks
def chunk_token_ids(token_ids, max_length=512, stride=256, min_tokens=10):
    chunk_token_limit = max_length - 2
    chunks = []

    for start in range(0, len(token_ids), stride):
        chunk = token_ids[start:start + chunk_token_limit]
        if len(chunk) >= min_tokens:
            chunks.append(chunk)
    return chunks

# Embed a single chunk
def embed_chunk(chunk_token_ids, tokenizer, model):
    input_ids_list = [tokenizer.cls_token_id] + chunk_token_ids + [tokenizer.sep_token_id]

    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Shape: [1, seq_len, hidden_dim]
    last_hidden = outputs.last_hidden_state

    # Exclude [CLS] and [SEP]
    token_embeddings = last_hidden[0, 1:-1, :]

    # Mean pooling across tokens in this chunk
    chunk_embedding = token_embeddings.mean(dim=0).cpu().numpy()

    return chunk_embedding

# Embed a full document
def embed_document(text, tokenizer, model, max_length=512, stride=256, min_tokens=10):

    # Embed one long document by:
    #   1. tokenizing the full document,
    #   2. splitting into overlapping chunks,
    #   3. embedding each chunk,
    #   4. combining chunk embeddings with a length-weighted average.
   
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    if len(token_ids) == 0:
        return None

    chunks = chunk_token_ids(
        token_ids,
        max_length=max_length,
        stride=stride,
        min_tokens=min_tokens
    )

    if not chunks:
        return None

    chunk_embeddings = []
    chunk_weights = []

    for chunk in chunks:
        chunk_embedding = embed_chunk(chunk, tokenizer, model)
        chunk_embeddings.append(chunk_embedding)
        chunk_weights.append(len(chunk))  # length-weight the final average

    document_embedding = np.average(
        np.array(chunk_embeddings),
        axis=0,
        weights=np.array(chunk_weights)
    )

    return document_embedding

####
# Main pipeline. 

def main():
    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.to(device)
    model.eval()

    #read data
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    songs = []
    for entry in data:
        songs.append({
            "Lyrics": entry["Lyrics"],
            "Song": entry["Song Title"]
        })
    print("Embedding documents (this will take a while)...\n")

    song_embeddings = []
    song_names = []

    for i, doc in enumerate(songs, start=1):
        embedding = embed_document(
            doc["Lyrics"],
            tokenizer,
            model,
            max_length=MAX_LENGTH,
            stride=STRIDE,
            min_tokens=MIN_TOKENS
        )

        if embedding is not None:
            song_embeddings.append(embedding.tolist())
            song_names.append(doc["Lyrics"])

        if i % 50 == 0 or i == len(songs):
            print(f"Processed {i}/{len(songs)} documents...")

    if not song_embeddings:
        print("No embeddings were created.")
        return

    print(f"\nSuccessfully embedded {len(song_embeddings)} documents.")
    print(f"Embedding dimensionality: {len(song_embeddings[0])}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    output = {
        "songs": song_names,
        "embeddings": song_embeddings
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f)

    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()