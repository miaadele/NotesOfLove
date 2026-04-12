#use bert-base-uncased model to embed the context windows around the word 'love
#since each set of text is a maximum of 25 words in length, chunking is not necessary

import json
import re
from pathlib import Path
import torch
import numpy as np
from transformers import BertModel, BertTokenizer

#configuration
cwd = Path.cwd()
parent = cwd.parent
DATA_PATH = parent/"love_contexts.json" 
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "cw_embeddings_bertbase.json"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Embed a context window
def embed_document(text, tokenizer, model, max_length=512):
    inputs = tokenizer(
        text, 
        return_tensors = "pt",
        truncation = True,
        padding = True, 
        max_length = max_length
    ).to(device) #Turns raw text into GPU-ready tensors that BERT understands
   
    with torch.no_grad():
        outputs = model(**inputs)
    
    last_hidden = outputs.last_hidden_state

    #mean pooling
    attn_mask = inputs["attention_mask"]
    mask = attn_mask.unsqueeze(-1).expand(last_hidden.size()).float()

    summed = torch.sum(last_hidden * mask, dim=1)
    counts = torch.clamp(mask.sum(dim = 1), min = 1e-9)

    document_embedding = (summed/counts).squeeze().cpu().numpy()
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
            "Lyrics": entry["Context"],
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