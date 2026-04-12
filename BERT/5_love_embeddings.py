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
OUTPUT_FILE = OUTPUT_DIR / "love_embeddings.json"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.to(device)
model.eval()

#load love occurrences
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"Loaded {len(data)} love occurrences\n")

#word embedding extraction fn
def get_word_embedding(sentence, target_word, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors ="pt", truncation = True, max_length=512)
    inputs = {k:v.to(device) for k, v in inputs.items()}
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())

    # Reconstruct words from subword tokens to find our target
    target_lower = target_word.lower()
    target_indices = []
    current_word = ""
    current_indices = []

    for idx, token in enumerate(tokens):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            if current_word == target_lower and current_indices:
                target_indices = current_indices
                break
            current_word = ""
            current_indices = []
            continue

        if token.startswith("##"):
            current_word += token[2:]
            current_indices.append(idx)
        else:
            if current_word == target_lower and current_indices:
                target_indices = current_indices
                break
            current_word = token
            current_indices = [idx]

    if not target_indices and current_word == target_lower and current_indices:
        target_indices = current_indices

    if not target_indices:
        return None

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.last_hidden_state[0].cpu()
    word_embedding = hidden_states[target_indices].mean(dim=0).numpy()

    return word_embedding


# Extract embeddings:
print("Extracting contextual embeddings for 'love'...\n")

love_embeddings = []
love_metadata = []

for i, occ in enumerate(data):
    #print(occ.keys())
    context = occ.get("Context", "")
    if "love" in context.lower():
        emb = get_word_embedding(context, "love", tokenizer, model)

    if emb is not None:
        love_embeddings.append(emb.tolist())
        love_metadata.append(occ)

    if (i + 1) % 25 == 0:
        print(f"  Processed {i + 1}/{len(data)}...")

print(f"\nExtracted {len(love_embeddings)} embeddings for 'love'.")

# Save
output = {
    "embeddings": love_embeddings,
    "metadata": love_metadata,
}
with open(Path("data") / "love_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False)

print("Saved to data/love_embeddings.json")