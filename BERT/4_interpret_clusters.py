import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


# Configuration
cwd = Path.cwd()
parent = cwd.parent
DATA_PATH = parent/"cleaned_lyrics.json" #list of dictionaries
CLUSTER_FILE = Path("data") / "cluster_assignments.json" #uses lyrics as identifiers

TOP_N_TERMS = 12
MAX_FEATURES = 3000
MIN_DF = 2
SNIPPET_LENGTH = 200
N_SAMPLE_FILES = 5
N_SAMPLE_SNIPPETS = 2

# Load cluster assignments
with open(CLUSTER_FILE, "r", encoding="utf-8") as f:
    cluster_data = json.load(f)

cluster_labels = np.array(cluster_data["cluster_labels"])
filenames = cluster_data["filenames"]
K = cluster_data["K"]

# Load documents
documents = {}
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
    for entry in data:
        title = entry["Song Title"]
        lyrics = entry["Lyrics"]
        documents[title] = lyrics

# Keep only filenames that appear in the cluster file and exist on disk
matched_filenames = [fn for fn in filenames if fn in documents]
matched_texts = [documents[fn] for fn in matched_filenames]

# Align labels to matched filenames
matched_indices = [i for i, fn in enumerate(filenames) if fn in documents]
matched_labels = cluster_labels[matched_indices]

print(f"Loaded {len(matched_filenames)} clustered documents for interpretation.\n")

# Build TF-IDF on the full corpus
# Filter out English stopwords 

vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    min_df=MIN_DF,
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(matched_texts)
feature_names = np.array(vectorizer.get_feature_names_out())

print("tfidf matrix was made")
# Helper function: 
def get_top_distinctive_terms(tfidf_matrix, labels, target_cluster, feature_names, top_n=10):

    in_cluster = labels == target_cluster
    out_cluster = labels != target_cluster
    if in_cluster.sum() == 0 or out_cluster.sum() == 0:
        return []

    mean_in = tfidf_matrix[in_cluster].mean(axis=0).A1
    mean_out = tfidf_matrix[out_cluster].mean(axis=0).A1
    diff = mean_in - mean_out  ## see the end of [2]

    top_indices = diff.argsort()[-top_n:][::-1]
    top_terms = [(feature_names[i], diff[i]) for i in top_indices if diff[i] > 0]

    return top_terms

# # Characterize each cluster
print(f"=== Cluster Characterization (k={K}) ===\n")

for cluster_id in range(K):
    cluster_mask = matched_labels == cluster_id
    cluster_files = [matched_filenames[i] for i in range(len(matched_filenames)) if cluster_mask[i]]
    cluster_texts = [matched_texts[i] for i in range(len(matched_texts)) if cluster_mask[i]]

    print("=" * 70)
    print(f"CLUSTER {cluster_id} — {len(cluster_files)} documents")
    print("=" * 70)

    # Distinctive TF-IDF terms (cluster vs. rest)
    top_terms = get_top_distinctive_terms(
        tfidf_matrix=tfidf_matrix,
        labels=matched_labels,
        target_cluster=cluster_id,
        feature_names=feature_names,
        top_n=TOP_N_TERMS
    )

    if top_terms:
        formatted_terms = ", ".join([term for term, score in top_terms])
        print(f"Top distinctive terms: {formatted_terms}")
    else:
        print("Top distinctive terms: (none found)")

    # Sample filenames
    print(f"Sample files: {cluster_files[:N_SAMPLE_FILES]}")

    # Sample snippets
    if cluster_texts:
        print("Sample snippets:")
        for snippet_text in cluster_texts[:N_SAMPLE_SNIPPETS]:
            snippet = " ".join(snippet_text.split())[:SNIPPET_LENGTH]
            print(f"  - {snippet}...")
    else:
        print("Sample snippets: (none)")

    print()