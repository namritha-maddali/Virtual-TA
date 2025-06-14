import os
import json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# === CONFIG ===
DISCOURSE_JSON_PATH = "data/raw/discourse_meta_data.json"
TDS_SITE_TEXT_PATH = "data/raw/tds_site_text.txt"
CHUNK_SIZE = 200  # words per chunk
FAISS_DIR = "data/faiss"
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "tds_index.index")
TEXTS_PATH = os.path.join(FAISS_DIR, "tds_texts.npy")

# === LOAD & CLEAN DATA ===

def clean_text(text):
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    return text.strip()

def load_discourse_posts():
    if not os.path.exists(DISCOURSE_JSON_PATH):
        return []
    with open(DISCOURSE_JSON_PATH, "r", encoding="utf-8") as f:
        posts = json.load(f)
    return [clean_text(post["content"]) for post in posts if post.get("content")]

def load_tds_site_text():
    if not os.path.exists(TDS_SITE_TEXT_PATH):
        return []
    with open(TDS_SITE_TEXT_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    return [clean_text(t) for t in text.split("\n\n") if t.strip()]

# === CHUNK TEXT ===

def chunk_text(texts, chunk_size=CHUNK_SIZE):
    chunks = []
    for text in texts:
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.split()) > 5:  # skip very short ones
                chunks.append(chunk)
    return chunks

# === EMBEDDING & FAISS ===

def build_faiss_index(chunks, embedding_model="all-MiniLM-L6-v2"):
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings, chunks

# === MAIN ===

def main():
    discourse_texts = load_discourse_posts()
    tds_texts = load_tds_site_text()
    all_texts = discourse_texts + tds_texts
    chunked_texts = chunk_text(all_texts)

    index, embeddings, texts = build_faiss_index(chunked_texts)

    # Save index and metadata
    os.makedirs(FAISS_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(TEXTS_PATH, np.array(texts))

    print(f"FAISS index created and saved with total chunks: {len(texts)}")

if __name__ == "__main__":
    main()
