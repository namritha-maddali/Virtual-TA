'''
    1377 chunks generated, each with 200 words
'''

import os
import json
import re
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# === CONFIG ===
DISCOURSE_JSON_PATH = "data/raw/discourse_meta_data.json"
TDS_SITE_TEXT_PATH = "data/raw/tds_site_text.txt"
CHUNK_SIZE = 200  # words per chunk
FAISS_DIR = "data/faiss"

# === CLEANING FUNCTIONS ===
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

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

# === CHUNKING ===
def chunk_text(texts, chunk_size=CHUNK_SIZE):
    chunks = []
    for text in texts:
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.split()) > 5:
                chunks.append(chunk)
    return chunks

# === MAIN ===

def main():
    discourse_texts = load_discourse_posts()
    tds_texts = load_tds_site_text()
    all_texts = discourse_texts + tds_texts
    chunked_texts = chunk_text(all_texts)

    documents = [Document(page_content=chunk) for chunk in chunked_texts]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build FAISS vector store
    db = FAISS.from_documents(documents, embeddings)
    os.makedirs(FAISS_DIR, exist_ok=True)
    db.save_local(FAISS_DIR)

    print(f"LangChain FAISS index created and saved to {FAISS_DIR}. Chunks: {len(documents)}")

if __name__ == "__main__":
    main()
