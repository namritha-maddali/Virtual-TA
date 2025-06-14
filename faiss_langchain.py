'''
    1357 chunks generated, each with 200 words
'''

import os
import json
import re
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# === CONFIG ===
DISCOURSE_JSON_PATH = "data/raw/discourse_posts.json"
TDS_SITE_JSON_PATH = "data/raw/tds_site_text.json"
CHUNK_SIZE = 200  # words per chunk
FAISS_DIR = "data/faiss"

# === CLEANING FUNCTIONS ===
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def load_json_posts(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        posts = json.load(f)
    docs = []
    for post in posts:
        content = clean_text(post.get("content", ""))
        if content and post.get("url"):
            docs.append({
                "text": content,
                "url": post["url"]
            })
    return docs

# === CHUNKING ===
# === CHUNKING ===
def chunk_documents(docs, chunk_size=CHUNK_SIZE):
    chunks = []
    for doc in docs:
        words = doc["text"].split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.split()) > 5:
                chunks.append(Document(
                    page_content=chunk,
                    metadata={"source": doc["url"]}
                ))
    return chunks

# === MAIN ===
def main():
    discourse_docs = load_json_posts(DISCOURSE_JSON_PATH)
    tds_docs = load_json_posts(TDS_SITE_JSON_PATH)
    all_docs = discourse_docs + tds_docs

    documents = chunk_documents(all_docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(documents, embeddings)
    os.makedirs(FAISS_DIR, exist_ok=True)
    db.save_local(FAISS_DIR)
    print(f"LangChain FAISS index created and saved to {FAISS_DIR}. Chunks: {len(documents)}")

if __name__ == "__main__":
    main()
