import os
os.environ["HF_HOME"] = "/tmp/.hf_cache"
os.makedirs("/tmp/.hf_cache", exist_ok=True)

import base64
from PIL import Image
from io import BytesIO
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from vlm_response import handle_with_vlm
from llm_rag import handle_with_llm_rag
from huggingface_hub import hf_hub_download

faiss_index_path = hf_hub_download(
    repo_id="nomri/Tadashi-TDS-TA",
    filename="data/faiss/index.faiss",
    repo_type="space"
)

pkl_path = hf_hub_download(
    repo_id="nomri/Tadashi-TDS-TA",
    filename="data/faiss/index.pkl",
    repo_type="space"
)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(
    folder_path=os.path.dirname(faiss_index_path),
    embeddings=embedding_model,
    index_name="index",
    allow_dangerous_deserialization=True
)

def load_request_payload(payload: dict):
    question = payload.get("question", "")
    image_b64 = payload.get("image")
    
    image = None
    if image_b64:
        image_data = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_data)).convert("RGB")

    return question, image

def get_answer(question, image=None):
    if image:
        return handle_with_vlm(question, image, db)
    else:
        return handle_with_llm_rag(question, db)
    