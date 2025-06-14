import base64
from PIL import Image
from io import BytesIO
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from vlm_response import handle_with_vlm
from llm_rag import handle_with_llm_rag

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(
    "data/faiss",
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



