import json
import os
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from groq import Groq
# from dotenv import load_dotenv

# load_dotenv()
groq_api = os.getenv("GROQ_API")

# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db = FAISS.load_local("data/faiss", embeddings=embedding_model, index_name="index", allow_dangerous_deserialization=True)

client = Groq(api_key=groq_api)

def handle_with_llm_rag(question, db):
    docs = db.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
        You are a knowledgeable virtual teaching assistant for the IIT Madras 'Tools in Data Science' course.

        Based on the context below, answer the student question strictly using the available information. If relevant links (e.g., Discourse posts or site URLs) are referenced in the context or needed for further reading, include them.

        Return your answer as a JSON object with exactly two keys:
        1. "answer" (string) - A clear, complete explanation which is concise and natural, as if a human is answering
        2. "links" (list) - Each item is a dictionary with keys "url" and "text".

        Student Question: {question}
        
        Context for answering the question: {context}

        IMPORTANT: Only use URLs that are **explicitly mentioned** in the provided context. Do NOT invent links.
        Output Format (no need to give introduction and conclusion before the json output):
        {{
            "answer": "answer to the question ...",
            "links": [
                {{
                    "url": "..." (this link MUST be present in the context above; do NOT make up or invent any links),,
                    "text": "explain how this link is useful ..."
                }}
            ]
        }}
    """

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful and structured assistant."},
            {"role": "user", "content": prompt}
        ],
        model = "llama3-70b-8192",
        temperature = 0.1,
        seed = 700
    )

    content = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(content)
        return parsed
    except json.JSONDecodeError:
        return {"answer": content, "links": []}
