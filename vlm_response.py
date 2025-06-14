from PIL import Image
import base64
import io
import json
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from groq import Groq
import moondream as md
from dotenv import load_dotenv
import os

'''
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision='2025-04-14')
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    # device_map=device,
    revision='2025-04-14',
    trust_remote_code=True
)
'''

load_dotenv()
moondream_api = os.getenv("MOONDREAM_API")
model = md.vl(api_key=moondream_api)

# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db = FAISS.load_local("data/faiss", embeddings=embedding_model, index_name="index", allow_dangerous_deserialization=True)

groq_api = os.getenv("GROQ_API")
groq_client = Groq(api_key=groq_api)

def handle_with_vlm(question, image, db):
    # Step 1: Retrieve context using similarity search
    docs = db.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 2. get just the answer from question and context. 
    vlm_answer = model.query(image, "Transcribe the text in natural reading order")['answer'].strip()

    # 3. Combine VLM's answer, original question, and context for the LLM
    llm_rag_prompt = f"""
        You are a knowledgeable virtual teaching assistant for the IIT Madras 'Tools in Data Science' course.

        A student asked the following question: "{question}"
        This question has been asked along with an image which has this text (given by a VLM): "{vlm_answer}"

        Additionally, here is some relevant course context:
        {context}

        Based on the student's original question, the VLM's OCR output, and the provided course context,
        generate a clear, complete explanation. If relevant links (e.g: Discourse posts or site URLs)
        are referenced in the context or would be useful for further reading, include them and provide an explanation on them.
        Pay attention to what's mentioned in the VLM's output and correlate it to the asked question.

        Return your answer as a JSON object with exactly two keys:
        1. "answer" (string) - A clear, complete explanation, integrating the VLM's response.
        2. "links" (list) - Each item is a dictionary with keys "url" and "text".

        Output Format  (no need to give introduction and conclusion before the json output):
        {{
            "answer": "answer to the question ...",
            "links": [
                {{
                "url": "https://...",
                "text": "explain how this link is useful ..."
                }}
            ]
        }}
        """

    try:
        llm_response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful and structured assistant that outputs valid JSON."},
                {"role": "user", "content": llm_rag_prompt}
            ],
            model="llama3-70b-8192", # Or whatever model you prefer for JSON generation
            temperature=0.1,
            seed=700
        )

        content = llm_response.choices[0].message.content.strip()
        parsed = json.loads(content)

        if "answer" not in parsed or "links" not in parsed:
            raise ValueError("LLM output JSON missing required 'answer' or 'links' keys.")
        if not isinstance(parsed["links"], list):
            raise ValueError("LLM output 'links' key is not a list.")

        return parsed

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: LLM output was not perfect JSON or missing keys. Error: {e}")
        print(f"Raw LLM response: {content if 'content' in locals() else 'No LLM response.'}")
        # Fallback if LLM fails to produce perfect JSON
        final_answer = vlm_answer # Start with VLM's direct answer
        if 'content' in locals() and content: # If LLM produced some content, append it
            final_answer += "\n\n(Further LLM processing: " + content + ")"
        
        return {
            "answer": final_answer.strip(),
            "links": []
        }