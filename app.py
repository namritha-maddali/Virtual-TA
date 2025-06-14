# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from parse_input import get_answer, load_request_payload

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to promptfoo server origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionPayload(BaseModel):
    question: str
    image: Optional[str] = None

@app.post("/api/")
async def answer(payload: QuestionPayload):
    try:
        question, image = load_request_payload(payload.dict())
        result = get_answer(question, image)
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "TDS Virtual TA is up!"}
