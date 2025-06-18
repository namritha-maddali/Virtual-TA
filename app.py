from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from parse_input import get_answer, load_request_payload
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend", html=True), name="static")

@app.get("/")
def serve_index():
    return FileResponse("frontend/index.html")

class QuestionPayload(BaseModel):
    question: str
    image: Optional[str] = None

@app.post("/api/")
async def answer(payload: QuestionPayload):
    try:
        question, image = load_request_payload(payload.dict())
        result = get_answer(question, image)
        print(result)
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/test")
def root():
    return {"message": "Tadashi (TDS TA) works guys!"}
