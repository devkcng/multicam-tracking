from fastapi import FastAPI
from services.retrieval import *
from pydantic import BaseModel
from typing import List

app = FastAPI()
class SearchRequest(BaseModel):
    text: str
    k: int = 10


class SearchResult(BaseModel):
    camera_id: str
    frame: int
    time_seconds: float

faiss_engine = Retrieval()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.post("/search", response_model=List[SearchResult])
def search(request: SearchRequest):
    result = faiss_engine.search(request.text, k=request.k)
    return result
