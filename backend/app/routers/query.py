from fastapi import APIRouter
from pydantic import BaseModel
from app.services.retrieval import query_rag

router = APIRouter()

class QueryIn(BaseModel):
    prompt: str

@router.post("/query")
def query(body: QueryIn):
    answer = query_rag(body.prompt)
    return answer
