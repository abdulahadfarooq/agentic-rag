from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.agents.crew import run_agentic_rag

router = APIRouter()

class AgentRunIn(BaseModel):
    question: str

@router.post("/agents/run")
def run_agents(body: AgentRunIn):
    if not body.question or not body.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")
    try:
        result = run_agentic_rag(body.question.strip())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent run failed: {e}")
