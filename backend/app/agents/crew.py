import json
from typing import Optional, Dict, Any
from crewai import Agent, Task, Crew, Process
from langchain_community.chat_models import ChatOllama
from crewai.tools import BaseTool
from app.services.retrieval import query_rag

from app.config import settings

class RagQueryTool(BaseTool):
    name: str = "rag_query"
    description: str = """
    Query the vector DB and return a JSON string:
    {"answer": str, "citations": [{file, page, score, snippet, doc_id}, ...]}
    """
    def _run(self, query: str) -> str:
        data = query_rag(query)
        return json.dumps(data)


def build_crew() -> Crew:
    llm = ChatOllama(model=settings.llm_model, base_url=settings.ollama_base_url)

    answer_agent = Agent(
        role="RAG Answering Agent",
        goal=(
            "Use the 'rag_query' tool to answer the user's question strictly from the vector DB. "
            "Return ONLY the tool's JSON output without rewriting or inventing citations."
        ),
        backstory=(
            "You are a precise retrieval specialist. You never fabricate sources and you always "
            "return JSON from the tool exactly as it is."
        ),
        tools=[RagQueryTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    task_template = (
        "Given the exact user query below, use the rag_query tool to search the vector database:\n\n"
        "QUERY: {query}\n\n"
        "Instructions:\n"
        "1) Call the rag_query tool with the user query as the 'query' parameter\n"
        "2) Do not invent or reformulate citations\n"
        "3) Do not summarize the tool's output\n"
        "4) Return ONLY the JSON string produced by the tool\n"
        "5) The tool expects a string parameter called 'query'"
    )

    answer_task = Task(
        description=task_template,
        expected_output='A JSON string like {"answer": "...", "citations": [...]}',
        agent=answer_agent,
        inputs={"query": "{query}"}
    )

    return Crew(
        agents=[answer_agent],
        tasks=[answer_task],
        process=Process.sequential,
        verbose=True,
    )

def run_agentic_rag(query: str) -> Dict[str, Any]:
    crew = build_crew()
    result = crew.kickoff(inputs={"query": query})
    try:
        text = result.raw if hasattr(result, "raw") else str(result)
    except Exception:
        text = str(result)
    return text
