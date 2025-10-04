from typing import Dict, Any, List
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from app.config import settings

def _format_citations(source_nodes, max_chars: int = 240) -> List[Dict[str, Any]]:
    cites = []
    for n in source_nodes:
        node = n.node
        meta = node.metadata or {}
        cites.append({
            "doc_id": getattr(node, "doc_id", None),
            "score": getattr(n, "score", None),
            "file": meta.get("filename") or meta.get("source") or meta.get("file"),
            "page": meta.get("page") or meta.get("page_label"),
            "snippet": (node.get_content() or "")[:max_chars].strip(),
        })
    return cites

def query_rag(prompt: str):
    vector_store = PGVectorStore.from_params(
        database=settings.db_name,
        host=settings.db_host,
        password=settings.db_pass,
        port=settings.db_port,
        user=settings.db_user,
        schema_name="public",
        table_name=settings.pgvector_collection,
        embed_dim=768
    )
    storage = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = OllamaEmbedding(model_name=settings.embed_model, base_url=settings.ollama_base_url)
    llm = Ollama(model=settings.llm_model, base_url=settings.ollama_base_url, request_timeout=600)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model, storage_context=storage)

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=5,
        response_mode="compact",
        verbose=False,
    )

    resp = query_engine.query(prompt)
    answer = str(resp)

    citations = _format_citations(getattr(resp, "source_nodes", []) or [])
    return {"answer": answer, "citations": citations}