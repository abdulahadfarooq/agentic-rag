from typing import Iterable, List
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

from app.config import settings
from app.services.doc_parse import docling_extract_text

def build_pgvector_store():
    return PGVectorStore.from_params(
        database=settings.db_name,
        host=settings.db_host,
        password=settings.db_pass,
        port=settings.db_port,
        user=settings.db_user,
        schema_name="public",
        table_name=settings.pgvector_collection,
        embed_dim=768
    )

def parse_to_documents(raw_texts: Iterable[str]) -> List[Document]:
    return [Document(text=t) for t in raw_texts]

def ingest_texts(texts: Iterable[str]) -> int:
    vector_store = build_pgvector_store()
    storage = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = OllamaEmbedding(model_name=settings.embed_model, base_url=settings.ollama_base_url)
    docs = parse_to_documents(texts)
    VectorStoreIndex.from_documents(docs, storage_context=storage, embed_model=embed_model)
    return len(docs)

def ingest_upload_with_docling(upload_file) -> dict:
    """
    Accepts a FastAPI UploadFile, extracts with Docling, and ingests.
    Returns metadata for observability.
    """
    text, meta = docling_extract_text(upload_file)
    count = ingest_texts([text])
    meta["ingested_documents"] = count
    return meta
