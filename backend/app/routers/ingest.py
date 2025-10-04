from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.indexing import ingest_texts, ingest_upload_with_docling

router = APIRouter()

@router.post("/ingest/text")
async def ingest_text_endpoint(text: str):
    """
    Simple text ingestion (raw string).
    """
    n = ingest_texts([text])
    return {"ingested_documents": n}

@router.post("/ingest/files")
async def ingest_files_endpoint(files: List[UploadFile] = File(...)):
    """
    Docling-powered ingestion for PDFs/DOCX/etc.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []
    for f in files:
        await f.seek(0)
        meta = ingest_upload_with_docling(f)
        results.append(meta)

    total = sum(m.get("ingested_documents", 0) for m in results)
    return {"total_ingested": total, "files": results}
