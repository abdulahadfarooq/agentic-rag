import os
import tempfile
from typing import Tuple

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import ConversionStatus
from fastapi import HTTPException, UploadFile


SUPPORTED_EXTS = {".pdf", ".docx", ".pptx", ".rtf", ".txt", ".md", ".html", ".png", ".jpg", ".jpeg", ".tiff"}

def _safe_suffix(filename: str) -> str:
    _, ext = os.path.splitext(filename.lower())
    return ext if ext in SUPPORTED_EXTS else ""

def docling_extract_text(upload: UploadFile) -> Tuple[str, dict]:
    """
    Returns (plain_text, meta) or raises HTTPException on failure.
    """
    suffix = _safe_suffix(upload.filename or "")
    if not suffix:
        raise HTTPException(status_code=400, detail=f"Unsupported file type for: {upload.filename}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.file.read())
        tmp_path = tmp.name

    try:
        converter = DocumentConverter()
        result = converter.convert(tmp_path)
        if result.status != ConversionStatus.SUCCESS:
            raise HTTPException(status_code=422, detail=f"Docling failed to convert {upload.filename}: {result.status}")

        markdown = result.document.export_to_markdown()
        text = markdown if markdown and markdown.strip() else result.document.export_to_text() or ""

        if not text.strip():
            raise HTTPException(status_code=422, detail=f"No extractable text found in {upload.filename}")

        meta = {
            "filename": upload.filename,
            "pages": len(getattr(result.document, "pages", []) or []),
            "status": str(result.status),
            "detectors": [d.name for d in getattr(result, "applied_detectors", [])] if hasattr(result, "applied_detectors") else [],
        }
        return text, meta
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
