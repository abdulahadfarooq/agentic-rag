from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.db import ensure_pgvector
from app.routers import health, ingest, query, agents

app = FastAPI(title="Agentic RAG Backend", debug=True)

@app.exception_handler(Exception)
async def unhandled_exc(_req: Request, exc: Exception):
    import traceback, sys
    traceback.print_exc(file=sys.stderr)
    return JSONResponse(status_code=500, content={"detail": str(exc)})

origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    try:
        ensure_pgvector()
        print("[startup] pgvector ready")
    except Exception as e:
        import traceback; traceback.print_exc()
        raise

app.include_router(health.router, tags=["health"])
app.include_router(ingest.router, tags=["ingest"])
app.include_router(query.router, tags=["query"])
app.include_router(agents.router, tags=["agents"])
