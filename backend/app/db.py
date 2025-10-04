from sqlalchemy import create_engine, text
from app.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)

def ensure_pgvector():
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
