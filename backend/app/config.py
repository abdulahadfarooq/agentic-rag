from math import e
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    db_host: str = "postgres"
    db_port: int = 5432
    db_name: str = "ragdb"
    db_user: str = "rag"
    db_pass: str = "ragpass"
    pgvector_collection: str = "documents"
    llm_model: str = "llama3.2:1b"
    embed_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://ollama:11434"
    phoenix_host: str = "http://phoenix:6006"
    cors_allow_origins: str = "http://localhost:8080,http://openwebui:8080"

    class Config:
        env_file = ".env"
    
settings = Settings()
