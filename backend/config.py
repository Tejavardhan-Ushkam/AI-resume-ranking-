from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Intelligent Candidate Discovery"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # AI Provider - uses Anthropic Claude for JD analysis + explanations
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None  # fallback for embeddings

    # Embedding model (local, no API key needed)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # fast, good quality

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION: str = "candidates"

    # Ranking weights (tunable)
    WEIGHT_SEMANTIC: float = 0.35
    WEIGHT_SKILLS: float = 0.25
    WEIGHT_CAREER_DNA: float = 0.25
    WEIGHT_BEHAVIORAL: float = 0.15

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()