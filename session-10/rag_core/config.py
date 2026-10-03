import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseModel):
    data_dir: Path = ROOT / "data" / "policies"
    index_dir: Path = ROOT / ".rag-index"
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = Field(default=500, ge=100, le=4000)
    chunk_overlap: int = Field(default=80, ge=0)
    top_k: int = Field(default=3, ge=1, le=8)
    min_score: float = Field(default=0.25, ge=0, le=1)
    timeout_seconds: float = Field(default=30, gt=0, le=120)
    max_retries: int = Field(default=2, ge=0, le=3)

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size")
        return self


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env", override=False)
    return Settings(
        chat_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        chunk_size=os.getenv("RAG_CHUNK_SIZE", "500"),
        chunk_overlap=os.getenv("RAG_CHUNK_OVERLAP", "80"),
        top_k=os.getenv("RAG_TOP_K", "3"),
        min_score=os.getenv("RAG_MIN_SCORE", "0.25"),
        timeout_seconds=os.getenv("RAG_TIMEOUT_SECONDS", "30"),
        max_retries=os.getenv("RAG_MAX_RETRIES", "2"),
    )
