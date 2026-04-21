from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    groq_api_key: str = Field(default="not-set", alias="GROQ_API_KEY")
    groq_model: str = "llama3-8b-8192"

    qdrant_path: Path = Path("./data/qdrant")
    bm25_index_path: Path = Path("./data/bm25_index")

    embedding_model: str = "all-MiniLM-L6-v2"

    chunk_size: int = 512
    chunk_overlap: int = 64

    top_k: int = 10
    rrf_k: int = 60

    data_dir: Path = Path("./data")

    def bm25_path_for(self, collection_slug: str) -> Path:
        return self.bm25_index_path / collection_slug

    def documents_path_for(self, collection_slug: str) -> Path:
        return self.data_dir / "documents" / collection_slug

    def ensure_dirs(self) -> None:
        for path in [self.qdrant_path, self.bm25_index_path, self.data_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def assert_groq_configured(self) -> None:
        if self.groq_api_key == "not-set":
            raise RuntimeError("GROQ_API_KEY is not set.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
