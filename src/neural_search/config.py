from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_env_file() -> Path:
    """Walk up from CWD to find .env at project root, not inside src/."""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return cwd / ".env"   # fallback — won't crash if missing


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_find_env_file()),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",          # #22: unknown env vars silently ignored, not crash
    )

    # Groq
    groq_api_key: str = "not-set"
    groq_model: str = "llama-3.1-8b-instant"

    # Qdrant
    qdrant_path: Path = Path("./data/qdrant")

    # BM25
    bm25_index_path: Path = Path("./data/bm25_index")

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k: int = 10
    rrf_k: int = 60

    # Data root
    data_dir: Path = Path("./data")

    def bm25_path_for(self, collection_slug: str) -> Path:
        return self.bm25_index_path / collection_slug

    def documents_path_for(self, collection_slug: str) -> Path:
        return self.data_dir / "documents" / collection_slug

    def snapshot_path_for(self, collection_slug: str) -> Path:
        """Per-collection JSONL snapshot path — fixes #9 multi-collection overwrite."""
        return self.data_dir / "snapshots" / f"{collection_slug}.jsonl"

    def ensure_dirs(self) -> None:
        for path in [self.qdrant_path, self.bm25_index_path, self.data_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def assert_groq_configured(self) -> None:
        if self.groq_api_key == "not-set":
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
