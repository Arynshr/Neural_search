from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Groq
    groq_api_key: str
    groq_model: str = "llama3-8b-8192"

    # Qdrant — collections are named dynamically; this is the storage root
    qdrant_path: Path = Path("./data/qdrant")

    # BM25 — root dir; each collection gets its own subdir
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

    def ensure_dirs(self) -> None:
        for path in [self.qdrant_path, self.bm25_index_path, self.data_dir]:
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
