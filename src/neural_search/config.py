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

    # Qdrant
    qdrant_path: Path = Path("./data/qdrant")
    qdrant_collection: str = "neural_search"

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

    # Paths
    documents_dir: Path = Path("./data/documents")

    def ensure_dirs(self) -> None:
        """Create all required data directories if they don't exist."""
        for path in [self.qdrant_path, self.bm25_index_path, self.documents_dir]:
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
