"""
Shared UI configuration — single source of truth for all components.
"""
import os


def _normalize_url(url: str) -> str:
    return url.rstrip("/")


API_BASE: str = _normalize_url(
    os.environ.get("NEURAL_SEARCH_API_URL", "http://localhost:8000")
)
