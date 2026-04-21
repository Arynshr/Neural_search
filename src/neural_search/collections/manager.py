"""
CollectionManager — handles lifecycle of named document collections.
"""
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger
from neural_search.config import settings

MAX_COLLECTIONS = 10


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


class CollectionManager:
    def __init__(self):
        self._base = settings.data_dir / "collections"
        self._base.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, slug: str) -> Path:
        return self._base / slug / "metadata.json"

    def _read_meta(self, slug: str) -> dict:
        path = self._meta_path(slug)
        if not path.exists():
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Corrupt metadata for collection '{slug}' — skipping")
            return {}

    def _write_meta(self, slug: str, meta: dict) -> None:
        path = self._meta_path(slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=2)
        tmp.replace(path)

    def list_collections(self) -> list[dict]:
        collections = []
        for path in sorted(self._base.iterdir()):
            if path.is_dir():
                meta = self._read_meta(path.name)
                if meta:
                    collections.append(meta)
        return collections

    def get_collection(self, slug: str) -> dict | None:
        meta = self._read_meta(slug)
        return meta if meta else None

    def create_collection(self, name: str, description: str = "") -> dict:
        if len(self.list_collections()) >= MAX_COLLECTIONS:
            raise ValueError(f"Collection limit reached ({MAX_COLLECTIONS}). Delete one first.")

        slug = slugify(name)
        if not slug:
            raise ValueError("Invalid collection name.")

        if self.get_collection(slug):
            raise ValueError(f"Collection '{name}' already exists.")

        now = datetime.now(timezone.utc)

        meta = {
            "slug": slug,
            "name": name,
            "description": description,
            "created_at": now,
            "updated_at": now,
            "files": [],
            "total_chunks": 0,
            "total_tokens": 0,
        }

        self._write_meta(slug, meta)
        logger.info(f"Collection created: '{name}' (slug={slug})")
        return meta

    def delete_collection(self, slug: str) -> None:
        if not self.get_collection(slug):
            raise ValueError(f"Collection '{slug}' not found.")

        for path in [
            settings.data_dir / "bm25_index" / slug,
            settings.data_dir / "documents" / slug,
            self._base / slug,
        ]:
            if path.exists():
                shutil.rmtree(path)

        logger.info(f"Collection deleted: '{slug}'")

    def add_file_record(self, slug: str, record: dict) -> None:
        meta = self._read_meta(slug)
        if not meta:
            raise ValueError(f"Collection '{slug}' not found.")

        existing = [f for f in meta.get("files", []) if f["filename"] != record["filename"]]
        existing.append(record)

        meta["files"] = existing
        meta["total_chunks"] = sum(f.get("chunks", 0) for f in existing)
        meta["total_tokens"] = sum(f.get("tokens", 0) for f in existing)
        meta["updated_at"] = datetime.now(timezone.utc)

        self._write_meta(slug, meta)

    def file_exists(self, slug: str, filename: str) -> bool:
        meta = self._read_meta(slug)
        return any(f.get("filename") == filename for f in meta.get("files", []))
