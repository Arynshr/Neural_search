"""
Unit tests for collections/manager.py
Tests: create, list, delete, duplicate prevention, file records, slug generation.
"""
import pytest
from neural_search.collections.manager import CollectionManager, slugify


@pytest.fixture
def manager(tmp_path, monkeypatch):
    from neural_search import config
    from unittest.mock import MagicMock
    mock = MagicMock()
    mock.data_dir = tmp_path / "data"
    mock.bm25_index_path = tmp_path / "data" / "bm25_index"
    monkeypatch.setattr(config, "settings", mock)
    m = CollectionManager.__new__(CollectionManager)
    m._base = tmp_path / "data" / "collections"
    m._base.mkdir(parents=True, exist_ok=True)
    return m


class TestSlugify:
    def test_lowercases(self):
        assert slugify("HR Policies") == "hr-policies"

    def test_replaces_spaces_with_dashes(self):
        assert slugify("Q3 Sales Report") == "q3-sales-report"

    def test_strips_special_chars(self):
        assert slugify("My Doc!@#") == "my-doc"

    def test_collapses_multiple_separators(self):
        assert slugify("HR  --  Policies") == "hr-policies"


class TestCreateCollection:
    def test_creates_and_returns_meta(self, manager):
        col = manager.create_collection("HR Policies", "HR docs")
        assert col["slug"] == "hr-policies"
        assert col["name"] == "HR Policies"
        assert col["description"] == "HR docs"
        assert col["total_chunks"] == 0
        assert col["files"] == []

    def test_metadata_persisted_to_disk(self, manager):
        manager.create_collection("Test Collection")
        col = manager.get_collection("test-collection")
        assert col is not None
        assert col["name"] == "Test Collection"

    def test_duplicate_name_raises(self, manager):
        manager.create_collection("HR Policies")
        with pytest.raises(ValueError, match="already exists"):
            manager.create_collection("HR Policies")

    def test_collection_limit_enforced(self, manager, monkeypatch):
        monkeypatch.setattr("neural_search.collections.manager.MAX_COLLECTIONS", 2)
        manager.create_collection("Col One")
        manager.create_collection("Col Two")
        with pytest.raises(ValueError, match="limit reached"):
            manager.create_collection("Col Three")

    def test_timestamps_are_set(self, manager):
        col = manager.create_collection("Timestamped")
        assert col["created_at"] is not None
        assert col["updated_at"] is not None


class TestListCollections:
    def test_empty_returns_empty_list(self, manager):
        assert manager.list_collections() == []

    def test_lists_all_created(self, manager):
        manager.create_collection("Alpha")
        manager.create_collection("Beta")
        cols = manager.list_collections()
        names = [c["name"] for c in cols]
        assert "Alpha" in names
        assert "Beta" in names

    def test_returns_sorted_by_slug(self, manager):
        manager.create_collection("Zebra")
        manager.create_collection("Alpha")
        cols = manager.list_collections()
        slugs = [c["slug"] for c in cols]
        assert slugs == sorted(slugs)


class TestDeleteCollection:
    def test_delete_removes_metadata(self, manager, tmp_path):
        manager.create_collection("Temp")
        manager.delete_collection("temp")
        assert manager.get_collection("temp") is None

    def test_delete_nonexistent_raises(self, manager):
        with pytest.raises(ValueError, match="not found"):
            manager.delete_collection("ghost")

    def test_after_delete_can_recreate_same_name(self, manager, tmp_path):
        manager.create_collection("Reusable")
        manager.delete_collection("reusable")
        col = manager.create_collection("Reusable")
        assert col["slug"] == "reusable"


class TestFileRecords:
    def test_add_file_record_updates_meta(self, manager):
        manager.create_collection("Docs")
        manager.add_file_record("docs", {
            "filename": "report.pdf",
            "pages": 5,
            "chunks": 20,
            "tokens": 1500,
            "ingested_at": "2026-04-20T10:00:00",
            "status": "ok",
        })
        col = manager.get_collection("docs")
        assert len(col["files"]) == 1
        assert col["total_chunks"] == 20
        assert col["total_tokens"] == 1500

    def test_re_adding_same_filename_overwrites(self, manager):
        manager.create_collection("Docs")
        for chunks in [10, 25]:
            manager.add_file_record("docs", {
                "filename": "report.pdf", "pages": 1,
                "chunks": chunks, "tokens": 100,
                "ingested_at": "2026-04-20T10:00:00", "status": "ok",
            })
        col = manager.get_collection("docs")
        assert len(col["files"]) == 1
        assert col["total_chunks"] == 25

    def test_file_exists_returns_true(self, manager):
        manager.create_collection("Docs")
        manager.add_file_record("docs", {
            "filename": "exist.pdf", "pages": 1, "chunks": 5,
            "tokens": 100, "ingested_at": "2026-04-20", "status": "ok",
        })
        assert manager.file_exists("docs", "exist.pdf") is True

    def test_file_exists_returns_false(self, manager):
        manager.create_collection("Docs")
        assert manager.file_exists("docs", "ghost.pdf") is False
