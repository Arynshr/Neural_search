"""
Integration tests for the full ingestion pipeline.
Tests parse → chunk → index flow end-to-end with real file I/O.
No mocking of core components — only external services (Qdrant, BM25).
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from neural_search.ingestion.pipeline import run_ingestion
from neural_search.ingestion.parser import parse_document
from neural_search.ingestion.chunker import chunk_pages


class TestParseThenChunk:
    def test_pdf_parse_and_chunk_produces_chunks(self, sample_pdf):
        pages = parse_document(sample_pdf)
        assert len(pages) > 0
        chunks = chunk_pages(pages)
        assert len(chunks) > 0

    def test_docx_parse_and_chunk_produces_chunks(self, sample_docx):
        pages = parse_document(sample_docx)
        chunks = chunk_pages(pages)
        assert len(chunks) > 0

    def test_chunks_have_valid_metadata(self, sample_pdf):
        pages = parse_document(sample_pdf)
        chunks = chunk_pages(pages)
        for c in chunks:
            assert c.chunk_id and len(c.chunk_id) == 16
            assert c.source_file == sample_pdf.name
            assert c.token_count > 0
            assert len(c.text.strip()) > 0

    def test_chunk_ids_unique_across_documents(self, sample_pdf, sample_docx):
        pages_pdf  = parse_document(sample_pdf)
        pages_docx = parse_document(sample_docx)
        chunks_pdf  = chunk_pages(pages_pdf)
        chunks_docx = chunk_pages(pages_docx)
        all_ids = [c.chunk_id for c in chunks_pdf + chunks_docx]
        assert len(all_ids) == len(set(all_ids))


class TestRunIngestion:
    def test_ingestion_calls_sparse_index(self, sample_pdf):
        sparse = MagicMock()
        chunks = run_ingestion(source=sample_pdf, sparse_retriever=sparse)
        assert len(chunks) > 0
        sparse.index.assert_called_once()
        indexed_chunks = sparse.index.call_args[0][0]
        assert len(indexed_chunks) == len(chunks)

    def test_ingestion_calls_dense_upsert(self, sample_pdf):
        dense = MagicMock()
        chunks = run_ingestion(source=sample_pdf, dense_retriever=dense)
        dense.upsert.assert_called_once()

    def test_ingestion_with_both_retrievers(self, sample_pdf):
        sparse = MagicMock()
        dense = MagicMock()
        chunks = run_ingestion(source=sample_pdf, sparse_retriever=sparse, dense_retriever=dense)
        assert len(chunks) > 0
        sparse.index.assert_called_once()
        dense.upsert.assert_called_once()

    def test_reset_wipes_before_indexing(self, sample_pdf):
        sparse = MagicMock()
        dense = MagicMock()
        run_ingestion(source=sample_pdf, sparse_retriever=sparse, dense_retriever=dense, reset=True)
        sparse.reset.assert_called_once()
        dense.reset.assert_called_once()

    def test_nonexistent_path_returns_empty(self, tmp_path):
        ghost = tmp_path / "ghost.pdf"
        chunks = run_ingestion(source=ghost)
        assert chunks == []

    def test_exports_jsonl_snapshot(self, sample_pdf, tmp_path, monkeypatch):
        from neural_search import config
        mock = MagicMock()
        mock.chunk_size = 256
        mock.chunk_overlap = 32
        mock.bm25_index_path = tmp_path / "bm25"
        mock.bm25_path_for = lambda s: tmp_path / "bm25" / s
        mock.documents_path_for = lambda s: tmp_path / "docs" / s
        monkeypatch.setattr(config, "settings", mock)

        snapshot_path = tmp_path / "bm25" / "snapshots" / "chunks.jsonl"
        run_ingestion(source=sample_pdf, export_snapshot=True)
        assert snapshot_path.exists()
        lines = snapshot_path.read_text().strip().splitlines()
        assert len(lines) > 0

    def test_directory_ingestion(self, tmp_path, sample_pdf, sample_docx):
        import shutil
        shutil.copy(sample_pdf, tmp_path / "doc1.pdf")
        shutil.copy(sample_docx, tmp_path / "doc2.docx")
        sparse = MagicMock()
        chunks = run_ingestion(source=tmp_path, sparse_retriever=sparse)
        assert len(chunks) > 0
        # Should index once with all chunks combined
        sparse.index.assert_called_once()
