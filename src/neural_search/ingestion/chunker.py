import hashlib
from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from neural_search.config import settings
from neural_search.ingestion.parser import ParsedPage


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_file: str
    page: int
    chunk_index: int
    text: str
    token_count: int
    metadata: dict = field(default_factory=dict)


def _make_chunk_id(source_file: str, chunk_index: int) -> str:
    raw = f"{source_file}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_pages(pages: list[ParsedPage]) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,   # character-based, honest about unit
    )

    all_chunks: list[Chunk] = []
    global_index = 0

    for page in pages:
        splits = splitter.split_text(page.text)
        if not splits:
            logger.debug(f"No chunks from {page.source_file} page {page.page}")
            continue

        for i, text in enumerate(splits):
            clean_text = text.strip()
            chunk = Chunk(
                chunk_id=_make_chunk_id(page.source_file, global_index),
                doc_id=page.doc_id,
                source_file=page.source_file,
                page=page.page,
                chunk_index=global_index,
                text=clean_text,
                token_count=len(clean_text.split()),   # approx word count
                metadata={**page.metadata, "split_index": i},
            )
            all_chunks.append(chunk)
            global_index += 1

    logger.info(f"Total chunks produced: {len(all_chunks)}")
    return all_chunks
