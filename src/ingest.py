"""Document loading and word-based chunking with overlap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    token_count: int


def load_documents(data_dir: str | Path) -> list[tuple[str, str, str]]:
    """Return (doc_id, title, text) for every .md/.txt file in data_dir."""
    docs = []
    for path in sorted(Path(data_dir).glob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8").strip()
        title = path.stem
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        docs.append((path.stem, title, text))
    if not docs:
        raise FileNotFoundError(f"No .md/.txt documents found in {data_dir}")
    return docs


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """Split text into word chunks of ~chunk_size words with `overlap` word overlap."""
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks, step = [], chunk_size - overlap
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_size]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return chunks


def build_corpus(
    data_dir: str | Path, chunk_size: int = 200, overlap: int = 40
) -> list[Chunk]:
    """Load every document and return its chunks with stable IDs."""
    corpus: list[Chunk] = []
    for doc_id, title, text in load_documents(data_dir):
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            corpus.append(
                Chunk(
                    chunk_id=f"{doc_id}::c{i:02d}",
                    doc_id=doc_id,
                    title=title,
                    text=piece,
                    token_count=len(piece.split()),
                )
            )
    return corpus
