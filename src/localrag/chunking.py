"""Simple word-based text chunking with configurable overlap."""

from .models import Chunk, Document


def chunk_documents(documents: list[Document], *, chunk_size: int = 180, overlap: int = 30) -> list[Chunk]:
    """Split documents into word chunks while retaining source metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")
    chunks: list[Chunk] = []
    for document in documents:
        words = document.text.split()
        step = chunk_size - overlap
        for start in range(0, len(words), step):
            piece = words[start : start + chunk_size]
            if piece:
                chunks.append(Chunk(text=" ".join(piece), source=document.source, chunk_id=len(chunks), metadata={**document.metadata, "start_word": start}))
    return chunks
