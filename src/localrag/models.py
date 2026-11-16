"""Core data models used by LocalRAG."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """A source document or page of a source document."""
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    """A chunk of text that can be indexed and retrieved."""
    text: str
    source: str
    chunk_id: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResult:
    """A retrieved chunk and its relevance score."""
    chunk: Chunk
    score: float


@dataclass(slots=True)
class RAGResponse:
    """Generated answer and the retrieved evidence behind it."""
    answer: str
    sources: list[SearchResult]
