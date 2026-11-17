"""LocalRAG: a small, local-first retrieval-augmented generation toolkit."""

from .chunking import chunk_documents
from .generators import OllamaGenerator
from .loaders import load_documents
from .models import Chunk, Document, RAGResponse, SearchResult
from .pipeline import LocalRAG
from .retrieval import SentenceTransformerRetriever, TfidfRetriever

__all__ = [
    "Chunk",
    "Document",
    "LocalRAG",
    "OllamaGenerator",
    "RAGResponse",
    "SearchResult",
    "SentenceTransformerRetriever",
    "TfidfRetriever",
    "chunk_documents",
    "load_documents",
]

__version__ = "0.3.0"
