"""High-level local RAG pipeline."""

from pathlib import Path
from typing import Protocol

from .chunking import chunk_documents
from .generators import build_prompt
from .loaders import load_documents
from .models import RAGResponse, SearchResult
from .retrieval import SentenceTransformerRetriever, TfidfRetriever


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


class LocalRAG:
    """Load documents, retrieve evidence, and optionally generate an answer."""

    def __init__(self, retriever) -> None:
        self.retriever = retriever

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        chunk_size: int = 180,
        overlap: int = 30,
        retriever: str = "tfidf",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> "LocalRAG":
        """Build a RAG pipeline using TF-IDF or optional semantic retrieval."""
        documents = load_documents(path)
        chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)

        if retriever == "tfidf":
            searcher = TfidfRetriever(chunks)
        elif retriever == "semantic":
            searcher = SentenceTransformerRetriever(chunks, model_name=embedding_model)
        else:
            raise ValueError("retriever must be either 'tfidf' or 'semantic'")
        return cls(searcher)

    def retrieve(self, question: str, *, top_k: int = 4) -> list[SearchResult]:
        return self.retriever.search(question, top_k=top_k)

    def ask(self, question: str, generator: TextGenerator, *, top_k: int = 4) -> RAGResponse:
        sources = self.retrieve(question, top_k=top_k)
        prompt = build_prompt(question, sources)
        return RAGResponse(answer=generator.generate(prompt), sources=sources)
