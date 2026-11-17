"""Local retrieval backends: lexical TF-IDF and optional sentence embeddings."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Chunk, SearchResult


class TfidfRetriever:
    """Retrieve text chunks using TF-IDF cosine similarity."""

    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required to build the index")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        try:
            self.matrix = self.vectorizer.fit_transform(chunk.text for chunk in chunks)
        except ValueError as exc:
            raise ValueError("The supplied documents contain no indexable text") from exc

    def search(self, query: str, *, top_k: int = 4) -> list[SearchResult]:
        """Return up to top_k relevant chunks, highest score first."""
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [
            SearchResult(self.chunks[index], float(score))
            for index, score in ranked[:top_k]
            if score > 0
        ]


class SentenceTransformerRetriever:
    """Retrieve chunks by semantic similarity using a local sentence-transformer model.

    Install the optional dependency with: pip install "local-rag-toolkit[semantic]".
    The model is downloaded by sentence-transformers on first use unless a local
    model path is supplied. A model object can be injected for testing.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model=None,
    ) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required to build the index")
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    'Semantic retrieval requires the optional dependency. '
                    'Install it with: pip install "local-rag-toolkit[semantic]"'
                ) from exc
            model = SentenceTransformer(model_name)

        self.chunks = chunks
        self.model = model
        embeddings = self.model.encode(
            [chunk.text for chunk in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

    def search(self, query: str, *, top_k: int = 4) -> list[SearchResult]:
        """Return chunks ranked by cosine similarity in embedding space."""
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_vector = np.asarray(query_embedding, dtype=np.float32)[0]
        scores = self.embeddings @ query_vector
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [
            SearchResult(self.chunks[index], float(score))
            for index, score in ranked[:top_k]
        ]
