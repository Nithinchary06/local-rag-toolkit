"""Lightweight TF-IDF retrieval baseline."""

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
        return [SearchResult(self.chunks[index], float(score)) for index, score in ranked[:top_k] if score > 0]
