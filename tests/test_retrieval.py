import numpy as np
import pytest

from localrag import Chunk, SentenceTransformerRetriever, TfidfRetriever
from localrag.store import SQLiteVectorStore


def test_retriever_ranks_relevant_chunk_first():
    chunks = [
        Chunk("Python is a programming language.", "python.txt", 0),
        Chunk("Bananas are a source of potassium.", "fruit.txt", 1),
    ]
    retriever = TfidfRetriever(chunks)
    results = retriever.search("Python programming", top_k=2)
    assert results[0].chunk.source == "python.txt"
    assert results[0].score > 0


def test_retriever_validates_query():
    retriever = TfidfRetriever([Chunk("sample text", "sample.txt", 0)])
    with pytest.raises(ValueError):
        retriever.search("   ")


class FakeEmbeddingModel:
    """Deterministic encoder for testing semantic ranking without model downloads."""

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            value = text.lower()
            if "vehicle" in value or "automobile" in value:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


def test_semantic_retriever_ranks_related_meaning():
    chunks = [
        Chunk("A vehicle is used for road travel.", "vehicle.txt", 0),
        Chunk("Fresh fruit can be sweet.", "fruit.txt", 1),
    ]
    retriever = SentenceTransformerRetriever(chunks, model=FakeEmbeddingModel())
    results = retriever.search("automobile", top_k=2)

    assert results[0].chunk.source == "vehicle.txt"
    assert results[0].score == pytest.approx(1.0)


def test_semantic_retriever_validates_query():
    chunks = [Chunk("sample content", "sample.txt", 0)]
    retriever = SentenceTransformerRetriever(chunks, model=FakeEmbeddingModel())

    with pytest.raises(ValueError):
        retriever.search(" ")



def test_semantic_retriever_reuses_persisted_document_embeddings(tmp_path):
    class CountingModel(FakeEmbeddingModel):
        def __init__(self):
            self.encoded_batches = []

        def encode(self, texts, **kwargs):
            self.encoded_batches.append(list(texts))
            return super().encode(texts, **kwargs)

    chunks = [Chunk("A vehicle travels on roads.", "vehicle.txt", 0)]
    model = CountingModel()
    store = SQLiteVectorStore(tmp_path / "index.sqlite3")

    first = SentenceTransformerRetriever(
        chunks,
        model=model,
        store=store,
        index_name="test-documents",
        signature="same-content",
    )
    assert len(model.encoded_batches) == 1

    second = SentenceTransformerRetriever(
        chunks,
        model=model,
        store=store,
        index_name="test-documents",
        signature="same-content",
    )
    assert len(model.encoded_batches) == 1
    assert second.chunks[0].text == first.chunks[0].text
