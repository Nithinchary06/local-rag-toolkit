import pytest

from localrag import Chunk, TfidfRetriever


def test_retriever_ranks_relevant_chunk_first():
    chunks = [Chunk("Python is a programming language.", "python.txt", 0), Chunk("Bananas are a source of potassium.", "fruit.txt", 1)]
    retriever = TfidfRetriever(chunks)
    results = retriever.search("Python programming", top_k=2)
    assert results[0].chunk.source == "python.txt"
    assert results[0].score > 0


def test_retriever_validates_query():
    retriever = TfidfRetriever([Chunk("sample text", "sample.txt", 0)])
    with pytest.raises(ValueError):
        retriever.search("   ")
