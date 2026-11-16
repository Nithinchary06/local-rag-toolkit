import pytest
from localrag import Document, chunk_documents


def test_chunking_preserves_metadata_and_overlap():
    document = Document("one two three four five six", "notes.txt", {"page": 2})
    chunks = chunk_documents([document], chunk_size=4, overlap=2)
    assert [chunk.text for chunk in chunks] == ["one two three four", "three four five six", "five six"]
    assert chunks[0].source == "notes.txt"
    assert chunks[0].metadata["page"] == 2


def test_chunking_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_documents([Document("hello", "a.txt")], chunk_size=3, overlap=3)
