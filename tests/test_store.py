import numpy as np

from localrag import Chunk
from localrag.store import SQLiteVectorStore


def test_sqlite_store_round_trips_chunks_and_embeddings(tmp_path):
    store = SQLiteVectorStore(tmp_path / "cache.sqlite3")
    chunks = [
        Chunk("A document about Python.", "python.md", 0, {"section": "intro"}),
        Chunk("A document about search.", "search.md", 1, {"page": 2}),
    ]
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    store.save("knowledge", "signature-1", "test-model", chunks, embeddings)
    cached = store.load("knowledge", "signature-1")

    assert cached is not None
    cached_chunks, cached_embeddings = cached
    assert [chunk.text for chunk in cached_chunks] == [chunk.text for chunk in chunks]
    assert cached_chunks[0].metadata == {"section": "intro"}
    np.testing.assert_array_equal(cached_embeddings, embeddings)


def test_sqlite_store_does_not_return_stale_signature(tmp_path):
    store = SQLiteVectorStore(tmp_path / "cache.sqlite3")
    chunks = [Chunk("hello world", "hello.txt", 0)]
    embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)

    store.save("knowledge", "old-signature", "test-model", chunks, embeddings)

    assert store.load("knowledge", "new-signature") is None


def test_sqlite_store_replaces_existing_index_atomically(tmp_path):
    store = SQLiteVectorStore(tmp_path / "cache.sqlite3")
    original = [Chunk("old content", "notes.txt", 0)]
    updated = [Chunk("new content", "notes.txt", 0)]
    store.save(
        "knowledge",
        "signature-1",
        "test-model",
        original,
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    store.save(
        "knowledge",
        "signature-2",
        "test-model",
        updated,
        np.asarray([[0.0, 1.0]], dtype=np.float32),
    )

    assert store.load("knowledge", "signature-1") is None
    cached = store.load("knowledge", "signature-2")
    assert cached is not None
    assert cached[0][0].text == "new content"
