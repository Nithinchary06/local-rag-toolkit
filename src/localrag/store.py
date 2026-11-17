"""SQLite persistence for reusable semantic vector indexes."""

import json
import sqlite3
from pathlib import Path

import numpy as np

from .models import Chunk


class SQLiteVectorStore:
    """Persist chunk text, metadata, and embedding vectors in a local SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_indexes (
                    index_name TEXT PRIMARY KEY,
                    signature TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    index_name TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    PRIMARY KEY (index_name, position),
                    FOREIGN KEY (index_name)
                        REFERENCES rag_indexes(index_name) ON DELETE CASCADE
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def load(
        self, index_name: str, signature: str
    ) -> tuple[list[Chunk], np.ndarray] | None:
        """Return a matching cached index, or None when it is absent or stale."""
        with self._connect() as connection:
            index = connection.execute(
                """
                SELECT dimension, chunk_count
                FROM rag_indexes
                WHERE index_name = ? AND signature = ?
                """,
                (index_name, signature),
            ).fetchone()
            if index is None:
                return None

            dimension, expected_count = index
            rows = connection.execute(
                """
                SELECT chunk_id, source, text, metadata_json, embedding
                FROM rag_chunks
                WHERE index_name = ?
                ORDER BY position
                """,
                (index_name,),
            ).fetchall()

        if len(rows) != expected_count:
            return None

        chunks: list[Chunk] = []
        vectors: list[np.ndarray] = []
        try:
            for chunk_id, source, text, metadata_json, embedding in rows:
                vector = np.frombuffer(embedding, dtype=np.float32)
                if vector.size != dimension:
                    return None
                chunks.append(
                    Chunk(
                        text=text,
                        source=source,
                        chunk_id=chunk_id,
                        metadata=json.loads(metadata_json),
                    )
                )
                vectors.append(vector.copy())
        except (ValueError, TypeError, json.JSONDecodeError):
            return None

        if not vectors:
            return None
        return chunks, np.vstack(vectors)

    def save(
        self,
        index_name: str,
        signature: str,
        model_name: str,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> None:
        """Atomically replace an index with the supplied chunks and vectors."""
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("Embeddings must be a 2D matrix with one row per chunk")

        with self._connect() as connection:
            connection.execute("BEGIN")
            connection.execute(
                "DELETE FROM rag_indexes WHERE index_name = ?", (index_name,)
            )
            connection.execute(
                """
                INSERT INTO rag_indexes
                    (index_name, signature, model_name, dimension, chunk_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (index_name, signature, model_name, vectors.shape[1], len(chunks)),
            )
            connection.executemany(
                """
                INSERT INTO rag_chunks
                    (index_name, position, chunk_id, source, text, metadata_json, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        index_name,
                        position,
                        chunk.chunk_id,
                        chunk.source,
                        chunk.text,
                        json.dumps(chunk.metadata, sort_keys=True, default=str),
                        vectors[position].tobytes(),
                    )
                    for position, chunk in enumerate(chunks)
                ],
            )
            connection.commit()
