"""Optional SQLite-vec backed vector store.

Requires: pip install imprint-mem[vector]
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import aiosqlite

if TYPE_CHECKING:
    pass

_VEC_INDEX_SQL = """
CREATE TABLE IF NOT EXISTS vec_index (
    memory_id  TEXT PRIMARY KEY,
    vec_rowid  INTEGER NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS vec_rowid_seq (
    id  INTEGER PRIMARY KEY AUTOINCREMENT
);
"""


class SQLiteVecStore:
    """sqlite-vec backed vector store.

    Embedding dimension must match whatever Embedder is in use -- mismatch
    raises at insert time. The vec0 virtual table uses integer rowids, so a
    companion table maps memory_id -> rowid.

    Requires: pip install imprint-mem[vector]
    """

    def __init__(self, conn: aiosqlite.Connection, dim: int) -> None:
        self._conn = conn
        self._dim = dim
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import sqlite_vec  # type: ignore[import-untyped]
        except ImportError as e:
            missing = getattr(e, "name", None)
            if missing == "sqlite_vec" or missing is None:
                raise ImportError(
                    "sqlite-vec is required for SQLiteVecStore; "
                    "install it with: pip install imprint-mem[vector]"
                ) from e
            raise ImportError(
                f"SQLiteVecStore failed to import sqlite-vec: missing transitive "
                f"dependency '{missing}'. Try: pip install imprint-mem[vector]"
            ) from e
        await self._conn.enable_load_extension(True)
        await self._conn.execute(
            "SELECT load_extension(?)",
            [sqlite_vec.loadable_path()],  # type: ignore[union-attr]
        )
        await self._conn.enable_load_extension(False)
        await self._conn.executescript(_VEC_INDEX_SQL)
        await self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings "
            f"USING vec0(embedding FLOAT[{self._dim}])"
        )
        await self._conn.commit()
        self._loaded = True

    async def upsert(self, id: str, embedding: list[float]) -> None:
        await self._ensure_loaded()
        if len(embedding) != self._dim:
            raise ValueError(f"embedding dim {len(embedding)} does not match store dim {self._dim}")
        serialized = _serialize(embedding)
        cursor = await self._conn.execute(
            "SELECT vec_rowid FROM vec_index WHERE memory_id = ?", (id,)
        )
        row = await cursor.fetchone()
        if row is not None:
            await self._conn.execute(
                "UPDATE vec_embeddings SET embedding = ? WHERE rowid = ?",
                (serialized, row[0]),
            )
        else:
            await self._conn.execute("INSERT INTO vec_rowid_seq VALUES (NULL)")
            cursor = await self._conn.execute("SELECT last_insert_rowid()")
            seq_row = await cursor.fetchone()
            assert seq_row is not None
            rowid = seq_row[0]
            await self._conn.execute(
                "INSERT INTO vec_embeddings(rowid, embedding) VALUES (?, ?)",
                (rowid, serialized),
            )
            await self._conn.execute(
                "INSERT INTO vec_index(memory_id, vec_rowid) VALUES (?, ?)",
                (id, rowid),
            )
        await self._conn.commit()

    async def search(self, embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        await self._ensure_loaded()
        serialized = _serialize(embedding)
        cursor = await self._conn.execute(
            "SELECT vi.memory_id, ve.distance "
            "FROM vec_embeddings ve "
            "JOIN vec_index vi ON ve.rowid = vi.vec_rowid "
            "WHERE ve.embedding MATCH ? AND ve.k = ?",
            (serialized, top_k),
        )
        rows = await cursor.fetchall()
        return [(row[0], row[1]) for row in rows]

    async def delete(self, id: str) -> None:
        await self._ensure_loaded()
        cursor = await self._conn.execute(
            "SELECT vec_rowid FROM vec_index WHERE memory_id = ?", (id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return
        rowid = row[0]
        await self._conn.execute("DELETE FROM vec_embeddings WHERE rowid = ?", (rowid,))
        await self._conn.execute("DELETE FROM vec_index WHERE memory_id = ?", (id,))
        await self._conn.commit()


def _serialize(embedding: list[float]) -> bytes:
    try:
        import sqlite_vec  # type: ignore[import-untyped]

        return sqlite_vec.serialize_float32(embedding)  # type: ignore[no-any-return]
    except ImportError:
        return struct.pack(f"{len(embedding)}f", *embedding)
