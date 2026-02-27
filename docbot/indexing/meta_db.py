from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
  file_path TEXT PRIMARY KEY,
  file_hash TEXT NOT NULL,
  mtime REAL NOT NULL,
  size INTEGER NOT NULL,
  last_indexed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  file_path TEXT NOT NULL,
  chunk_type TEXT NOT NULL,
  paragraph_index INTEGER,
  line_start INTEGER,
  line_end INTEGER,
  created_at TEXT NOT NULL,
  FOREIGN KEY(file_path) REFERENCES files(file_path) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path);
"""


@dataclass
class FileRecord:
    file_path: str
    file_hash: str
    mtime: float
    size: int


class MetaDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def get_file_record(self, file_path: str) -> FileRecord | None:
        row = self.conn.execute(
            "SELECT file_path, file_hash, mtime, size FROM files WHERE file_path = ?", (file_path,)
        ).fetchone()
        if not row:
            return None
        return FileRecord(
            file_path=row["file_path"],
            file_hash=row["file_hash"],
            mtime=row["mtime"],
            size=row["size"],
        )

    def upsert_file(self, file_path: str, file_hash: str, mtime: float, size: int, indexed_at: str) -> None:
        self.conn.execute(
            """
            INSERT INTO files(file_path, file_hash, mtime, size, last_indexed_at)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
              file_hash=excluded.file_hash,
              mtime=excluded.mtime,
              size=excluded.size,
              last_indexed_at=excluded.last_indexed_at
            """,
            (file_path, file_hash, mtime, size, indexed_at),
        )
        self.conn.commit()

    def add_chunk(self, chunk_id: str, file_path: str, metadata: dict[str, Any], created_at: str) -> None:
        self.conn.execute(
            """
            INSERT INTO chunks(chunk_id, file_path, chunk_type, paragraph_index, line_start, line_end, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
              file_path=excluded.file_path,
              chunk_type=excluded.chunk_type,
              paragraph_index=excluded.paragraph_index,
              line_start=excluded.line_start,
              line_end=excluded.line_end,
              created_at=excluded.created_at
            """,
            (
                chunk_id,
                file_path,
                metadata.get("chunk_type", "unknown"),
                metadata.get("paragraph_index"),
                metadata.get("line_start"),
                metadata.get("line_end"),
                created_at,
            ),
        )
        self.conn.commit()

    def get_chunks_by_file(self, file_path: str) -> list[str]:
        rows = self.conn.execute("SELECT chunk_id FROM chunks WHERE file_path = ?", (file_path,)).fetchall()
        return [row["chunk_id"] for row in rows]

    def delete_file_and_chunks(self, file_path: str) -> None:
        self.conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
        self.conn.execute("DELETE FROM files WHERE file_path = ?", (file_path,))
        self.conn.commit()

    def status(self) -> dict[str, Any]:
        file_count = self.conn.execute("SELECT COUNT(*) AS c FROM files").fetchone()["c"]
        chunk_count = self.conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
        last_run = self.conn.execute("SELECT MAX(last_indexed_at) AS m FROM files").fetchone()["m"]
        return {"files": file_count, "chunks": chunk_count, "last_indexed_at": last_run}
