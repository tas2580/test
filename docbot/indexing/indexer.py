from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from docbot.indexing.chunkers import chunk_csv_file, chunk_txt_file
from docbot.indexing.chroma_store import ChromaStore
from docbot.indexing.embedder import OllamaEmbedder, OllamaEmbeddingError
from docbot.indexing.loader import collect_file_info, iter_doc_files
from docbot.indexing.meta_db import MetaDB

logger = logging.getLogger(__name__)


class Indexer:
    def __init__(self, db: MetaDB, store: ChromaStore, embedder: OllamaEmbedder):
        self.db = db
        self.store = store
        self.embedder = embedder

    def index_path(
        self,
        docs_path: Path,
        rebuild: bool = False,
        recursive: bool = False,
        txt_max_chars: int = 2000,
        csv_group_size: int = 50,
    ) -> dict:
        files = list(iter_doc_files(docs_path, recursive=recursive))
        stats = {"files_total": len(files), "files_indexed": 0, "files_skipped": 0, "chunks": 0, "errors": 0}

        logger.info("Indexierung gestartet", extra={"docs_path": str(docs_path), "files_total": len(files)})

        for file_path in files:
            try:
                info = collect_file_info(file_path)
                existing = self.db.get_file_record(str(file_path))
                unchanged = (
                    existing
                    and not rebuild
                    and existing.mtime == info.mtime
                    and existing.size == info.size
                    and existing.file_hash == info.file_hash
                )

                if unchanged:
                    stats["files_skipped"] += 1
                    logger.info("Datei unverändert, übersprungen", extra={"file": str(file_path)})
                    continue

                old_chunks = self.db.get_chunks_by_file(str(file_path))
                if old_chunks:
                    self.store.delete_chunks(old_chunks)
                    self.db.delete_file_and_chunks(str(file_path))

                if file_path.suffix.lower() == ".txt":
                    chunks = chunk_txt_file(file_path, max_chars=txt_max_chars)
                else:
                    chunks = chunk_csv_file(file_path, group_size=csv_group_size)

                indexed_at = datetime.now(timezone.utc).isoformat()
                file_chunk_count = 0
                for chunk in chunks:
                    result = self.embedder.embed(chunk.text)
                    self.store.upsert_chunk(chunk.chunk_id, chunk.text, chunk.metadata, result.embedding)
                    self.db.add_chunk(chunk.chunk_id, str(file_path), chunk.metadata, indexed_at)
                    file_chunk_count += 1
                    stats["chunks"] += 1
                    logger.info(
                        "Chunk indexiert",
                        extra={
                            "file": str(file_path),
                            "chunk_id": chunk.chunk_id,
                            "latency_ms": round(result.latency_ms, 2),
                            "embedding_model": result.used_model,
                        },
                    )

                self.db.upsert_file(str(file_path), info.file_hash, info.mtime, info.size, indexed_at)
                stats["files_indexed"] += 1
                logger.info(
                    "Datei indexiert",
                    extra={"file": str(file_path), "chunk_count": file_chunk_count, "indexed_at": indexed_at},
                )
            except OllamaEmbeddingError as exc:
                stats["errors"] += 1
                logger.exception("Embedding fehlgeschlagen", extra={"file": str(file_path), "error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                stats["errors"] += 1
                logger.exception("Indexierungsfehler", extra={"file": str(file_path), "error": str(exc)})

        logger.info("Indexierung beendet", extra=stats)
        return stats
