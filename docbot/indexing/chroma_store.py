from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection


class ChromaStore:
    def __init__(self, persist_dir: Path, collection_name: str = "docbot_chunks"):
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection: Collection = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    def upsert_chunk(self, chunk_id: str, text: str, metadata: dict[str, Any], embedding: list[float]) -> None:
        self.collection.upsert(ids=[chunk_id], documents=[text], metadatas=[metadata], embeddings=[embedding])

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            self.collection.delete(ids=chunk_ids)

    def query(self, query_embedding: list[float], top_k: int = 8) -> dict[str, Any]:
        return self.collection.query(query_embeddings=[query_embedding], n_results=top_k, include=["documents", "metadatas", "distances"])

    def count(self) -> int:
        return self.collection.count()
