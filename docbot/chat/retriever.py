from __future__ import annotations

from dataclasses import dataclass

from docbot.indexing.chroma_store import ChromaStore
from docbot.indexing.embedder import OllamaEmbedder


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    score: float


class Retriever:
    def __init__(self, store: ChromaStore, embedder: OllamaEmbedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed(query).embedding
        result = self.store.query(query_embedding, top_k=top_k)

        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        chunks: list[RetrievedChunk] = []
        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists, strict=False):
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=meta or {},
                    score=float(dist) if dist is not None else 0.0,
                )
            )
        return chunks
