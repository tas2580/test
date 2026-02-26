from __future__ import annotations

import time
from dataclasses import dataclass

import httpx


class OllamaEmbeddingError(RuntimeError):
    pass


@dataclass
class EmbeddingResult:
    embedding: list[float]
    latency_ms: float
    used_model: str


class OllamaEmbedder:
    def __init__(
        self,
        base_url: str,
        primary_model: str,
        fallback_model: str,
        timeout: float = 30.0,
        retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.timeout = timeout
        self.retries = retries

    def embed(self, text: str) -> EmbeddingResult:
        error_messages: list[str] = []

        for model in [self.primary_model, self.fallback_model]:
            for attempt in range(self.retries + 1):
                started = time.perf_counter()
                try:
                    with httpx.Client(timeout=self.timeout) as client:
                        response = client.post(
                            f"{self.base_url}/api/embeddings",
                            json={"model": model, "prompt": text},
                        )
                    latency_ms = (time.perf_counter() - started) * 1000

                    if response.status_code == 404:
                        error_messages.append(
                            f"Modell '{model}' nicht gefunden. Bitte 'ollama pull {model}' ausführen."
                        )
                        break
                    response.raise_for_status()
                    payload = response.json()
                    embedding = payload.get("embedding")
                    if not embedding:
                        error_messages.append(f"Keine Embedding-Daten von Modell '{model}' erhalten.")
                        break
                    return EmbeddingResult(embedding=embedding, latency_ms=latency_ms, used_model=model)
                except (httpx.HTTPError, ValueError) as exc:
                    error_messages.append(f"Embedding-Fehler ({model}, Versuch {attempt + 1}): {exc}")
                    if attempt >= self.retries:
                        break

        raise OllamaEmbeddingError("; ".join(error_messages))
