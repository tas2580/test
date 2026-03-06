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
        self.embedding_endpoints = ("/api/embed", "/api/embeddings")
        self._disabled_models: set[str] = set()
        self._cached_model_candidates: list[str] | None = None

    def embed(self, text: str) -> EmbeddingResult:
        error_messages: list[str] = []

        for model in self._embedding_model_candidates():
            if model in self._disabled_models:
                continue

            for endpoint in self.embedding_endpoints:
                result = self._try_model_endpoint(model=model, endpoint=endpoint, text=text)
                if isinstance(result, EmbeddingResult):
                    return result

                error_messages.extend(result)

                if any("404_ENDPOINT" in entry for entry in result):
                    continue

                if any(tag in entry for entry in result for tag in ("MODEL_NOT_FOUND", "SERVER_5XX", "NO_EMBEDDING")):
                    self._disabled_models.add(model)
                    break

        hint = (
            " Kein nutzbares Embedding-Modell gefunden. "
            "Bitte lokal z. B. 'ollama pull nomic-embed-text' ausführen "
            "und embedding_model/embedding_fallback_model prüfen."
        )
        raise OllamaEmbeddingError("; ".join(error_messages) + hint)

    def _embedding_model_candidates(self) -> list[str]:
        if self._cached_model_candidates is not None:
            return self._cached_model_candidates

        preferred = [
            self.primary_model,
            self.fallback_model,
            "nomic-embed-text",
            "mxbai-embed-large",
            "all-minilm",
        ]
        installed = self._installed_models()

        ordered: list[str] = []
        for model in preferred:
            if model and model not in ordered:
                ordered.append(model)

        for model in installed:
            if model not in ordered and any(token in model.lower() for token in ("embed", "bge", "e5", "minilm")):
                ordered.append(model)

        self._cached_model_candidates = ordered
        return ordered

    def _installed_models(self) -> list[str]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            payload = response.json()
            models = payload.get("models", [])
            names: list[str] = []
            for item in models:
                name = item.get("name")
                if name:
                    names.append(str(name).split(":", 1)[0])
            return names
        except (httpx.HTTPError, ValueError, AttributeError):
            return []

    def _try_model_endpoint(self, model: str, endpoint: str, text: str) -> EmbeddingResult | list[str]:
        errors: list[str] = []
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        f"{self.base_url}{endpoint}",
                        json={"model": model, "input": text, "prompt": text},
                    )
                latency_ms = (time.perf_counter() - started) * 1000

                if response.status_code == 404:
                    body = response.text.lower()
                    if "model" in body or "pull" in body or model.lower() in body:
                        return [f"MODEL_NOT_FOUND: Modell '{model}' nicht gefunden. Bitte 'ollama pull {model}' ausführen."]
                    return [f"404_ENDPOINT: Endpoint {endpoint} nicht verfügbar."]

                if response.status_code >= 500:
                    return [
                        f"SERVER_5XX: Serverfehler für Modell '{model}' an {endpoint}: "
                        f"{response.status_code} {response.text.strip()}"
                    ]

                response.raise_for_status()
                payload = response.json()
                embedding = self._extract_embedding(payload)
                if not embedding:
                    return [f"NO_EMBEDDING: Keine Embedding-Daten von Modell '{model}' an Endpoint {endpoint} erhalten."]
                return EmbeddingResult(embedding=embedding, latency_ms=latency_ms, used_model=model)
            except (httpx.HTTPError, ValueError) as exc:
                errors.append(f"Embedding-Fehler ({model}, {endpoint}, Versuch {attempt + 1}): {exc}")
                if attempt >= self.retries:
                    break

        return errors

    @staticmethod
    def _extract_embedding(payload: dict) -> list[float] | None:
        if "embedding" in payload and payload["embedding"]:
            return payload["embedding"]
        data = payload.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            embedding = data[0].get("embedding")
            if embedding:
                return embedding
        embeddings = payload.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            first = embeddings[0]
            if isinstance(first, list):
                return first
        return None
