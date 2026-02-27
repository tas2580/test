import pytest

httpx = pytest.importorskip("httpx")

from docbot.indexing.embedder import EmbeddingResult, OllamaEmbedder


class _PatchedEmbedder(OllamaEmbedder):
    def __init__(self, *args, transport: httpx.BaseTransport, **kwargs):
        super().__init__(*args, **kwargs)
        self._transport = transport

    def _try_model_endpoint(self, model: str, endpoint: str, text: str):
        errors: list[str] = []
        for attempt in range(self.retries + 1):
            try:
                with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
                    response = client.post(
                        f"{self.base_url}{endpoint}",
                        json={"model": model, "input": text, "prompt": text},
                    )
                if response.status_code == 404:
                    body = response.text.lower()
                    if "model" in body or "pull" in body or model.lower() in body:
                        return [f"MODEL_NOT_FOUND: Modell '{model}' nicht gefunden."]
                    return [f"404_ENDPOINT: Endpoint {endpoint} nicht verfügbar."]
                if response.status_code >= 500:
                    return [f"Serverfehler für Modell '{model}' an {endpoint}: {response.status_code}"]
                response.raise_for_status()
                payload = response.json()
                embedding = self._extract_embedding(payload)
                if embedding:
                    return EmbeddingResult(embedding=embedding, latency_ms=1.0, used_model=model)
                return ["Keine Embedding-Daten"]
            except (httpx.HTTPError, ValueError) as exc:
                errors.append(str(exc))
                if attempt >= self.retries:
                    break
        return errors


def test_embedder_fallback_to_second_model_on_500() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode("utf-8")
        if '"model":"gemma3"' in body:
            return httpx.Response(500, text="model does not support embeddings")
        return httpx.Response(200, json={"embedding": [0.1, 0.2]})

    transport = httpx.MockTransport(handler)
    embedder = _PatchedEmbedder(
        base_url="http://test",
        primary_model="gemma3",
        fallback_model="nomic-embed-text",
        retries=0,
        transport=transport,
    )

    result = embedder.embed("hallo")
    assert result.used_model == "nomic-embed-text"
    assert result.embedding == [0.1, 0.2]


def test_extract_embedding_supports_multiple_payload_shapes() -> None:
    embedder = OllamaEmbedder("http://x", "a", "b")
    assert embedder._extract_embedding({"embedding": [1.0]}) == [1.0]
    assert embedder._extract_embedding({"data": [{"embedding": [2.0]}]}) == [2.0]
    assert embedder._extract_embedding({"embeddings": [[3.0]]}) == [3.0]
