import pytest

pytest.importorskip("httpx")

from docbot.indexing.embedder import EmbeddingResult, OllamaEmbedder


class _ScriptedEmbedder(OllamaEmbedder):
    def __init__(
        self,
        *args,
        scripted_results: dict[tuple[str, str], EmbeddingResult | list[str]],
        installed_models: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scripted_results = scripted_results
        self.calls: list[tuple[str, str]] = []
        self._installed_models_for_test = installed_models or []

    def _try_model_endpoint(self, model: str, endpoint: str, text: str):
        self.calls.append((model, endpoint))
        return self.scripted_results[(model, endpoint)]

    def _installed_models(self) -> list[str]:
        return self._installed_models_for_test


def test_embedder_switches_model_immediately_on_server_5xx() -> None:
    scripted = {
        ("gemma3", "/api/embed"): ["SERVER_5XX: 500"],
        ("nomic-embed-text", "/api/embed"): EmbeddingResult([0.1, 0.2], 1.0, "nomic-embed-text"),
    }
    embedder = _ScriptedEmbedder("http://test", "gemma3", "nomic-embed-text", scripted_results=scripted)

    result = embedder.embed("hallo")

    assert result.used_model == "nomic-embed-text"
    assert embedder.calls == [
        ("gemma3", "/api/embed"),
        ("nomic-embed-text", "/api/embed"),
    ]


def test_embedder_uses_installed_embedding_model_if_fallback_missing() -> None:
    scripted = {
        ("gemma3", "/api/embed"): ["SERVER_5XX: 500"],
        ("nomic-embed-text", "/api/embed"): ["MODEL_NOT_FOUND: missing"],
        ("mxbai-embed-large", "/api/embed"): EmbeddingResult([0.3, 0.4], 1.0, "mxbai-embed-large"),
    }
    embedder = _ScriptedEmbedder(
        "http://test",
        "gemma3",
        "nomic-embed-text",
        scripted_results=scripted,
        installed_models=["gemma3", "mxbai-embed-large"],
    )

    result = embedder.embed("hallo")

    assert result.used_model == "mxbai-embed-large"
    assert embedder.calls == [
        ("gemma3", "/api/embed"),
        ("nomic-embed-text", "/api/embed"),
        ("mxbai-embed-large", "/api/embed"),
    ]


def test_extract_embedding_supports_multiple_payload_shapes() -> None:
    embedder = OllamaEmbedder("http://x", "a", "b")
    assert embedder._extract_embedding({"embedding": [1.0]}) == [1.0]
    assert embedder._extract_embedding({"data": [{"embedding": [2.0]}]}) == [2.0]
    assert embedder._extract_embedding({"embeddings": [[3.0]]}) == [3.0]
