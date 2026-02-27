from docbot.indexing.embedder import EmbeddingResult, OllamaEmbedder


class _ScriptedEmbedder(OllamaEmbedder):
    def __init__(self, *args, scripted_results: dict[tuple[str, str], EmbeddingResult | list[str]], **kwargs):
        super().__init__(*args, **kwargs)
        self.scripted_results = scripted_results
        self.calls: list[tuple[str, str]] = []

    def _try_model_endpoint(self, model: str, endpoint: str, text: str):
        self.calls.append((model, endpoint))
        return self.scripted_results[(model, endpoint)]


def test_embedder_switches_model_immediately_on_server_5xx() -> None:
    scripted = {
        ("gemma3", "/api/embed"): ["SERVER_5XX: 500"],
        ("gemma3", "/api/embeddings"): ["SHOULD_NOT_BE_CALLED"],
        ("nomic-embed-text", "/api/embed"): EmbeddingResult([0.1, 0.2], 1.0, "nomic-embed-text"),
        ("nomic-embed-text", "/api/embeddings"): ["SHOULD_NOT_BE_CALLED"],
    }
    embedder = _ScriptedEmbedder("http://test", "gemma3", "nomic-embed-text", scripted_results=scripted)

    result = embedder.embed("hallo")

    assert result.used_model == "nomic-embed-text"
    assert embedder.calls == [
        ("gemma3", "/api/embed"),
        ("nomic-embed-text", "/api/embed"),
    ]


def test_extract_embedding_supports_multiple_payload_shapes() -> None:
    embedder = OllamaEmbedder("http://x", "a", "b")
    assert embedder._extract_embedding({"embedding": [1.0]}) == [1.0]
    assert embedder._extract_embedding({"data": [{"embedding": [2.0]}]}) == [2.0]
    assert embedder._extract_embedding({"embeddings": [[3.0]]}) == [3.0]
