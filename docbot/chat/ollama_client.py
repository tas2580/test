from __future__ import annotations

import time

import httpx


class OllamaChatError(RuntimeError):
    pass


class OllamaChatClient:
    def __init__(self, base_url: str, model: str, timeout: float = 60.0, retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries

    def chat(self, system_prompt: str, user_prompt: str) -> tuple[str, float]:
        errors: list[str] = []
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        f"{self.base_url}/api/chat",
                        json={
                            "model": self.model,
                            "stream": False,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                        },
                    )
                latency_ms = (time.perf_counter() - started) * 1000

                if response.status_code == 404:
                    raise OllamaChatError(f"Modell '{self.model}' nicht vorhanden. Bitte 'ollama pull {self.model}' ausführen.")

                response.raise_for_status()
                payload = response.json()
                message = payload.get("message", {})
                content = message.get("content", "").strip()
                if not content:
                    raise OllamaChatError("Leere Chat-Antwort von Ollama erhalten.")
                return content, latency_ms
            except (httpx.HTTPError, ValueError, OllamaChatError) as exc:
                errors.append(f"Versuch {attempt + 1}: {exc}")
                if attempt >= self.retries:
                    break
        raise OllamaChatError("; ".join(errors))
