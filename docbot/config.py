from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Settings:
    docs_path: str = "./data/docs"
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "gemma3"
    embedding_model: str = "gemma3"
    embedding_fallback_model: str = "nomic-embed-text"
    chroma_persist_dir: str = "./data/chroma"
    sqlite_path: str = "./data/docbot.db"
    top_k: int = 8
    max_context_chunks: int = 5
    txt_max_chars: int = 2000
    csv_group_size: int = 50
    answer_mode: str = "short"
    log_level: str = "INFO"
    log_dir: str = "./logs"
    chat_log_path: str = "./logs/chat.jsonl"

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir)

    @property
    def sqlite_file(self) -> Path:
        return Path(self.sqlite_path)


def _flatten(prefix: str, value: Any, target: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, inner in value.items():
            new_prefix = f"{prefix}_{key}" if prefix else key
            _flatten(new_prefix, inner, target)
    else:
        target[prefix] = value


def load_settings(config_path: str | None = None, **overrides: Any) -> Settings:
    settings = Settings()
    merged: dict[str, Any] = {}

    candidate = Path(config_path or "config.toml")
    if candidate.exists():
        with candidate.open("rb") as handle:
            data = tomllib.load(handle)
        _flatten("", data, merged)

    for key, value in overrides.items():
        if value is not None:
            merged[key] = value

    valid = {field: merged[field] for field in settings.__dataclass_fields__ if field in merged}
    if valid:
        settings = replace(settings, **valid)

    Path(settings.log_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.chat_log_path).parent.mkdir(parents=True, exist_ok=True)

    return settings
