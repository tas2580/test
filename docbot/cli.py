from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from pythonjsonlogger import jsonlogger
from rich.console import Console
from rich.table import Table

from docbot.chat.ollama_client import OllamaChatClient
from docbot.chat.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from docbot.chat.retriever import Retriever
from docbot.chat.session_logger import SessionLogger
from docbot.config import load_settings
from docbot.indexing.chroma_store import ChromaStore
from docbot.indexing.embedder import OllamaEmbedder
from docbot.indexing.indexer import Indexer
from docbot.indexing.meta_db import MetaDB

app = typer.Typer(help="Offline-Doku-Chatbot für TXT/CSV mit Ollama + Chroma")
console = Console()


def setup_logging(log_dir: str, level: str) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)

    json_handler = logging.FileHandler(Path(log_dir) / "docbot.log.jsonl", encoding="utf-8")
    json_handler.setFormatter(jsonlogger.JsonFormatter())

    root.addHandler(console_handler)
    root.addHandler(json_handler)

    # HTTP-Request-Noise reduzieren; relevante Fehler werden weiterhin explizit geloggt.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _services(config_path: Optional[str] = None, **overrides):
    settings = load_settings(config_path, **overrides)
    setup_logging(settings.log_dir, settings.log_level)

    db = MetaDB(settings.sqlite_file)
    store = ChromaStore(settings.chroma_path)
    embedder = OllamaEmbedder(
        base_url=settings.ollama_base_url,
        primary_model=settings.embedding_model,
        fallback_model=settings.embedding_fallback_model,
    )
    return settings, db, store, embedder


@app.command()
def index(
    path: str = typer.Option(..., "--path", help="Pfad zu Dokumenten"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Komplett neu indexieren"),
    recursive: bool = typer.Option(False, "--recursive", help="Unterordner rekursiv einbeziehen"),
    csv_group_size: Optional[int] = typer.Option(None, "--csv-group-size"),
    txt_max_chars: Optional[int] = typer.Option(None, "--txt-max-chars"),
    config: Optional[str] = typer.Option(None, "--config"),
) -> None:
    settings, db, store, embedder = _services(config)
    indexer = Indexer(db, store, embedder)

    stats = indexer.index_path(
        docs_path=Path(path),
        rebuild=rebuild,
        recursive=recursive,
        txt_max_chars=txt_max_chars or settings.txt_max_chars,
        csv_group_size=csv_group_size or settings.csv_group_size,
    )
    db.close()
    console.print("[green]Indexierung abgeschlossen[/green]")
    console.print_json(data=stats)


@app.command()
def ask(
    question: str,
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    short: bool = typer.Option(False, "--short"),
    long: bool = typer.Option(False, "--long"),
    config: Optional[str] = typer.Option(None, "--config"),
) -> None:
    mode = "long" if long else "short"
    if short:
        mode = "short"

    settings, db, store, embedder = _services(config)
    retriever = Retriever(store, embedder)
    chat_client = OllamaChatClient(settings.ollama_base_url, settings.chat_model)
    chat_logger = SessionLogger(Path(settings.chat_log_path))

    retrieved = retriever.retrieve(question, top_k=top_k or settings.top_k)
    context_chunks = retrieved[: settings.max_context_chunks]
    user_prompt = build_user_prompt(question, context_chunks, mode=mode)
    answer, latency_ms = chat_client.chat(SYSTEM_PROMPT, user_prompt)

    payload = {
        "mode": "ask",
        "query": question,
        "top_k": top_k or settings.top_k,
        "retrieved_chunks": [chunk.chunk_id for chunk in context_chunks],
        "retrieval_scores": {chunk.chunk_id: chunk.score for chunk in context_chunks},
        "prompt": user_prompt,
        "completion": answer,
        "latency_ms": latency_ms,
    }
    chat_logger.log(payload)

    console.print(answer)
    db.close()


@app.command()
def chat(
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    max_context_chunks: Optional[int] = typer.Option(None, "--max-context-chunks"),
    short: bool = typer.Option(False, "--short"),
    long: bool = typer.Option(False, "--long"),
    config: Optional[str] = typer.Option(None, "--config"),
) -> None:
    mode = "long" if long else "short"
    if short:
        mode = "short"

    settings, db, store, embedder = _services(config)
    retriever = Retriever(store, embedder)
    chat_client = OllamaChatClient(settings.ollama_base_url, settings.chat_model)
    chat_logger = SessionLogger(Path(settings.chat_log_path))

    top_k = top_k or settings.top_k
    max_context_chunks = max_context_chunks or settings.max_context_chunks

    console.print("[bold cyan]Docbot Chat gestartet[/bold cyan] (beenden mit /exit)")
    while True:
        question = typer.prompt("Du")
        if question.strip().lower() in {"/exit", "exit", "quit", "/quit"}:
            break

        retrieved = retriever.retrieve(question, top_k=top_k)
        context_chunks = retrieved[:max_context_chunks]
        user_prompt = build_user_prompt(question, context_chunks, mode=mode)
        answer, latency_ms = chat_client.chat(SYSTEM_PROMPT, user_prompt)
        console.print(f"\n[bold green]Docbot:[/bold green] {answer}\n")

        chat_logger.log(
            {
                "mode": "chat",
                "query": question,
                "top_k": top_k,
                "max_context_chunks": max_context_chunks,
                "retrieved_chunks": [chunk.chunk_id for chunk in context_chunks],
                "retrieval_scores": {chunk.chunk_id: chunk.score for chunk in context_chunks},
                "prompt": user_prompt,
                "completion": answer,
                "latency_ms": latency_ms,
            }
        )

    db.close()


@app.command()
def status(config: Optional[str] = typer.Option(None, "--config")) -> None:
    settings, db, store, _embedder = _services(config)
    status_data = db.status()
    table = Table(title="Docbot Status")
    table.add_column("Feld")
    table.add_column("Wert")
    table.add_row("Dateien indexiert", str(status_data["files"]))
    table.add_row("Chunks indexiert", str(status_data["chunks"]))
    table.add_row("Letzter Lauf", str(status_data["last_indexed_at"]))
    table.add_row("Chroma Pfad", settings.chroma_persist_dir)
    table.add_row("SQLite Pfad", settings.sqlite_path)
    table.add_row("Chat Modell", settings.chat_model)
    table.add_row("Embedding Modell", settings.embedding_model)
    table.add_row("Chunks in Chroma", str(store.count()))
    console.print(table)
    db.close()


@app.command()
def architecture() -> None:
    """Gibt Architektur und Datenfluss als JSON aus."""
    payload = {
        "flow": [
            "Dateien (TXT/CSV) -> Loader + Hashing",
            "Chunking (Absatz-/Zeilengruppen) -> Embeddings via Ollama",
            "Upsert in Chroma + Chunk/File-Metadaten in SQLite",
            "Query -> Embedding -> Retrieval Top-K -> Prompting -> Antwort inkl. Quellen",
            "Session- und System-Logging als JSONL",
        ]
    }
    console.print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
