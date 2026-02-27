# Docbot (Offline CLI Doku-Chatbot)

Docbot indexiert lokale `TXT`- und `CSV`-Dateien (Semikolon, UTF-8) und macht sie über ein Chat-CLI semantisch abfragbar.

- **LLM/Embeddings:** lokal über Ollama
- **Antwortmodell (Default):** `gemma3`
- **Vektor-Store:** Chroma (persistent auf Disk)
- **Metadaten/State:** SQLite
- **Schnittstelle:** CLI (Typer)
- **Sprache:** strikt Deutsch

## Architektur & Datenfluss

1. **Indexierung (`docbot index`)**
   - Dateienscan (`.txt`, `.csv`) im Zielordner.
   - Änderungsprüfung über `mtime`, `size`, `sha256`.
   - TXT-Chunking: absatzbasiert, lange Absätze werden unterteilt (`txt_max_chars`).
   - CSV-Chunking: Header + gruppierte Datenzeilen (`csv_group_size`) inkl. Zeilenbereich.
   - Pro Chunk: Embedding via Ollama, Upsert in Chroma, Metadaten in SQLite.

2. **Abfrage (`docbot ask` / `docbot chat`)**
   - Query-Embedding über Ollama.
   - Semantische Suche in Chroma (`top_k`), dann Begrenzung auf `max_context_chunks`.
   - Prompting an `gemma3` mit strikten Regeln (nur Kontext, Rückfragen bei Unklarheit, Quellenpflicht).
   - Antwort + Quellenabschnitt.
   - Query-/Retrieval-/Prompt-/Antwort-Logging in JSONL.

3. **RAM-schonend**
   - Dateien werden streamend gelesen.
   - Chunk-Erzeugung als Generator.
   - Embeddings/Upserts nacheinander pro Chunk.
   - Prompt-Kontext begrenzt (`max_context_chunks`, Excerpt-Trimming).

## Voraussetzungen

- Debian Linux (offline nutzbar)
- Python 3.11+
- Lokales Ollama (`http://localhost:11434`)
- Modelle lokal vorhanden:
  ```bash
  ollama pull gemma3
  # optionaler Embedding-Fallback
  ollama pull nomic-embed-text
  ```

## Installation lokal

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
cp config.example.toml config.toml
```

## Konfiguration

Datei: `config.toml` (Beispiel siehe `config.example.toml`)

Wichtige Defaults:
- `top_k = 8`
- `max_context_chunks = 5`
- `csv_group_size = 50`
- `txt_max_chars = 2000`
- `answer_mode = "short"`

## Nutzung

### 1) Index aufbauen

```bash
docbot index --path /data/docs
```

Optionen:
- `--rebuild`: vollständiger Neuaufbau
- `--recursive`: rekursiv durch Unterordner
- `--csv-group-size N`
- `--txt-max-chars M`

### 2) Einzelfrage (skriptbar)

```bash
docbot ask "Welche Risiken wurden im letzten Audit genannt?" --top-k 8 --short
```

### 3) Interaktiver Chat

```bash
docbot chat --top-k 8 --max-context-chunks 5 --short
```

Beenden mit `/exit`.

### 4) Status prüfen

```bash
docbot status
```

## Docker

### Build

```bash
docker compose build
```

### Beispielaufruf

```bash
docker compose run --rm app index --path /workspace/sample_data
```

> Standardmäßig nutzt `app` das lokale Dateisystem über Volume-Mount.

### Optional Ollama in Compose

```bash
docker compose --profile ollama up -d ollama
```

Dann `ollama_base_url` entsprechend auf `http://ollama:11434` setzen, falls `app` im selben Compose-Netz läuft.

## Logging

- Strukturierte JSON-Logs: `logs/docbot.log.jsonl`
- Chat-Sessions: `logs/chat.jsonl`
- Enthalten u.a. Retrieval-Scores, Prompts, Modellantworten, Fehler und Latenzen.

## Tests

```bash
pytest
```

Enthalten:
- TXT-Chunking inkl. langem Absatz
- CSV-Chunking (Semikolon + Header + Gruppen)
- Hash/Change-Detection
- Smoke-Test für SQLite-Metadaten

## Troubleshooting

1. **Ollama nicht erreichbar**
   - Prüfe `ollama_base_url`
   - Läuft der Dienst? `systemctl status ollama` oder `ollama serve`

2. **Modell fehlt**
   - Fehlermeldung enthält Hinweis: `ollama pull <model>`

3. **Embeddings mit `gemma3` nicht verfügbar**
   - Docbot versucht automatisch Fallback auf `embedding_fallback_model` (Default `nomic-embed-text`).

4. **Berechtigungsprobleme bei Chroma/SQLite**
   - Stelle Schreibrechte auf `data/` und `logs/` sicher.

5. **Leerer Kontext / schwache Treffer**
   - `top_k` erhöhen
   - `max_context_chunks` erhöhen
   - Chunking-Parameter (`csv_group_size`, `txt_max_chars`) anpassen
