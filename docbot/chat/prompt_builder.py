from __future__ import annotations

from docbot.chat.retriever import RetrievedChunk

SYSTEM_PROMPT = """
Du bist ein interner Doku-Assistent. Antworte ausschließlich auf Deutsch.
Regeln:
1) Nutze nur den bereitgestellten Kontext.
2) Wenn der Kontext nicht reicht oder die Frage unklar ist: sage das explizit und stelle eine gezielte Rückfrage.
3) Keine Halluzinationen.
4) Gib immer einen Abschnitt 'Quellen' aus.
5) CSV-Inhalte bevorzugt aggregiert zusammenfassen (Muster, Trends, relevante Spalten), außer der Nutzer fordert Rohdetails.
""".strip()


def _format_source(chunk: RetrievedChunk) -> str:
    meta = chunk.metadata
    if meta.get("chunk_type") == "txt":
        area = f"Absatz {meta.get('paragraph_index')}"
    else:
        area = f"Zeilen {meta.get('line_start')}-{meta.get('line_end')}"
    return f"{meta.get('file_name', 'unbekannt')} | {area} | Chunk-ID: {chunk.chunk_id}"


def build_user_prompt(question: str, chunks: list[RetrievedChunk], mode: str = "short", max_chars_per_chunk: int = 1200) -> str:
    length_instruction = (
        "Antworte kurz (3-8 Sätze), prägnant und konkret."
        if mode == "short"
        else "Antworte ausführlich und strukturiert mit sinnvollen Bulletpoints."
    )

    context_blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        excerpt = chunk.text[:max_chars_per_chunk]
        source = _format_source(chunk)
        context_blocks.append(
            f"[{idx}] Quelle: {source}\nRelevanz-Score (kleiner = ähnlicher): {chunk.score:.4f}\nInhalt:\n{excerpt}"
        )

    context = "\n\n".join(context_blocks) if context_blocks else "Kein Kontext gefunden."

    return (
        f"Nutzerfrage: {question}\n"
        f"Antwortmodus: {mode}\n"
        f"{length_instruction}\n\n"
        "Kontextquellen:\n"
        f"{context}\n\n"
        "Gib die Antwort mit einem abschließenden Abschnitt 'Quellen' aus. "
        "Jede Quelle mit Dateiname, Bereich (Absatz oder Zeilenbereich) und Chunk-ID nennen."
    )
