from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]

    words = paragraph.split()
    segments: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        if current and current_len + 1 + len(word) > max_chars:
            segments.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len = current_len + len(word) + (1 if len(current) > 1 else 0)

    if current:
        segments.append(" ".join(current))

    return segments


def chunk_txt_file(file_path: Path, max_chars: int = 2000) -> Generator[Chunk, None, None]:
    paragraph_buffer: list[str] = []
    paragraph_index = 0

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.rstrip("\n")
            if stripped.strip() == "":
                if paragraph_buffer:
                    paragraph = "\n".join(paragraph_buffer).strip()
                    for sub_index, piece in enumerate(_split_long_paragraph(paragraph, max_chars), start=1):
                        chunk_id = f"{file_path.name}:txt:{paragraph_index}:{sub_index}"
                        yield Chunk(
                            chunk_id=chunk_id,
                            text=piece,
                            metadata={
                                "file_name": file_path.name,
                                "file_path": str(file_path),
                                "chunk_type": "txt",
                                "paragraph_index": paragraph_index,
                                "sub_chunk_index": sub_index,
                            },
                        )
                    paragraph_index += 1
                    paragraph_buffer = []
                continue
            paragraph_buffer.append(stripped)

    if paragraph_buffer:
        paragraph = "\n".join(paragraph_buffer).strip()
        for sub_index, piece in enumerate(_split_long_paragraph(paragraph, max_chars), start=1):
            chunk_id = f"{file_path.name}:txt:{paragraph_index}:{sub_index}"
            yield Chunk(
                chunk_id=chunk_id,
                text=piece,
                metadata={
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "chunk_type": "txt",
                    "paragraph_index": paragraph_index,
                    "sub_chunk_index": sub_index,
                },
            )


def chunk_csv_file(file_path: Path, group_size: int = 50) -> Iterable[Chunk]:
    # utf-8-sig erlaubt CSVs mit BOM, die sonst im Header Probleme verursachen können.
    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        try:
            header = next(reader)
        except StopIteration:
            return

        # Leere Header-Zeile behandeln (z. B. wenn die Datei mit Leerzeilen startet).
        if not any(cell.strip() for cell in header):
            for candidate in reader:
                if any(cell.strip() for cell in candidate):
                    header = candidate
                    break
            else:
                return

        row_group: list[list[str]] = []
        start_line = 2
        line_num = 1
        group_index = 0
        data_rows_seen = 0

        for row in reader:
            line_num += 1
            if not any(cell.strip() for cell in row):
                continue

            row_group.append(row)
            data_rows_seen += 1
            if len(row_group) == group_size:
                end_line = line_num
                yield _build_csv_chunk(file_path, header, row_group, start_line, end_line, group_index)
                group_index += 1
                row_group = []
                start_line = line_num + 1

        if row_group:
            end_line = line_num
            yield _build_csv_chunk(file_path, header, row_group, start_line, end_line, group_index)
        elif data_rows_seen == 0:
            # Header-only CSV nicht ignorieren: als Schema-Chunk indexieren.
            yield _build_csv_chunk(file_path, header, [["<keine datenzeilen>"]], 2, 2, group_index)


def _build_csv_chunk(
    file_path: Path,
    header: list[str],
    rows: list[list[str]],
    start_line: int,
    end_line: int,
    group_index: int,
) -> Chunk:
    schema_info = ", ".join(header)
    row_texts = ["; ".join(row) for row in rows]
    text = (
        f"CSV-Datei: {file_path.name}\n"
        f"Spalten: {schema_info}\n"
        f"Zeilenbereich: {start_line}-{end_line}\n"
        + "\n".join(row_texts)
    )
    chunk_id = f"{file_path.name}:csv:{start_line}-{end_line}:{group_index}"

    return Chunk(
        chunk_id=chunk_id,
        text=text,
        metadata={
            "file_name": file_path.name,
            "file_path": str(file_path),
            "chunk_type": "csv",
            "line_start": start_line,
            "line_end": end_line,
            "header": schema_info,
            "group_index": group_index,
        },
    )
