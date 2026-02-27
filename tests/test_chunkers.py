from pathlib import Path

from docbot.indexing.chunkers import chunk_csv_file, chunk_txt_file


def test_txt_paragraph_chunking_with_long_paragraph(tmp_path: Path) -> None:
    content = (
        "Erster Absatz.\n\n"
        + " ".join(["Wort"] * 900)
        + "\n\nDritter Absatz."
    )
    file_path = tmp_path / "demo.txt"
    file_path.write_text(content, encoding="utf-8")

    chunks = list(chunk_txt_file(file_path, max_chars=200))

    assert len(chunks) >= 3
    assert chunks[0].metadata["paragraph_index"] == 0
    assert all(len(chunk.text) <= 200 for chunk in chunks[1:-1])


def test_csv_semicolon_group_chunking(tmp_path: Path) -> None:
    csv_data = "name;wert\nA;1\nB;2\nC;3\nD;4\n"
    file_path = tmp_path / "data.csv"
    file_path.write_text(csv_data, encoding="utf-8")

    chunks = list(chunk_csv_file(file_path, group_size=2))

    assert len(chunks) == 2
    assert chunks[0].metadata["line_start"] == 2
    assert chunks[0].metadata["line_end"] == 3
    assert "name, wert" in chunks[0].metadata["header"]
