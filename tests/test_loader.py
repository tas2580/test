from pathlib import Path

from docbot.indexing.loader import collect_file_info, hash_file


def test_hash_changes_on_file_update(tmp_path: Path) -> None:
    file_path = tmp_path / "a.txt"
    file_path.write_text("Hallo", encoding="utf-8")

    first_hash = hash_file(file_path)
    first_info = collect_file_info(file_path)

    file_path.write_text("Hallo Welt", encoding="utf-8")

    second_hash = hash_file(file_path)
    second_info = collect_file_info(file_path)

    assert first_hash != second_hash
    assert first_info.size != second_info.size
