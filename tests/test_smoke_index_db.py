from pathlib import Path

import pytest

from docbot.indexing.meta_db import MetaDB


@pytest.mark.smoke
def test_meta_db_smoke(tmp_path: Path) -> None:
    db = MetaDB(tmp_path / "meta.db")
    db.upsert_file("/tmp/demo.txt", "abc", 1.0, 100, "2024-01-01T00:00:00Z")
    db.add_chunk(
        chunk_id="demo:txt:0:1",
        file_path="/tmp/demo.txt",
        metadata={"chunk_type": "txt", "paragraph_index": 0},
        created_at="2024-01-01T00:00:00Z",
    )

    status = db.status()
    chunks = db.get_chunks_by_file("/tmp/demo.txt")

    assert status["files"] == 1
    assert status["chunks"] == 1
    assert chunks == ["demo:txt:0:1"]
    db.close()
