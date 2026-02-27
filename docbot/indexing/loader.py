from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SUPPORTED_EXTENSIONS = {".txt", ".csv"}


@dataclass
class FileInfo:
    path: Path
    mtime: float
    size: int
    file_hash: str


def iter_doc_files(base_path: Path, recursive: bool = False) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for item in base_path.glob(pattern):
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield item


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_file_info(path: Path) -> FileInfo:
    stat = path.stat()
    return FileInfo(path=path, mtime=stat.st_mtime, size=stat.st_size, file_hash=hash_file(path))
