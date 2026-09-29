"""Uniform read access to the raw data, whether it is `raw_data.zip` or an extracted folder."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path


class RawData:
    """Files are addressed relative to the `raw_data/` root, e.g. 'CSV Files/Mental_Health_FAQ.csv'."""

    ROOT = "raw_data/"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Raw data not found at {self.path}. Expected data/raw_data.zip (tracked in git) "
                "or an extracted data/raw_data/ directory."
            )
        self._zip = zipfile.ZipFile(self.path) if self.path.is_file() else None

    def list(self, folder: str, suffix: str = "") -> list[str]:
        folder = folder.rstrip("/") + "/"
        if self._zip is not None:
            names = [
                n[len(self.ROOT):]
                for n in self._zip.namelist()
                if n.startswith(self.ROOT + folder) and not n.endswith("/")
            ]
        else:
            base = self.path / folder
            names = [str(p.relative_to(self.path)) for p in base.rglob("*") if p.is_file()] if base.exists() else []
        return sorted(n for n in names if n.lower().endswith(suffix.lower()))

    def read(self, name: str) -> bytes:
        if self._zip is not None:
            return self._zip.read(self.ROOT + name)
        return (self.path / name).read_bytes()

    def exists(self, name: str) -> bool:
        if self._zip is not None:
            return (self.ROOT + name) in self._zip.namelist()
        return (self.path / name).exists()

    def content_hash(self, names: list[str]) -> str:
        """Hash of the given files' bytes (order-independent)."""
        h = hashlib.sha256()
        for n in sorted(names):
            h.update(n.encode())
            h.update(hashlib.sha256(self.read(n)).digest())
        return h.hexdigest()
