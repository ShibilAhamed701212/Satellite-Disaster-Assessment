"""
Data cache for managing downloaded satellite imagery.
"""

import os
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime


@dataclass
class CacheEntry:
    """A cached data product."""
    key: str
    file_path: str
    source: str
    cached_at: str = ""
    size_bytes: int = 0
    metadata: Dict = field(default_factory=dict)


class DataCache:
    """Simple file-based data cache with provenance tracking."""

    def __init__(self, cache_dir: str = "cache", max_size_gb: float = 10.0):
        self.cache_dir = cache_dir
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self._index_path = os.path.join(cache_dir, "_cache_index.json")
        self._entries: Dict[str, CacheEntry] = {}
        os.makedirs(cache_dir, exist_ok=True)
        self._load_index()

    def _load_index(self):
        if os.path.isfile(self._index_path):
            try:
                with open(self._index_path, "r") as f:
                    data = json.load(f)
                for key, entry_dict in data.items():
                    self._entries[key] = CacheEntry(**entry_dict)
            except Exception:
                self._entries = {}

    def _save_index(self):
        try:
            data = {
                k: {
                    "key": e.key,
                    "file_path": e.file_path,
                    "source": e.source,
                    "cached_at": e.cached_at,
                    "size_bytes": e.size_bytes,
                    "metadata": e.metadata,
                }
                for k, e in self._entries.items()
            }
            with open(self._index_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get a cached entry by key."""
        entry = self._entries.get(key)
        if entry and os.path.isfile(entry.file_path):
            return entry
        return None

    def put(self, key: str, file_path: str, source: str, metadata: Optional[Dict] = None):
        """Add a file to cache."""
        size = os.path.getsize(file_path) if os.path.isfile(file_path) else 0
        entry = CacheEntry(
            key=key,
            file_path=file_path,
            source=source,
            cached_at=datetime.now().isoformat(),
            size_bytes=size,
            metadata=metadata or {},
        )
        self._entries[key] = entry
        self._save_index()

    def has(self, key: str) -> bool:
        entry = self._entries.get(key)
        return entry is not None and os.path.isfile(entry.file_path)

    def make_key(self, **kwargs) -> str:
        """Generate a cache key from parameters."""
        raw = json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    @property
    def total_size_bytes(self) -> int:
        return sum(e.size_bytes for e in self._entries.values())

    @property
    def count(self) -> int:
        return len(self._entries)
