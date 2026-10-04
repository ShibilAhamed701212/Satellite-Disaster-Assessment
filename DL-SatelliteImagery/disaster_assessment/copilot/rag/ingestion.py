"""
Document ingestion for the disaster intelligence RAG system.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DocumentChunk:
    """A chunk of ingested text."""
    chunk_id: str
    text: str
    source: str = ""
    metadata: dict = field(default_factory=dict)


class RAGIngestor:
    """Ingest disaster documentation for RAG retrieval.

    Ingests SOPs, guidelines, satellite imagery documentation,
    and user-provided disaster reports.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._documents: List[DocumentChunk] = []

    def ingest_text(self, text: str, source: str = "manual") -> List[DocumentChunk]:
        """Chunk and ingest a text document."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk_text = text[i:i + self.chunk_size]
            chunk = DocumentChunk(
                chunk_id=f"{source}_{len(chunks)}",
                text=chunk_text,
                source=source,
            )
            chunks.append(chunk)
            self._documents.append(chunk)
        return chunks

    def ingest_file(self, file_path: str) -> List[DocumentChunk]:
        """Ingest a text file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            return self.ingest_text(text, source=file_path)
        except Exception:
            return []

    @property
    def num_documents(self) -> int:
        return len(self._documents)
