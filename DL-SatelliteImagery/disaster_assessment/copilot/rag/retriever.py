"""
Retriever for the disaster intelligence RAG system.
"""

from typing import List


from .ingestion import DocumentChunk


class RAGRetriever:
    """Simple keyword-based retriever for document chunks.

    For production use, replace with embedding-based similarity search
    using sentence-transformers or similar.
    """

    def __init__(self):
        self._documents: List[DocumentChunk] = []

    def index(self, documents: List[DocumentChunk]):
        """Index document chunks for retrieval."""
        self._documents = documents

    def retrieve(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """Retrieve relevant chunks based on keyword matching.

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            List of matching DocumentChunks.
        """
        if not self._documents:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored = []
        for doc in self._documents:
            doc_words = set(doc.text.lower().split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                scored.append((overlap, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]
