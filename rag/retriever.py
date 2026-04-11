"""
Retriever
=========
FAISS vector search returning chunks with similarity scores and doc_ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from rag.config import DEFAULT_TOP_K
from rag.indexer import Chunk, DocumentIndexer


@dataclass
class RetrievalResult:
    """A retrieved chunk with its similarity score."""
    chunk: Chunk
    score: float       # L2 distance (lower = more similar with normalized vectors)
    rank: int          # 0-indexed rank position

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

    @property
    def source_article(self) -> str:
        return self.chunk.source_article


class Retriever:
    """FAISS-based retriever. Embeds the query and returns top-K nearest chunks."""

    def __init__(self, indexer: DocumentIndexer, top_k: int = DEFAULT_TOP_K):
        self.indexer = indexer
        self.top_k = top_k

    def search(self, query: str, top_k: Optional[int] = None) -> list[RetrievalResult]:
        """
        Embed query and search FAISS index.
        Returns RetrievalResult objects sorted by distance (best first).
        """
        k = top_k or self.top_k
        query_vec = self.indexer.embedder.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        distances, indices = self.indexer.index.search(query_vec, k)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.indexer.chunks):
                continue
            results.append(RetrievalResult(
                chunk=self.indexer.chunks[idx],
                score=float(dist),
                rank=rank,
            ))
        return results

    def get_retrieved_doc_ids(self, results: list[RetrievalResult]) -> list[str]:
        """Extract unique doc_ids from retrieval results, preserving rank order."""
        seen = set()
        doc_ids = []
        for r in results:
            if r.doc_id not in seen:
                seen.add(r.doc_id)
                doc_ids.append(r.doc_id)
        return doc_ids
