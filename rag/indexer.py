"""
Document Indexer
================
Fetch Wikipedia articles, chunk them, embed with sentence-transformers,
and build a FAISS L2 index. Saves to disk so indexing only runs once.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    INDEX_DIR,
)


@dataclass
class Chunk:
    """A single text chunk from a document."""
    text: str
    doc_id: str
    source_article: str
    cluster: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class DocumentIndexer:
    """
    Fetches Wikipedia articles, chunks them, embeds them, builds a FAISS index.
    The index and chunk store are saved to disk so indexing only runs once.
    """

    CLUSTERS = {
        "ai_ml": [
            "Large language model",
            "Transformer (deep learning architecture)",
            "Retrieval-augmented generation",
            "Prompt engineering",
            "Vector database",
            "Word embedding",
            "BERT (language model)",
            "GPT-4",
            "Fine-tuning (deep learning)",
            "Hallucination (artificial intelligence)",
        ],
        "history": [
            "World War II",
            "French Revolution",
            "Roman Empire",
            "Byzantine Empire",
            "Mongol Empire",
            "Industrial Revolution",
            "Cold War",
            "Renaissance",
            "Age of Enlightenment",
            "British Empire",
        ],
        "science": [
            "DNA",
            "Quantum mechanics",
            "Black hole",
            "CRISPR",
            "Vaccine",
            "Climate change",
            "Photosynthesis",
            "Theory of relativity",
            "Periodic table",
            "Neuroscience",
        ],
    }

    def __init__(
        self,
        index_dir: str | Path = INDEX_DIR,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.index_dir = Path(index_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model

        self._embedder = None
        self.chunks: list[Chunk] = []
        self.index: Optional[faiss.IndexFlatL2] = None
        self.articles: dict[str, str] = {}

    @property
    def embedder(self):
        """Lazy-load the embedding model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def is_built(self) -> bool:
        return (
            (self.index_dir / "faiss.index").exists()
            and (self.index_dir / "chunks.pkl").exists()
            and (self.index_dir / "articles.json").exists()
        )

    def fetch_articles(self, progress_callback=None) -> dict[str, dict]:
        """Fetch all 30 Wikipedia articles."""
        import wikipedia

        articles = {}
        for cluster_name, titles in self.CLUSTERS.items():
            for title in titles:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    articles[page.title] = {
                        "content": page.content,
                        "cluster": cluster_name,
                    }
                    if progress_callback:
                        progress_callback(f"  ✓ {page.title} ({len(page.content):,} chars)")
                except wikipedia.DisambiguationError as e:
                    for opt in e.options[:5]:
                        try:
                            page = wikipedia.page(opt, auto_suggest=False)
                            articles[page.title] = {
                                "content": page.content,
                                "cluster": cluster_name,
                            }
                            if progress_callback:
                                progress_callback(f"  ✓ {page.title} (disambig → {opt})")
                            break
                        except Exception:
                            continue
                except wikipedia.PageError:
                    if progress_callback:
                        progress_callback(f"  ✗ {title} — not found")
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"  ✗ {title} — {e}")

        self.articles = {t: d["content"] for t, d in articles.items()}
        return articles

    def chunk_articles(self, articles: dict[str, dict], progress_callback=None) -> list[Chunk]:
        """Split articles into chunks with doc_id tracking."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = []
        article_titles = sorted(articles.keys())
        for doc_idx, title in enumerate(article_titles):
            info = articles[title]
            texts = splitter.split_text(info["content"])
            doc_id = f"doc_{doc_idx}"
            for i, text in enumerate(texts):
                chunks.append(Chunk(
                    text=text,
                    doc_id=doc_id,
                    source_article=title,
                    cluster=info["cluster"],
                    chunk_index=i,
                    metadata={"doc_id": doc_id, "title": title},
                ))

        self.chunks = chunks
        if progress_callback:
            progress_callback(f"Created {len(chunks)} chunks from {len(articles)} articles")
        return chunks

    def build_index(self, chunks: list[Chunk], progress_callback=None) -> faiss.IndexFlatL2:
        """Embed all chunks and build a FAISS L2 index."""
        texts = [c.text for c in chunks]

        if progress_callback:
            progress_callback(f"Embedding {len(texts)} chunks...")

        embeddings = self.embedder.encode(
            texts,
            show_progress_bar=False,
            batch_size=64,
            normalize_embeddings=True,
        )

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype(np.float32))

        self.index = index
        if progress_callback:
            progress_callback(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
        return index

    def save(self):
        """Save index, chunks, and articles to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_dir / "faiss.index"))
        with open(self.index_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(self.index_dir / "articles.json", "w") as f:
            json.dump(self.articles, f)

    def load(self):
        """Load index, chunks, and articles from disk. Handles legacy chunk format."""
        self.index = faiss.read_index(str(self.index_dir / "faiss.index"))
        with open(self.index_dir / "chunks.pkl", "rb") as f:
            raw_chunks = pickle.load(f)
        with open(self.index_dir / "articles.json", "r") as f:
            self.articles = json.load(f)

        # Build article→doc_id mapping for backward compat
        article_to_doc_id: dict[str, str] = {}
        sorted_articles = sorted(self.articles.keys())
        for i, title in enumerate(sorted_articles):
            article_to_doc_id[title] = f"doc_{i}"

        # Convert legacy chunks (missing doc_id) to new Chunk format
        self.chunks = []
        for c in raw_chunks:
            if isinstance(c, Chunk) and hasattr(c, "doc_id") and c.doc_id:
                self.chunks.append(c)
            else:
                src = getattr(c, "source_article", "")
                doc_id = article_to_doc_id.get(src, "doc_unknown")
                self.chunks.append(Chunk(
                    text=c.text,
                    doc_id=doc_id,
                    source_article=src,
                    cluster=getattr(c, "cluster", ""),
                    chunk_index=getattr(c, "chunk_index", 0),
                    metadata=getattr(c, "metadata", {}),
                ))

    def build_all(self, progress_callback=None) -> tuple[list[Chunk], faiss.IndexFlatL2]:
        """Full pipeline: fetch → chunk → embed → index → save. Loads from disk if built."""
        if self.is_built():
            if progress_callback:
                progress_callback("Index exists on disk — loading...")
            self.load()
            if progress_callback:
                progress_callback(f"Loaded {len(self.chunks)} chunks, {self.index.ntotal} vectors")
            return self.chunks, self.index

        if progress_callback:
            progress_callback("Fetching Wikipedia articles...")
        articles = self.fetch_articles(progress_callback)

        if progress_callback:
            progress_callback("Chunking articles...")
        chunks = self.chunk_articles(articles, progress_callback)

        if progress_callback:
            progress_callback("Building FAISS index...")
        index = self.build_index(chunks, progress_callback)

        self.save()
        if progress_callback:
            progress_callback(f"Saved to {self.index_dir}/")

        return chunks, index
