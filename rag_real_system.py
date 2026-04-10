"""
Real RAG Pipeline — Build, Retrieve, Generate, Judge, Monitor
=============================================================

A complete RAG system using real documents, real embeddings, and real LLM-as-judge
evaluation. No synthetic data, no keyword overlap metrics.

Components:
    DocumentIndexer  — fetch Wikipedia, chunk, embed, build FAISS index
    RAGPipeline      — retrieve + generate using the index and local Qwen
    LLMJudge         — score RAG outputs on 5 dimensions via local Qwen
    DriftDetector    — statistical process control (ported from rag_testing_system.py)

All components use local models only (sentence-transformers + llama.cpp Qwen).
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single text chunk from a document."""
    text: str
    source_article: str
    cluster: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """A retrieved chunk with its similarity score."""
    chunk: Chunk
    score: float  # L2 distance from FAISS (lower = more similar)


@dataclass
class RAGOutput:
    """Complete output from a single RAG query with all metrics."""
    query: str
    retrieved: list[RetrievalResult]
    context: str  # concatenated chunk texts sent to LLM
    answer: str
    ground_truth: Optional[str] = None

    # LLM-as-judge metrics (filled by LLMJudge)
    contextual_precision: float = 0.0
    contextual_recall: float = 0.0
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    answer_correctness: float = 0.0
    latency_ms: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    query_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    phase: str = ""


@dataclass
class DriftAlert:
    """Raised when a metric drifts beyond threshold."""
    metric_name: str
    baseline_mean: float
    baseline_std: float
    current_mean: float
    drift_score: float  # Cohen's d
    p_value: float
    severity: str  # "warning" or "critical"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: str = ""


@dataclass
class QAPair:
    """A ground truth question-answer pair."""
    question: str
    ground_truth_answer: str
    source_article: str
    cluster: str


# ─────────────────────────────────────────────────────────────
# 1. Document Indexer
# ─────────────────────────────────────────────────────────────

class DocumentIndexer:
    """
    Fetches Wikipedia articles, chunks them, embeds them, builds a FAISS index.

    The index and chunk store are saved to disk so Phase 0 only runs once.
    Subsequent phases load from disk.
    """

    # Topic clusters — 10 articles each, 30 total
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
        index_dir: str = "rag_index",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.index_dir = Path(index_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model

        self._embedder = None
        self.chunks: list[Chunk] = []
        self.index: Optional[faiss.IndexFlatL2] = None
        self.articles: dict[str, str] = {}  # title -> content

    @property
    def embedder(self):
        """Lazy load the embedding model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def is_built(self) -> bool:
        """Check if the index already exists on disk."""
        return (
            (self.index_dir / "faiss.index").exists()
            and (self.index_dir / "chunks.pkl").exists()
            and (self.index_dir / "articles.json").exists()
        )

    def fetch_articles(self, progress_callback=None) -> dict[str, dict]:
        """
        Fetch all 30 Wikipedia articles. Returns {title: {content, cluster}}.
        Handles disambiguation gracefully.
        """
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
                    # Take the first option that isn't a list page
                    for opt in e.options[:5]:
                        try:
                            page = wikipedia.page(opt, auto_suggest=False)
                            articles[page.title] = {
                                "content": page.content,
                                "cluster": cluster_name,
                            }
                            if progress_callback:
                                progress_callback(f"  ✓ {page.title} (disambig → {opt}, {len(page.content):,} chars)")
                            break
                        except Exception:
                            continue
                except wikipedia.PageError:
                    if progress_callback:
                        progress_callback(f"  ✗ {title} — page not found, skipping")
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"  ✗ {title} — {e}")

        self.articles = {t: d["content"] for t, d in articles.items()}
        return articles

    def chunk_articles(self, articles: dict[str, dict], progress_callback=None) -> list[Chunk]:
        """
        Split all articles into chunks using LangChain's RecursiveCharacterTextSplitter.
        Returns list of Chunk objects with source/cluster metadata.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = []
        for title, info in articles.items():
            texts = splitter.split_text(info["content"])
            for i, text in enumerate(texts):
                chunks.append(Chunk(
                    text=text,
                    source_article=title,
                    cluster=info["cluster"],
                    chunk_index=i,
                ))

        self.chunks = chunks
        if progress_callback:
            progress_callback(f"Created {len(chunks)} chunks from {len(articles)} articles")
        return chunks

    def build_index(self, chunks: list[Chunk], progress_callback=None) -> faiss.IndexFlatL2:
        """
        Embed all chunks and build a FAISS L2 index.
        all-MiniLM-L6-v2 produces 384-dim vectors.
        """
        texts = [c.text for c in chunks]

        if progress_callback:
            progress_callback(f"Embedding {len(texts)} chunks...")

        # Batch encode for efficiency
        embeddings = self.embedder.encode(
            texts,
            show_progress_bar=False,
            batch_size=64,
            normalize_embeddings=True,  # normalise so L2 ≈ cosine
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
        """Load index, chunks, and articles from disk."""
        self.index = faiss.read_index(str(self.index_dir / "faiss.index"))
        with open(self.index_dir / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        with open(self.index_dir / "articles.json", "r") as f:
            self.articles = json.load(f)

    def build_all(self, progress_callback=None) -> tuple[list[Chunk], faiss.IndexFlatL2]:
        """
        Full pipeline: fetch → chunk → embed → index → save.
        If already built, loads from disk instead.
        """
        if self.is_built():
            if progress_callback:
                progress_callback("Index already exists on disk — loading...")
            self.load()
            if progress_callback:
                progress_callback(f"Loaded {len(self.chunks)} chunks, {self.index.ntotal} vectors")
            return self.chunks, self.index

        if progress_callback:
            progress_callback("Fetching Wikipedia articles...")
        articles = self.fetch_articles(progress_callback)

        if progress_callback:
            progress_callback("\nChunking articles...")
        chunks = self.chunk_articles(articles, progress_callback)

        if progress_callback:
            progress_callback("\nBuilding FAISS index...")
        index = self.build_index(chunks, progress_callback)

        self.save()
        if progress_callback:
            progress_callback(f"\nSaved to {self.index_dir}/")

        return chunks, index


# ─────────────────────────────────────────────────────────────
# 2. RAG Pipeline
# ─────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Retrieve-then-generate pipeline.

    retrieve() → searches FAISS index for top-K chunks
    generate() → sends query + context to local Qwen
    query()   → end-to-end: retrieve + generate
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a knowledgeable assistant. Answer the question using ONLY the "
        "provided context. If the context doesn't contain enough information, "
        "say so. Be concise and factual."
    )

    def __init__(
        self,
        indexer: DocumentIndexer,
        llm_base_url: str = "http://127.0.0.1:8001/v1",
        llm_model: str = "unsloth/Qwen3.5-35B-A3B",
        top_k: int = 5,
        system_prompt: Optional[str] = None,
    ):
        self.indexer = indexer
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.top_k = top_k
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._client = None

    @property
    def client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.llm_base_url,
                api_key="not-needed",
            )
        return self._client

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievalResult]:
        """
        Embed the query and search the FAISS index for the top-K nearest chunks.
        Returns RetrievalResult objects sorted by distance (best first).
        """
        k = top_k or self.top_k
        query_vec = self.indexer.embedder.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        distances, indices = self.indexer.index.search(query_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.indexer.chunks):
                continue
            results.append(RetrievalResult(
                chunk=self.indexer.chunks[idx],
                score=float(dist),
            ))
        return results

    def generate(self, query: str, context: str) -> tuple[str, float]:
        """
        Send query + retrieved context to local Qwen. Returns (answer, latency_ms).
        Strips Qwen's <think> tags from the response.
        """
        user_msg = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer concisely based on the context. /no_think"

        start = time.time()
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        latency = (time.time() - start) * 1000

        answer = response.choices[0].message.content or ""
        # Strip Qwen's thinking tags
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

        # Fallback: if content is empty, check reasoning_content
        if not answer:
            extra = getattr(response.choices[0].message, "model_extra", {}) or {}
            reasoning = extra.get("reasoning_content", "")
            if reasoning:
                # Extract any actual answer from after the thinking
                answer = reasoning.split("</think>")[-1].strip() if "</think>" in reasoning else reasoning[-500:]

        return answer, latency

    def query(self, question: str, ground_truth: Optional[str] = None) -> RAGOutput:
        """
        Full RAG pipeline: retrieve → build context → generate → return structured output.
        """
        retrieved = self.retrieve(question)

        # Build context from retrieved chunks
        context_parts = []
        for i, r in enumerate(retrieved):
            context_parts.append(f"[Source: {r.chunk.source_article}]\n{r.chunk.text}")
        context = "\n\n---\n\n".join(context_parts)

        answer, latency = self.generate(question, context)

        return RAGOutput(
            query=question,
            retrieved=retrieved,
            context=context,
            answer=answer,
            ground_truth=ground_truth,
            latency_ms=latency,
        )


# ─────────────────────────────────────────────────────────────
# 3. LLM-as-Judge
# ─────────────────────────────────────────────────────────────

class LLMJudge:
    """
    Scores RAG outputs on 5 dimensions using local Qwen as judge.

    Each metric gets its own focused prompt. The judge returns a float 0-1.
    We parse robustly — the LLM may wrap the score in explanation text.
    """

    def __init__(
        self,
        llm_base_url: str = "http://127.0.0.1:8001/v1",
        llm_model: str = "unsloth/Qwen3.5-35B-A3B",
    ):
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.llm_base_url, api_key="not-needed")
        return self._client

    def _ask_judge(self, prompt: str) -> float:
        """
        Send a judge prompt to Qwen, parse a float 0-1 from the response.
        Robust parsing: handles "0.85", "Score: 0.85", "The score is 0.85/1.0", etc.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt + " /no_think"}],
                max_tokens=512,
                temperature=0.1,
            )
            text = response.choices[0].message.content or ""
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

            if not text:
                extra = getattr(response.choices[0].message, "model_extra", {}) or {}
                text = extra.get("reasoning_content", "")

            return self._parse_score(text)
        except Exception as e:
            print(f"  Judge error: {e}")
            return 0.0

    def _parse_score(self, text: str) -> float:
        """Extract a float 0-1 from LLM output. Tries multiple patterns."""
        # Strip trailing punctuation that Qwen sometimes adds (e.g. "0.0." or "0.8.")
        text = text.strip().rstrip(".")

        # Try JSON first
        json_match = re.search(r'"score"\s*:\s*([\d.]+)', text)
        if json_match:
            val = json_match.group(1).rstrip(".")
            return max(0.0, min(1.0, float(val)))

        # Try "Score: X" pattern
        score_match = re.search(r'[Ss]core\s*[:=]\s*([\d.]+)', text)
        if score_match:
            val = score_match.group(1).rstrip(".")
            return max(0.0, min(1.0, float(val)))

        # Try any float between 0 and 1
        floats = re.findall(r'\b(0\.\d+|1\.0|0|1)\b', text)
        if floats:
            return max(0.0, min(1.0, float(floats[0])))

        # Last resort: try to find any number
        nums = re.findall(r'(\d+\.?\d*)', text)
        for n in nums:
            val = float(n)
            if 0 <= val <= 1:
                return val
            elif 1 < val <= 10:
                return val / 10.0  # handle "8/10" style
            elif 10 < val <= 100:
                return val / 100.0  # handle percentage

        return 0.0

    def score_precision(self, query: str, chunks: list[RetrievalResult]) -> float:
        """
        Contextual precision: what fraction of retrieved chunks are relevant?
        Batches all chunks into one judge call for efficiency.
        """
        if not chunks:
            return 0.0

        chunk_list = "\n\n".join(
            f"Chunk {i+1}: {r.chunk.text[:300]}" for i, r in enumerate(chunks)
        )
        prompt = (
            f"You are evaluating retrieval quality for a RAG system.\n\n"
            f"Query: {query}\n\n"
            f"Retrieved chunks:\n{chunk_list}\n\n"
            f"What fraction of these chunks are relevant to answering the query? "
            f"Reply with ONLY a score from 0.0 to 1.0 where 1.0 means all chunks "
            f"are relevant and 0.0 means none are relevant."
        )
        return self._ask_judge(prompt)

    def score_recall(self, query: str, context: str) -> float:
        """
        Contextual recall: does the retrieved context contain enough info to answer?
        """
        prompt = (
            f"You are evaluating retrieval completeness for a RAG system.\n\n"
            f"Query: {query}\n\n"
            f"Retrieved context:\n{context[:1500]}\n\n"
            f"Does this context contain enough information to fully and correctly "
            f"answer the query? Reply with ONLY a score from 0.0 to 1.0 where "
            f"1.0 means the context is complete and sufficient."
        )
        return self._ask_judge(prompt)

    def score_faithfulness(self, answer: str, context: str) -> float:
        """
        Faithfulness: does the answer only contain info supported by the context?
        This catches hallucination — the answer adds facts not in the chunks.
        """
        prompt = (
            f"You are evaluating answer faithfulness for a RAG system.\n\n"
            f"Context provided to the model:\n{context[:1500]}\n\n"
            f"Generated answer:\n{answer}\n\n"
            f"Does the answer contain ONLY information that is supported by the context? "
            f"Score 0.0 if the answer adds unsupported claims (hallucination). "
            f"Score 1.0 if every claim in the answer is grounded in the context. "
            f"Reply with ONLY a score from 0.0 to 1.0."
        )
        return self._ask_judge(prompt)

    def score_relevance(self, query: str, answer: str) -> float:
        """
        Answer relevance: does the answer actually address the question asked?
        """
        prompt = (
            f"You are evaluating answer relevance.\n\n"
            f"Question: {query}\n\n"
            f"Answer: {answer}\n\n"
            f"Does the answer directly and specifically address the question? "
            f"Score 1.0 if fully relevant, 0.0 if completely off-topic. "
            f"Reply with ONLY a score from 0.0 to 1.0."
        )
        return self._ask_judge(prompt)

    def score_correctness(self, answer: str, ground_truth: str) -> float:
        """
        Answer correctness: how close is the answer to the ground truth?
        Only meaningful when ground_truth is provided.
        """
        if not ground_truth:
            return 0.0

        prompt = (
            f"You are evaluating answer correctness.\n\n"
            f"Ground truth answer: {ground_truth}\n\n"
            f"Generated answer: {answer}\n\n"
            f"How correct is the generated answer compared to the ground truth? "
            f"They don't need to be word-for-word identical — judge semantic equivalence. "
            f"Score 1.0 if fully correct, 0.0 if completely wrong. "
            f"Reply with ONLY a score from 0.0 to 1.0."
        )
        return self._ask_judge(prompt)

    def score_all(self, output: RAGOutput, progress_callback=None) -> RAGOutput:
        """
        Run all 5 judge evaluations on a RAGOutput. Mutates and returns it.
        Each judge call is a separate LLM call (~5 calls per query).
        """
        if progress_callback:
            progress_callback("precision")
        output.contextual_precision = self.score_precision(output.query, output.retrieved)

        if progress_callback:
            progress_callback("recall")
        output.contextual_recall = self.score_recall(output.query, output.context)

        if progress_callback:
            progress_callback("faithfulness")
        output.faithfulness = self.score_faithfulness(output.answer, output.context)

        if progress_callback:
            progress_callback("relevance")
        output.answer_relevance = self.score_relevance(output.query, output.answer)

        if progress_callback:
            progress_callback("correctness")
        output.answer_correctness = self.score_correctness(
            output.answer, output.ground_truth or ""
        )

        return output


# ─────────────────────────────────────────────────────────────
# 4. Drift Detector (ported from rag_testing_system.py)
# ─────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Statistical process control for RAG metrics.

    Uses Welch's t-test and Cohen's d effect size to detect when current
    performance deviates from an established baseline.

    Alert thresholds:
        warning:  Cohen's d > 0.8  OR  mean < baseline - 2σ
        critical: Cohen's d > 1.5  OR  mean < baseline - 3σ

    Why Cohen's d?
        P-values shrink with sample size — a trivial difference becomes
        'significant' with enough data. Effect size tells you if the
        change is actually large enough to care about.
    """

    METRICS = [
        "contextual_precision", "contextual_recall",
        "faithfulness", "answer_relevance", "answer_correctness",
    ]

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.baseline: dict[str, np.ndarray] = {}
        self.baseline_stats: dict[str, dict] = {}
        self.current: dict[str, list[float]] = defaultdict(list)
        self.alerts: list[DriftAlert] = []
        self.history: list[RAGOutput] = []

    def set_baseline(self, results: list[RAGOutput]) -> dict[str, dict]:
        """Establish baseline statistics from known-good results (need 10+)."""
        if len(results) < 5:
            raise ValueError(f"Need at least 5 baseline results, got {len(results)}")

        for metric in self.METRICS:
            values = np.array([getattr(r, metric) for r in results])
            self.baseline[metric] = values
            self.baseline_stats[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.1,
                "p25": float(np.percentile(values, 25)),
                "p50": float(np.percentile(values, 50)),
                "p75": float(np.percentile(values, 75)),
                "n": len(values),
            }
        return self.baseline_stats

    def add_result(self, result: RAGOutput) -> list[DriftAlert]:
        """Add a result to the rolling window. Returns any new drift alerts."""
        self.history.append(result)
        new_alerts = []

        for metric in self.METRICS:
            value = getattr(result, metric)
            window = self.current[metric]
            window.append(value)
            if len(window) > self.window_size:
                window.pop(0)

            # Need enough recent data to compare
            if len(window) < 3 or metric not in self.baseline:
                continue

            alert = self._check_drift(metric, window)
            if alert:
                new_alerts.append(alert)
                self.alerts.append(alert)

        return new_alerts

    def _check_drift(self, metric: str, current_values: list[float]) -> Optional[DriftAlert]:
        """Compare current window vs baseline using t-test + Cohen's d."""
        baseline = self.baseline[metric]
        current = np.array(current_values)
        b_stats = self.baseline_stats[metric]

        # Welch's t-test
        try:
            t_stat, p_value = stats.ttest_ind(baseline, current, equal_var=False)
        except Exception:
            return None

        # Cohen's d
        b_std = np.std(baseline, ddof=1)
        c_std = np.std(current, ddof=1)
        pooled_std = np.sqrt((b_std**2 + c_std**2) / 2)
        if pooled_std < 1e-10:
            cohens_d = 0.0
        else:
            cohens_d = abs(np.mean(baseline) - np.mean(current)) / pooled_std

        current_mean = float(np.mean(current))
        threshold_2sigma = b_stats["mean"] - 2 * max(b_stats["std"], 0.05)
        threshold_3sigma = b_stats["mean"] - 3 * max(b_stats["std"], 0.05)

        severity = None
        if cohens_d > 1.5 or current_mean < threshold_3sigma:
            severity = "critical"
        elif cohens_d > 0.8 or current_mean < threshold_2sigma:
            severity = "warning"

        # Only alert on degradation (current worse than baseline)
        if severity and current_mean < b_stats["mean"]:
            return DriftAlert(
                metric_name=metric,
                baseline_mean=b_stats["mean"],
                baseline_std=b_stats["std"],
                current_mean=current_mean,
                drift_score=cohens_d,
                p_value=float(p_value) if not np.isnan(p_value) else 1.0,
                severity=severity,
                details=(
                    f"{metric}: baseline={b_stats['mean']:.3f}±{b_stats['std']:.3f}, "
                    f"current={current_mean:.3f}, d={cohens_d:.2f}, p={p_value:.4f}"
                ),
            )
        return None

    def reset_window(self):
        """Clear the current window without touching the baseline."""
        self.current = defaultdict(list)

    def get_status(self) -> pd.DataFrame:
        """Dashboard: current vs baseline for all metrics."""
        rows = []
        for metric in self.METRICS:
            b = self.baseline_stats.get(metric, {})
            c = self.current.get(metric, [])
            c_mean = float(np.mean(c)) if c else 0
            delta = c_mean - b.get("mean", 0) if c else 0
            is_drift = c and c_mean < b.get("mean", 0) - 2 * max(b.get("std", 0), 0.05)
            rows.append({
                "metric": metric,
                "baseline_mean": round(b.get("mean", 0), 3),
                "baseline_std": round(b.get("std", 0), 3),
                "current_mean": round(c_mean, 3) if c else "—",
                "current_n": len(c),
                "delta": round(delta, 3) if c else "—",
                "status": "⚠ DRIFT" if is_drift else "✓ OK",
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 5. QA Pair Generator
# ─────────────────────────────────────────────────────────────

class QAGenerator:
    """
    Generates ground truth QA pairs by asking Qwen factual questions
    about each article's content.
    """

    def __init__(
        self,
        llm_base_url: str = "http://127.0.0.1:8001/v1",
        llm_model: str = "unsloth/Qwen3.5-35B-A3B",
    ):
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.llm_base_url, api_key="not-needed")
        return self._client

    def generate_qa(self, title: str, content: str, cluster: str) -> QAPair:
        """
        Generate one factual QA pair from an article.

        Strategy: create a factual question from the title, then extract
        a ground truth answer from the article's opening sentences.
        This is more reliable than asking the LLM to generate both
        (Qwen3.5 struggles with format instructions for this task).
        """
        # Extract first 2-3 sentences as ground truth
        sentences = re.split(r'(?<=[.!?])\s+', content[:1000])
        ground_truth = " ".join(sentences[:3]).strip()
        if len(ground_truth) > 300:
            ground_truth = ground_truth[:300]

        # Generate a question based on cluster + title
        cluster_questions = {
            "ai_ml": [
                f"What is {title} and how does it work?",
                f"Explain the key concepts behind {title}.",
                f"What is the significance of {title} in artificial intelligence?",
            ],
            "history": [
                f"What was {title} and what were its main causes and consequences?",
                f"Describe the key events and significance of {title}.",
                f"What role did {title} play in shaping the modern world?",
            ],
            "science": [
                f"What is {title} and why is it important in science?",
                f"Explain the fundamental principles of {title}.",
                f"How does {title} work and what are its key applications?",
            ],
        }

        import hashlib
        # Deterministic but varied question selection based on title hash
        idx = int(hashlib.md5(title.encode()).hexdigest(), 16) % 3
        questions = cluster_questions.get(cluster, cluster_questions["science"])
        question = questions[idx]

        return QAPair(
            question=question,
            ground_truth_answer=ground_truth,
            source_article=title,
            cluster=cluster,
        )

    @staticmethod
    def save_qa_pairs(pairs: list[QAPair], path: str = "rag_qa_pairs.jsonl"):
        """Save QA pairs to JSONL."""
        with open(path, "w") as f:
            for p in pairs:
                f.write(json.dumps({
                    "question": p.question,
                    "ground_truth_answer": p.ground_truth_answer,
                    "source_article": p.source_article,
                    "cluster": p.cluster,
                }) + "\n")

    @staticmethod
    def load_qa_pairs(path: str = "rag_qa_pairs.jsonl") -> list[QAPair]:
        """Load QA pairs from JSONL."""
        pairs = []
        with open(path) as f:
            for line in f:
                d = json.loads(line.strip())
                pairs.append(QAPair(**d))
        return pairs
