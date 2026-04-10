"""
RAG Testing & Drift Detection System
=====================================
A complete system for testing RAG pipelines, detecting performance drift,
and simulating failures to validate your monitoring catches problems.

Three core components:
  1. MetricsCalculator - scores every RAG query on 5 dimensions
  2. DriftDetector     - statistical process control (t-test + Cohen's d)
  3. FailureSimulator  - 7 injectable failure modes

Usage:
    system = RAGTestingSystem()
    system.establish_baseline(good_results)
    result, alert = system.test_query(query, chunks, answer, ground_truth)
"""

from __future__ import annotations

import random
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A single chunk returned by the retriever."""
    text: str
    score: float            # similarity score (0-1)
    source: str = ""        # document source identifier
    timestamp: str = ""     # when the chunk was indexed
    metadata: dict = field(default_factory=dict)


@dataclass
class RAGResult:
    """Complete result from a single RAG query, with all metrics."""
    query: str
    retrieved_chunks: list[RetrievedChunk]
    generated_answer: str
    ground_truth: Optional[str] = None

    # Metrics (filled by MetricsCalculator)
    contextual_precision: float = 0.0
    contextual_recall: float = 0.0
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    answer_correctness: float = 0.0
    latency_ms: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    query_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    simulation_mode: Optional[str] = None


@dataclass
class DriftAlert:
    """Raised when a metric drifts beyond threshold."""
    metric_name: str
    baseline_mean: float
    baseline_std: float
    current_mean: float
    drift_score: float     # Cohen's d effect size
    p_value: float
    severity: str          # "warning" (2sigma) or "critical" (3sigma)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: str = ""


# ─────────────────────────────────────────────────────────────
# 1. Metrics Calculator
# ─────────────────────────────────────────────────────────────

class MetricsCalculator:
    """
    Calculates RAG quality metrics using keyword overlap.

    This is a lightweight implementation suitable for demos and baselines.
    The interface is designed so you can swap in DeepEval, RAGAS, or
    LLM-as-judge scoring by subclassing and overriding individual methods.

    Metrics:
        contextual_precision: What fraction of top-K chunks are relevant?
        contextual_recall:    What fraction of answer info appears in chunks?
        faithfulness:         What fraction of answer claims are grounded in chunks?
        answer_relevance:     Does the answer address the query?
        answer_correctness:   How close is the answer to ground truth?
    """

    def _tokenize(self, text: str) -> set[str]:
        """Simple word-level tokenization with lowercasing."""
        return set(re.findall(r'\b\w{2,}\b', text.lower()))

    def _keyword_overlap(self, text_a: str, text_b: str) -> float:
        """Jaccard-like overlap between two texts."""
        tokens_a = self._tokenize(text_a)
        tokens_b = self._tokenize(text_b)
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    def contextual_precision(self, query: str, chunks: list[RetrievedChunk],
                              ground_truth: Optional[str] = None, k: int = 5) -> float:
        """
        What % of top-K chunks contain query-relevant information?

        High precision = retriever is not polluting context with junk.
        Low precision  = LLM gets confused by irrelevant chunks.
        """
        reference = ground_truth if ground_truth else query
        top_k = chunks[:k]
        if not top_k:
            return 0.0
        # Use a low threshold: any meaningful overlap counts as relevant
        relevant = sum(1 for c in top_k if self._keyword_overlap(reference, c.text) > 0.05)
        return relevant / len(top_k)

    def contextual_recall(self, query: str, chunks: list[RetrievedChunk],
                           ground_truth: Optional[str] = None) -> float:
        """
        What % of the ground truth information appears in retrieved chunks?

        High recall = retriever found everything needed.
        Low recall  = missing context means LLM hallucinates or says I don't know.
        """
        reference = ground_truth if ground_truth else query
        ref_tokens = self._tokenize(reference)
        if not ref_tokens:
            return 0.0
        all_chunk_text = " ".join(c.text for c in chunks)
        chunk_tokens = self._tokenize(all_chunk_text)
        covered = ref_tokens & chunk_tokens
        return len(covered) / len(ref_tokens)

    def faithfulness(self, answer: str, chunks: list[RetrievedChunk]) -> float:
        """
        What % of the answer's claims are grounded in the retrieved context?

        High faithfulness = answer sticks to what the context says.
        Low faithfulness  = hallucination. Answer invents facts not in context.

        This is the number one RAG failure mode to watch.
        """
        answer_tokens = self._tokenize(answer)
        if not answer_tokens:
            return 0.0
        context = " ".join(c.text for c in chunks)
        context_tokens = self._tokenize(context)
        grounded = answer_tokens & context_tokens
        return len(grounded) / len(answer_tokens)

    def answer_relevance(self, query: str, answer: str) -> float:
        """
        Does the answer actually address the query?

        Catches: off-topic answers, refusals, generic responses.
        """
        return self._keyword_overlap(query, answer)

    def answer_correctness(self, answer: str, ground_truth: Optional[str]) -> float:
        """
        How close is the answer to the known ground truth?

        Only meaningful when ground_truth is available (test datasets).
        """
        if not ground_truth:
            return 0.0
        return self._keyword_overlap(answer, ground_truth)

    def calculate_all(self, result: RAGResult) -> RAGResult:
        """Calculate all metrics and attach them to the result object."""
        result.contextual_precision = self.contextual_precision(
            result.query, result.retrieved_chunks, result.ground_truth)
        result.contextual_recall = self.contextual_recall(
            result.query, result.retrieved_chunks, result.ground_truth)
        result.faithfulness = self.faithfulness(
            result.generated_answer, result.retrieved_chunks)
        result.answer_relevance = self.answer_relevance(
            result.query, result.generated_answer)
        result.answer_correctness = self.answer_correctness(
            result.generated_answer, result.ground_truth)
        return result


# ─────────────────────────────────────────────────────────────
# 2. Drift Detector
# ─────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Statistical process control for RAG metric drift.

    Uses two-sample Welch's t-test and Cohen's d effect size to compare
    recent results against an established baseline.

    Alert thresholds:
        warning:  Cohen's d > 0.8  OR  current_mean < baseline - 2 sigma
        critical: Cohen's d > 1.5  OR  current_mean < baseline - 3 sigma

    Why Cohen's d instead of just p-values?
        P-values shrink with sample size. A tiny irrelevant change becomes
        significant with enough data. Effect size tells you if the change
        is actually large enough to matter.
    """

    METRICS = ["contextual_precision", "contextual_recall", "faithfulness",
               "answer_relevance", "answer_correctness"]

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.baseline: dict[str, np.ndarray] = {}
        self.baseline_stats: dict[str, dict] = {}
        self.current: dict[str, list[float]] = defaultdict(list)

    def set_baseline(self, results: list[RAGResult]) -> dict[str, dict]:
        """
        Establish baseline from known-good results.

        You need 30+ results for stable statistics (CLT).
        Returns summary stats for each metric.
        """
        if len(results) < 10:
            raise ValueError(f"Need at least 10 baseline results, got {len(results)}")

        for metric in self.METRICS:
            values = np.array([getattr(r, metric) for r in results])
            self.baseline[metric] = values
            self.baseline_stats[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "p25": float(np.percentile(values, 25)),
                "p50": float(np.percentile(values, 50)),
                "p75": float(np.percentile(values, 75)),
                "n": len(values),
            }
        return self.baseline_stats

    def add_result(self, result: RAGResult) -> list[DriftAlert]:
        """
        Add a new result to the rolling window. Returns any drift alerts.

        Each metric is tracked independently. Embedding drift might tank
        precision while faithfulness stays fine.
        """
        alerts = []
        for metric in self.METRICS:
            value = getattr(result, metric)
            window = self.current[metric]
            window.append(value)
            # Keep only the most recent window_size values
            if len(window) > self.window_size:
                window.pop(0)

            # Need enough data to compare (at least 5 recent values)
            if len(window) < 5 or metric not in self.baseline:
                continue

            alert = self._check_drift(metric, window)
            if alert:
                alerts.append(alert)
        return alerts

    def _check_drift(self, metric: str, current_values: list[float]) -> Optional[DriftAlert]:
        """
        Compare current window against baseline using Welch's t-test + Cohen's d.
        """
        baseline = self.baseline[metric]
        current = np.array(current_values)
        b_stats = self.baseline_stats[metric]

        # Welch's t-test (unequal variance)
        t_stat, p_value = stats.ttest_ind(baseline, current, equal_var=False)

        # Cohen's d effect size
        pooled_std = np.sqrt((np.std(baseline, ddof=1)**2 + np.std(current, ddof=1)**2) / 2)
        if pooled_std < 1e-10:
            cohens_d = 0.0
        else:
            cohens_d = abs(np.mean(baseline) - np.mean(current)) / pooled_std

        current_mean = float(np.mean(current))
        threshold_2sigma = b_stats["mean"] - 2 * b_stats["std"]
        threshold_3sigma = b_stats["mean"] - 3 * b_stats["std"]

        severity = None
        if cohens_d > 1.5 or current_mean < threshold_3sigma:
            severity = "critical"
        elif cohens_d > 0.8 or current_mean < threshold_2sigma:
            severity = "warning"

        if severity and current_mean < b_stats["mean"]:
            return DriftAlert(
                metric_name=metric,
                baseline_mean=b_stats["mean"],
                baseline_std=b_stats["std"],
                current_mean=current_mean,
                drift_score=cohens_d,
                p_value=p_value,
                severity=severity,
                details=(f"{metric}: baseline={b_stats['mean']:.3f} +/- {b_stats['std']:.3f}, "
                         f"current={current_mean:.3f}, Cohen's d={cohens_d:.2f}, p={p_value:.4f}"),
            )
        return None

    def get_status(self) -> pd.DataFrame:
        """Current metrics summary compared to baseline."""
        rows = []
        for metric in self.METRICS:
            b = self.baseline_stats.get(metric, {})
            c = self.current.get(metric, [])
            rows.append({
                "metric": metric,
                "baseline_mean": b.get("mean", 0),
                "baseline_std": b.get("std", 0),
                "current_mean": float(np.mean(c)) if c else 0,
                "current_n": len(c),
                "delta": (float(np.mean(c)) - b.get("mean", 0)) if c else 0,
                "status": "OK" if not c or float(np.mean(c)) >= b.get("mean", 0) - 2*b.get("std", 0) else "DRIFT",
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 3. Failure Simulator
# ─────────────────────────────────────────────────────────────

class FailureSimulator:
    """
    Injects realistic failures into RAG pipeline components.

    7 failure modes at configurable intensity (0.0 = no effect, 1.0 = maximum).

    Why simulate failures?
        You cannot wait for production to break. Simulate known failure patterns,
        verify your drift detector catches them, then deploy with confidence.
    """

    MODES = [
        "embedding_drift",      # Vector similarity degrades
        "retrieval_noise",      # Irrelevant chunks ranked highly
        "context_truncation",   # Chunks removed before LLM sees them
        "model_degradation",    # LLM ignores context, hallucinates
        "latency_spike",        # Response time explodes
        "staleness",            # Chunks are outdated
        "index_corruption",     # Chunks replaced with garbage
    ]

    def __init__(self):
        self.active_mode: Optional[str] = None
        self.intensity: float = 0.0

    def enable(self, mode: str, intensity: float = 0.5):
        """Activate a failure mode. intensity: 0.0 (none) to 1.0 (severe)."""
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode '{mode}'. Available: {self.MODES}")
        self.active_mode = mode
        self.intensity = max(0.0, min(1.0, intensity))

    def disable(self):
        """Turn off failure simulation."""
        self.active_mode = None
        self.intensity = 0.0

    def apply(self, chunks: list[RetrievedChunk], answer: str,
              latency_ms: float) -> tuple[list[RetrievedChunk], str, float]:
        """
        Apply the active failure mode to a RAG result.
        Returns modified (chunks, answer, latency_ms).
        """
        if not self.active_mode:
            return chunks, answer, latency_ms

        method = getattr(self, f"_apply_{self.active_mode}", None)
        if method:
            return method(chunks, answer, latency_ms)
        return chunks, answer, latency_ms

    def _apply_embedding_drift(self, chunks, answer, latency_ms):
        """Reduce similarity scores and shuffle ranking."""
        modified = []
        for c in chunks:
            noise = random.gauss(0, 0.15 * self.intensity)
            new_score = max(0.0, min(1.0, c.score + noise - 0.2 * self.intensity))
            modified.append(RetrievedChunk(
                text=c.text, score=new_score, source=c.source,
                timestamp=c.timestamp, metadata=c.metadata))
        random.shuffle(modified)
        return modified, answer, latency_ms

    def _apply_retrieval_noise(self, chunks, answer, latency_ms):
        """Replace good chunks with irrelevant ones. At intensity=0.7, most chunks are junk."""
        noise_texts = [
            "The weather in Tokyo is typically warm in summer with temperatures reaching 35C.",
            "Python was created by Guido van Rossum and first released in 1991.",
            "The stock market experienced significant volatility in Q3 2024.",
            "Bananas are the most popular fruit in the world by consumption volume.",
            "The FIFA World Cup is held every four years since 1930.",
            "Mount Everest is 8849 metres tall and located on the Nepal Tibet border.",
            "The Roman Empire fell in 476 AD after centuries of decline.",
        ]
        # Replace chunks probabilistically based on intensity
        modified = []
        replaced = 0
        for c in chunks:
            if random.random() < self.intensity:
                modified.append(RetrievedChunk(
                    text=random.choice(noise_texts),
                    score=random.uniform(0.75, 0.95),
                    source="noise_injection",
                ))
                replaced += 1
            else:
                modified.append(c)
        # If most chunks replaced, answer would also degrade in a real system
        # Simulate this: corrupt the answer proportionally
        if replaced / max(len(chunks), 1) > 0.5:
            generic = [
                "Based on the available information, there are several factors to consider.",
                "The context suggests multiple possible interpretations of this topic.",
                "This question touches on a broad area with many related concepts.",
            ]
            if random.random() < self.intensity * 0.6:
                answer = random.choice(generic)
        return modified, answer, latency_ms

    def _apply_context_truncation(self, chunks, answer, latency_ms):
        """Remove chunks before the LLM sees them."""
        keep = max(1, int(len(chunks) * (1 - self.intensity)))
        return chunks[:keep], answer, latency_ms

    def _apply_model_degradation(self, chunks, answer, latency_ms):
        """LLM ignores context and produces generic/hallucinated answers."""
        generic_answers = [
            "Based on my knowledge, I believe the answer involves multiple factors that need consideration.",
            "This is a complex topic with many perspectives to consider.",
            "The answer depends on various circumstances and context.",
            "I think there are several possible answers to this question.",
            "From what I understand, this topic has been widely discussed.",
        ]
        if random.random() < self.intensity:
            answer = random.choice(generic_answers)
        return chunks, answer, latency_ms

    def _apply_latency_spike(self, chunks, answer, latency_ms):
        """Multiply latency by up to 10x."""
        multiplier = 1 + (9 * self.intensity * random.uniform(0.5, 1.0))
        return chunks, answer, latency_ms * multiplier

    def _apply_staleness(self, chunks, answer, latency_ms):
        """Mark chunks as outdated."""
        modified = []
        for c in chunks:
            if random.random() < self.intensity:
                modified.append(RetrievedChunk(
                    text=c.text + " [Note: this information may be outdated as of 2019]",
                    score=c.score * 0.7, source=c.source,
                    timestamp="2019-01-01", metadata={**c.metadata, "stale": True}))
            else:
                modified.append(c)
        return modified, answer, latency_ms

    def _apply_index_corruption(self, chunks, answer, latency_ms):
        """Replace chunk content with garbage."""
        garbage = [
            "asd832j fk32j 29dk corrupt data",
            "NULL NULL NULL ERROR SEGMENT FAULT",
            "DOCTYPE html head title 404 title head",
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "error index_segment_corrupted code 500",
        ]
        modified = []
        for c in chunks:
            if random.random() < self.intensity:
                modified.append(RetrievedChunk(
                    text=random.choice(garbage), score=c.score,
                    source="corrupted", metadata={"corrupted": True}))
            else:
                modified.append(c)
        return modified, answer, latency_ms


# ─────────────────────────────────────────────────────────────
# 4. RAG Testing System (integration class)
# ─────────────────────────────────────────────────────────────

class RAGTestingSystem:
    """
    Main integration class tying together metrics, drift detection,
    and failure simulation.

    Usage:
        system = RAGTestingSystem()
        system.establish_baseline(good_results)
        system.enable_simulation("retrieval_noise", 0.7)
        result, alerts = system.test_query(query, chunks, answer, ground_truth)
        dashboard = system.get_dashboard()
    """

    def __init__(self, window_size: int = 100):
        self.calculator = MetricsCalculator()
        self.detector = DriftDetector(window_size=window_size)
        self.simulator = FailureSimulator()
        self.history: list[RAGResult] = []
        self.alerts: list[DriftAlert] = []

    def establish_baseline(self, results: list[RAGResult]) -> pd.DataFrame:
        """
        Set the baseline from known-good results.
        Call this once with your test suite results before enabling monitoring.
        """
        stats = self.detector.set_baseline(results)
        df = pd.DataFrame(stats).T
        df.index.name = "metric"
        return df

    def enable_simulation(self, mode: str, intensity: float = 0.5):
        """Activate a failure mode for testing drift detection."""
        self.simulator.enable(mode, intensity)

    def disable_simulation(self):
        """Turn off failure simulation."""
        self.simulator.disable()

    def test_query(self, query: str, retrieved_chunks: list[RetrievedChunk],
                   generated_answer: str, ground_truth: Optional[str] = None,
                   latency_ms: float = 0.0) -> tuple[RAGResult, list[DriftAlert]]:
        """
        Test a single RAG query: calculate metrics, check for drift.

        Returns:
            (result, alerts) where result has all metrics, alerts is empty if no drift
        """
        # Apply failure simulation if active
        chunks, answer, lat = self.simulator.apply(
            retrieved_chunks, generated_answer, latency_ms)

        result = RAGResult(
            query=query,
            retrieved_chunks=chunks,
            generated_answer=answer,
            ground_truth=ground_truth,
            latency_ms=lat,
            simulation_mode=self.simulator.active_mode,
        )

        # Calculate metrics
        self.calculator.calculate_all(result)

        # Check for drift
        alerts = self.detector.add_result(result)

        self.history.append(result)
        self.alerts.extend(alerts)

        return result, alerts

    def get_dashboard(self) -> pd.DataFrame:
        """Current metrics status vs baseline."""
        return self.detector.get_status()

    def get_history_df(self) -> pd.DataFrame:
        """All results as a DataFrame for plotting."""
        rows = []
        for r in self.history:
            rows.append({
                "query_id": r.query_id,
                "timestamp": r.timestamp,
                "contextual_precision": r.contextual_precision,
                "contextual_recall": r.contextual_recall,
                "faithfulness": r.faithfulness,
                "answer_relevance": r.answer_relevance,
                "answer_correctness": r.answer_correctness,
                "latency_ms": r.latency_ms,
                "simulation_mode": r.simulation_mode,
            })
        return pd.DataFrame(rows)

    def plot_metrics(self, metrics: list[str] = None, figsize=(14, 8)):
        """Plot metric history with baseline bands and alert markers."""
        import matplotlib.pyplot as plt

        df = self.get_history_df()
        if df.empty:
            print("No history to plot")
            return

        if metrics is None:
            metrics = ["faithfulness", "contextual_precision",
                       "contextual_recall", "answer_relevance"]

        fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
        if len(metrics) == 1:
            axes = [axes]

        colors = {
            "faithfulness": "#e74c3c",
            "contextual_precision": "#3498db",
            "contextual_recall": "#2ecc71",
            "answer_relevance": "#9b59b6",
            "answer_correctness": "#f39c12",
        }

        for ax, metric in zip(axes, metrics):
            values = df[metric].values
            x = range(len(values))
            color = colors.get(metric, "#333")

            ax.plot(x, values, 'o-', color=color, markersize=3, linewidth=1, alpha=0.8)

            # Baseline band
            b = self.detector.baseline_stats.get(metric, {})
            if b:
                mean = b["mean"]
                std = b["std"]
                ax.axhline(mean, color=color, linestyle='--', alpha=0.5,
                           label=f'baseline={mean:.2f}')
                ax.fill_between(x, mean - 2*std, mean + 2*std,
                                alpha=0.1, color=color, label='+/- 2 sigma')

            # Mark simulation periods
            sim_mask = df["simulation_mode"].notna()
            if sim_mask.any():
                sim_indices = df.index[sim_mask].tolist()
                ax.scatter(sim_indices, df.loc[sim_mask, metric],
                          marker='x', color='red', s=50, zorder=5,
                          label='simulated failure')

            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=9)
            ax.legend(fontsize=7, loc='lower left')
            ax.set_ylim(-0.05, 1.05)

        axes[-1].set_xlabel("Query Index")
        fig.suptitle("RAG Metrics Over Time with Drift Detection",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        return fig


# ─────────────────────────────────────────────────────────────
# 5. Synthetic Data Generator
# ─────────────────────────────────────────────────────────────

class SyntheticRAGData:
    """
    Generates realistic synthetic RAG query/answer/chunk triples
    for testing without needing a real vector database.
    """

    QA_PAIRS = [
        {"q": "What is the capital of France?",
         "a": "The capital of France is Paris.",
         "chunks": ["Paris is the capital and largest city of France.",
                    "France is a country in Western Europe.",
                    "The Eiffel Tower is located in Paris, France."]},
        {"q": "Who wrote Romeo and Juliet?",
         "a": "William Shakespeare wrote Romeo and Juliet.",
         "chunks": ["Romeo and Juliet is a tragedy written by William Shakespeare early in his career.",
                    "Shakespeare was an English playwright born in 1564.",
                    "The play was first published in 1597."]},
        {"q": "What is photosynthesis?",
         "a": "Photosynthesis is the process by which plants convert sunlight, CO2 and water into glucose and oxygen.",
         "chunks": ["Photosynthesis is a process used by plants to convert light energy into chemical energy.",
                    "During photosynthesis, carbon dioxide and water are converted to glucose and oxygen.",
                    "Chlorophyll in plant cells absorbs sunlight to drive photosynthesis."]},
        {"q": "What is the speed of light?",
         "a": "The speed of light in a vacuum is approximately 299,792,458 metres per second.",
         "chunks": ["The speed of light in vacuum is exactly 299,792,458 m/s.",
                    "Light speed is a fundamental physical constant denoted c.",
                    "Nothing with mass can travel at the speed of light."]},
        {"q": "What is machine learning?",
         "a": "Machine learning is a subset of AI where systems learn from data to improve without explicit programming.",
         "chunks": ["Machine learning is a branch of artificial intelligence focused on building systems that learn from data.",
                    "ML algorithms improve automatically through experience and data.",
                    "Common ML approaches include supervised, unsupervised, and reinforcement learning."]},
        {"q": "Who painted the Mona Lisa?",
         "a": "The Mona Lisa was painted by Leonardo da Vinci.",
         "chunks": ["The Mona Lisa is a portrait painted by Italian artist Leonardo da Vinci.",
                    "Da Vinci began painting the Mona Lisa around 1503.",
                    "The painting hangs in the Louvre Museum in Paris."]},
        {"q": "What is the largest planet in our solar system?",
         "a": "Jupiter is the largest planet in our solar system.",
         "chunks": ["Jupiter is the largest planet in our solar system with a mass over 300 times that of Earth.",
                    "Jupiter is a gas giant composed mainly of hydrogen and helium.",
                    "The Great Red Spot on Jupiter is a storm larger than Earth."]},
        {"q": "What is DNA?",
         "a": "DNA or deoxyribonucleic acid is a molecule that carries genetic instructions for life.",
         "chunks": ["DNA stands for deoxyribonucleic acid and contains genetic instructions.",
                    "The structure of DNA is a double helix, discovered by Watson and Crick in 1953.",
                    "DNA is found in the nucleus of every cell and encodes proteins."]},
        {"q": "What causes earthquakes?",
         "a": "Earthquakes are caused by sudden release of energy in the Earth crust, usually due to tectonic plate movements.",
         "chunks": ["Earthquakes occur when tectonic plates shift and release built-up stress.",
                    "The point where an earthquake originates is called the focus or hypocenter.",
                    "Most earthquakes occur along plate boundaries called fault lines."]},
        {"q": "What is the boiling point of water?",
         "a": "Water boils at 100 degrees Celsius or 212 degrees Fahrenheit at standard atmospheric pressure.",
         "chunks": ["Water boils at 100 degrees Celsius at sea level atmospheric pressure.",
                    "The boiling point of water decreases at higher altitudes due to lower pressure.",
                    "Water molecules gain enough energy to transition from liquid to gas at the boiling point."]},
    ]

    @classmethod
    def generate_good_result(cls, noise: float = 0.05) -> RAGResult:
        """Generate a single good-quality RAG result with mild noise."""
        qa = random.choice(cls.QA_PAIRS)
        chunks = [RetrievedChunk(
            text=t, score=random.uniform(0.80, 0.99),
            source="knowledge_base", timestamp="2024-01-01"
        ) for t in qa["chunks"]]

        answer = qa["a"]
        if random.random() < noise:
            answer = answer + " This is well established in the literature."

        result = RAGResult(
            query=qa["q"], retrieved_chunks=chunks,
            generated_answer=answer, ground_truth=qa["a"],
            latency_ms=random.uniform(50, 200),
        )
        MetricsCalculator().calculate_all(result)
        return result

    @classmethod
    def generate_baseline(cls, n: int = 50) -> list[RAGResult]:
        """Generate n good results for establishing a baseline."""
        return [cls.generate_good_result() for _ in range(n)]

    @classmethod
    def get_test_query(cls) -> dict:
        """Get a random query with chunks, answer, and ground truth."""
        qa = random.choice(cls.QA_PAIRS)
        chunks = [RetrievedChunk(
            text=t, score=random.uniform(0.80, 0.99),
            source="knowledge_base"
        ) for t in qa["chunks"]]
        return {
            "query": qa["q"],
            "chunks": chunks,
            "answer": qa["a"],
            "ground_truth": qa["a"],
        }
