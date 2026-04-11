"""
Drift Detector
===============
Statistical drift detection over evaluation scores.
Compares baseline vs current run using z-scores.
Persists baseline to disk and posts warning spans to Phoenix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from opentelemetry import trace

from rag.config import BASELINE_PATH, PHOENIX_URL

tracer = trace.get_tracer("rag.drift")

TRACKED_METRICS = [
    "hit_rate", "mrr", "hallucination_score",
    "relevance_score", "correctness_score", "latency_ms",
]


@dataclass
class MetricDrift:
    """Drift status for a single metric."""
    metric: str
    baseline_mean: float
    baseline_std: float
    current_mean: float
    delta: float
    z_score: float
    is_drifted: bool


@dataclass
class DriftReport:
    """Aggregate drift report across all tracked metrics."""
    metrics: list[MetricDrift]
    any_drift: bool = False

    def __post_init__(self):
        self.any_drift = any(m.is_drifted for m in self.metrics)

    def summary(self) -> str:
        lines = ["Drift Report", "=" * 50]
        for m in self.metrics:
            flag = "⚠ DRIFT" if m.is_drifted else "✓ OK"
            lines.append(
                f"  {m.metric:25s}  baseline={m.baseline_mean:.3f}±{m.baseline_std:.3f}  "
                f"current={m.current_mean:.3f}  z={m.z_score:+.2f}  {flag}"
            )
        return "\n".join(lines)


class DriftDetector:
    """
    Compares a baseline evaluation run against a current run and flags regressions.

    Uses z-scores (|z| > 2.0 = drift). After the first clean run, saves
    baseline stats to baseline.json. On subsequent runs, loads and compares.
    """

    def __init__(
        self,
        baseline_path: str | Path = BASELINE_PATH,
        z_threshold: float = 2.0,
    ):
        self.baseline_path = Path(baseline_path)
        self.z_threshold = z_threshold
        self._baseline: Optional[dict[str, dict]] = None

    @property
    def has_baseline(self) -> bool:
        return self._baseline is not None or self.baseline_path.exists()

    def _load_baseline(self) -> dict[str, dict]:
        if self._baseline is not None:
            return self._baseline
        if self.baseline_path.exists():
            with open(self.baseline_path) as f:
                self._baseline = json.load(f)
            return self._baseline
        return {}

    def fit(self, results: pd.DataFrame) -> None:
        """
        Compute and save baseline statistics (mean/std per metric).
        Call this after the first clean evaluation run.
        """
        baseline = {}
        for metric in TRACKED_METRICS:
            if metric in results.columns:
                values = results[metric].dropna()
                if len(values) > 0:
                    baseline[metric] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.1,
                        "n": int(len(values)),
                    }
        self._baseline = baseline
        with open(self.baseline_path, "w") as f:
            json.dump(baseline, f, indent=2)

    def check(self, current: pd.DataFrame) -> DriftReport:
        """
        Compare current evaluation results against baseline.
        Returns DriftReport with per-metric z-scores and drift flags.
        Posts a drift_check span to Phoenix if drift is detected.
        """
        baseline = self._load_baseline()
        if not baseline:
            raise ValueError(
                "No baseline found. Call fit() first or ensure baseline.json exists."
            )

        metric_drifts = []
        for metric in TRACKED_METRICS:
            if metric not in baseline or metric not in current.columns:
                continue
            b = baseline[metric]
            values = current[metric].dropna()
            if len(values) == 0:
                continue

            current_mean = float(np.mean(values))
            delta = current_mean - b["mean"]
            std = max(b["std"], 0.05)  # floor to avoid division by zero
            z_score = delta / std

            # For latency, higher is worse; for all others, lower is worse
            if metric == "latency_ms":
                is_drifted = z_score > self.z_threshold
            else:
                is_drifted = z_score < -self.z_threshold

            metric_drifts.append(MetricDrift(
                metric=metric,
                baseline_mean=b["mean"],
                baseline_std=b["std"],
                current_mean=current_mean,
                delta=delta,
                z_score=z_score,
                is_drifted=is_drifted,
            ))

        report = DriftReport(metrics=metric_drifts)

        # Post warning span to Phoenix if drift detected
        if report.any_drift:
            self._post_drift_warning(report)

        return report

    def _post_drift_warning(self, report: DriftReport) -> None:
        """Create a warning span in Phoenix so drift appears in Tracing timeline."""
        with tracer.start_as_current_span("drift_warning") as span:
            drifted = [m for m in report.metrics if m.is_drifted]
            span.set_attribute("drift.detected", True)
            span.set_attribute("drift.num_metrics", len(drifted))
            span.set_attribute("drift.metrics", str([m.metric for m in drifted]))
            for m in drifted:
                span.set_attribute(f"drift.{m.metric}.z_score", m.z_score)
                span.set_attribute(f"drift.{m.metric}.baseline_mean", m.baseline_mean)
                span.set_attribute(f"drift.{m.metric}.current_mean", m.current_mean)
