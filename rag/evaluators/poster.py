"""
Phoenix Annotation Poster
==========================
POST evaluation scores to Phoenix /v1/trace_annotations so they appear
in the Tracing UI alongside the original query spans.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import requests

from rag.config import PHOENIX_URL


def post_eval_scores(
    eval_df: pd.DataFrame,
    eval_name: str,
    trace_ids: list[str],
    annotator_kind: str = "LLM",
    phoenix_url: str = PHOENIX_URL,
) -> int:
    """
    POST evaluate_dataframe results to Phoenix as trace annotations.

    The eval_df should have a column named '{eval_name}_score' containing
    dicts with 'score', 'label', and optional 'explanation' keys —
    this is what phoenix.evals.evaluate_dataframe produces.

    Returns the number of annotations successfully posted.
    """
    col = f"{eval_name}_score"
    if col not in eval_df.columns:
        # Try without _score suffix
        col = eval_name
        if col not in eval_df.columns:
            return 0

    annotations = []
    for trace_id, val in zip(trace_ids, eval_df[col]):
        if val is None:
            continue
        if isinstance(val, dict):
            result = {
                "score": val.get("score"),
                "label": val.get("label"),
                "explanation": val.get("explanation"),
            }
        elif isinstance(val, (int, float)):
            result = {"score": float(val)}
        else:
            continue

        annotations.append({
            "trace_id": trace_id,
            "name": eval_name,
            "annotator_kind": annotator_kind,
            "result": result,
        })

    if not annotations:
        return 0

    resp = requests.post(
        f"{phoenix_url}/v1/trace_annotations",
        json={"data": annotations},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return len(annotations)


def post_code_eval_scores(
    trace_ids: list[str],
    metric_name: str,
    scores: list[float],
    phoenix_url: str = PHOENIX_URL,
) -> int:
    """Post simple numeric scores (from code evaluators) as trace annotations."""
    annotations = []
    for trace_id, score in zip(trace_ids, scores):
        annotations.append({
            "trace_id": trace_id,
            "name": metric_name,
            "annotator_kind": "CODE",
            "result": {"score": float(score)},
        })

    if not annotations:
        return 0

    resp = requests.post(
        f"{phoenix_url}/v1/trace_annotations",
        json={"data": annotations},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return len(annotations)
