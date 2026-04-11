"""
Evaluation Runner
==================
Orchestrates pipeline queries, evaluations, and Phoenix posting.
Strategy: run cheap code evals on everything first, then only run
expensive LLM evals on rows where code evals flag problems.
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd
from phoenix.evals import evaluate_dataframe

from rag.config import PHOENIX_URL
from rag.dataset import QAPair
from rag.drift import DriftDetector
from rag.evaluators.code_evals import (
    answer_length_ok,
    contains_expected,
    hit_rate_at_k,
    mrr,
    precision_at_k,
)
from rag.evaluators.llm_evals import (
    correctness_eval,
    hallucination_eval,
    relevance_eval,
)
from rag.evaluators.poster import post_code_eval_scores, post_eval_scores
from rag.pipeline import RAGPipeline, RAGResult


class EvaluationRunner:
    """
    Orchestrates pipeline queries, evaluations, and Phoenix posting.

    Strategy:
      1. Run pipeline.query() for each QA pair (captures trace_id)
      2. Run code evaluators on all rows (instant, free)
      3. Run LLM evaluators ONLY on flagged rows (slow, expensive)
      4. Post all scores to Phoenix as trace annotations
      5. If drift detector present, check for drift
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        post_to_phoenix: bool = True,
        drift_detector: Optional[DriftDetector] = None,
        phoenix_url: str = PHOENIX_URL,
        # Code eval thresholds for LLM eval gating
        hit_rate_threshold: float = 0.5,
        answer_length_min: int = 10,
    ):
        self.pipeline = pipeline
        self.post_to_phoenix = post_to_phoenix
        self.drift_detector = drift_detector
        self.phoenix_url = phoenix_url
        self.hit_rate_threshold = hit_rate_threshold
        self.answer_length_min = answer_length_min

    def run(
        self,
        qa_pairs: list[QAPair],
        run_llm_evals: bool = True,
        force_all_llm_evals: bool = False,
        progress_callback=None,
    ) -> pd.DataFrame:
        """
        Run the full evaluation pipeline.

        Args:
            qa_pairs: List of QA pairs to evaluate
            run_llm_evals: Whether to run LLM-as-judge evaluators at all
            force_all_llm_evals: If True, run LLM evals on ALL rows (skip gating)
            progress_callback: Optional callable for progress updates

        Returns:
            DataFrame with all results and scores
        """
        # Step 1: Run pipeline queries
        if progress_callback:
            progress_callback("Running pipeline queries...")

        results: list[RAGResult] = []
        for i, qa in enumerate(qa_pairs):
            if progress_callback:
                progress_callback(f"  Query {i+1}/{len(qa_pairs)}: {qa.question[:50]}...")
            result = self.pipeline.query(
                question=qa.question,
                ground_truth=qa.ground_truth_answer,
                relevant_doc_ids=qa.relevant_doc_ids,
                cluster=qa.cluster,
            )
            results.append(result)

        # Build results DataFrame
        rows = []
        for r in results:
            rows.append({
                "question": r.question,
                "generated_answer": r.answer,
                "expected_answer": r.ground_truth or "",
                "context": r.context,
                "trace_id": r.trace_id,
                "latency_ms": r.latency_ms,
                "retrieved_doc_ids": r.retrieved_doc_ids,
                "relevant_doc_ids": r.relevant_doc_ids or [],
                "cluster": r.cluster,
            })
        df = pd.DataFrame(rows)

        # Step 2: Code evaluators (all rows)
        if progress_callback:
            progress_callback("Running code evaluators...")

        df["hit_rate"] = df.apply(
            lambda row: hit_rate_at_k(row["retrieved_doc_ids"], row["relevant_doc_ids"]),
            axis=1,
        )
        df["mrr"] = df.apply(
            lambda row: mrr(row["retrieved_doc_ids"], row["relevant_doc_ids"]),
            axis=1,
        )
        df["precision_at_k"] = df.apply(
            lambda row: precision_at_k(row["retrieved_doc_ids"], row["relevant_doc_ids"]),
            axis=1,
        )
        df["contains_expected"] = df.apply(
            lambda row: contains_expected(row["generated_answer"], row["expected_answer"]),
            axis=1,
        )
        df["answer_length_ok"] = df.apply(
            lambda row: answer_length_ok(row["generated_answer"]),
            axis=1,
        )

        # Post code eval scores to Phoenix
        if self.post_to_phoenix:
            trace_ids = df["trace_id"].tolist()
            for metric in ["hit_rate", "mrr", "precision_at_k", "contains_expected", "answer_length_ok"]:
                try:
                    post_code_eval_scores(
                        trace_ids=trace_ids,
                        metric_name=metric,
                        scores=df[metric].tolist(),
                        phoenix_url=self.phoenix_url,
                    )
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"  Warning: failed to post {metric}: {e}")

        # Step 3: LLM evaluators (gated by code eval results)
        if run_llm_evals:
            if force_all_llm_evals:
                llm_eval_mask = pd.Series([True] * len(df))
            else:
                # Gate: only run LLM evals on rows with potential issues
                llm_eval_mask = (
                    (df["hit_rate"] < self.hit_rate_threshold)
                    | (df["answer_length_ok"] < 1.0)
                    | (df["contains_expected"] < 0.1)
                )

            flagged_count = llm_eval_mask.sum()
            if progress_callback:
                progress_callback(
                    f"Running LLM evals on {flagged_count}/{len(df)} flagged rows..."
                )

            # Initialize LLM eval columns
            df["hallucination_score"] = None
            df["relevance_score"] = None
            df["correctness_score"] = None

            if flagged_count > 0:
                flagged_df = df[llm_eval_mask].copy()

                # Run hallucination eval
                if progress_callback:
                    progress_callback("  Running hallucination eval...")
                hall_results = evaluate_dataframe(
                    dataframe=flagged_df,
                    evaluators=[hallucination_eval],
                )
                if "hallucination_score" in hall_results.columns:
                    for idx, val in zip(flagged_df.index, hall_results["hallucination_score"]):
                        df.at[idx, "hallucination_score"] = val

                # Run relevance eval
                if progress_callback:
                    progress_callback("  Running relevance eval...")
                rel_results = evaluate_dataframe(
                    dataframe=flagged_df,
                    evaluators=[relevance_eval],
                )
                if "relevance_score" in rel_results.columns:
                    for idx, val in zip(flagged_df.index, rel_results["relevance_score"]):
                        df.at[idx, "relevance_score"] = val

                # Run correctness eval
                if progress_callback:
                    progress_callback("  Running correctness eval...")
                corr_results = evaluate_dataframe(
                    dataframe=flagged_df,
                    evaluators=[correctness_eval],
                )
                if "correctness_score" in corr_results.columns:
                    for idx, val in zip(flagged_df.index, corr_results["correctness_score"]):
                        df.at[idx, "correctness_score"] = val

                # Post LLM eval scores to Phoenix
                if self.post_to_phoenix:
                    flagged_trace_ids = flagged_df["trace_id"].tolist()
                    for eval_name, eval_col in [
                        ("hallucination", "hallucination_score"),
                        ("relevance", "relevance_score"),
                        ("correctness", "correctness_score"),
                    ]:
                        try:
                            eval_series = df.loc[llm_eval_mask, eval_col]
                            eval_df_temp = pd.DataFrame({f"{eval_name}_score": eval_series.values})
                            post_eval_scores(
                                eval_df=eval_df_temp,
                                eval_name=eval_name,
                                trace_ids=flagged_trace_ids,
                                phoenix_url=self.phoenix_url,
                            )
                        except Exception as e:
                            if progress_callback:
                                progress_callback(f"  Warning: failed to post {eval_name}: {e}")

        # Step 4: Extract numeric scores for drift detection
        def _extract_score(val):
            if isinstance(val, dict):
                return val.get("score", 0.0)
            if isinstance(val, (int, float)):
                return float(val)
            return None

        for col in ["hallucination_score", "relevance_score", "correctness_score"]:
            if col in df.columns:
                df[col + "_numeric"] = df[col].apply(_extract_score)

        # Step 5: Drift detection
        if self.drift_detector:
            if progress_callback:
                progress_callback("Checking for drift...")

            drift_df = df[["hit_rate", "mrr", "latency_ms"]].copy()
            for col in ["hallucination_score", "relevance_score", "correctness_score"]:
                numeric_col = col + "_numeric"
                if numeric_col in df.columns:
                    drift_df[col] = df[numeric_col]

            if self.drift_detector.has_baseline:
                report = self.drift_detector.check(drift_df)
                if progress_callback:
                    progress_callback(report.summary())
            else:
                self.drift_detector.fit(drift_df)
                if progress_callback:
                    progress_callback("Baseline saved — no drift check on first run.")

        if progress_callback:
            progress_callback(f"Evaluation complete: {len(df)} queries processed.")

        return df
