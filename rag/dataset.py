"""
QA Dataset
===========
Load/save QA pairs and upload them to Phoenix Datasets for experiment tracking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from rag.config import PHOENIX_URL, QA_PAIRS_PATH


@dataclass
class QAPair:
    """A ground truth question-answer pair with metadata."""
    question: str
    ground_truth_answer: str
    source_article: str
    cluster: str
    relevant_doc_ids: Optional[list[str]] = None


class QADataset:
    """Load, save, and upload QA pairs."""

    def __init__(self, path: str | Path = QA_PAIRS_PATH):
        self.path = Path(path)
        self.pairs: list[QAPair] = []

    def load(self) -> list[QAPair]:
        """Load QA pairs from JSONL file."""
        pairs = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                pairs.append(QAPair(
                    question=d["question"],
                    ground_truth_answer=d["ground_truth_answer"],
                    source_article=d["source_article"],
                    cluster=d["cluster"],
                    relevant_doc_ids=d.get("relevant_doc_ids"),
                ))
        self.pairs = pairs
        return pairs

    def save(self, pairs: Optional[list[QAPair]] = None) -> None:
        """Save QA pairs to JSONL file."""
        pairs = pairs or self.pairs
        with open(self.path, "w") as f:
            for p in pairs:
                f.write(json.dumps({
                    "question": p.question,
                    "ground_truth_answer": p.ground_truth_answer,
                    "source_article": p.source_article,
                    "cluster": p.cluster,
                    "relevant_doc_ids": p.relevant_doc_ids,
                }) + "\n")

    def upload_to_phoenix(
        self,
        dataset_name: str = "rag-qa-pairs",
        phoenix_url: str = PHOENIX_URL,
    ) -> str:
        """
        Upload QA pairs as a Phoenix dataset.
        Returns the dataset ID.
        """
        if not self.pairs:
            self.load()

        # Create dataset
        resp = requests.post(
            f"{phoenix_url}/v1/datasets",
            json={
                "name": dataset_name,
                "description": f"RAG QA pairs: {len(self.pairs)} questions across 3 clusters",
            },
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        dataset_id = resp.json()["data"]["id"]

        # Upload examples
        examples = []
        for p in self.pairs:
            examples.append({
                "input": {"question": p.question},
                "output": {"answer": p.ground_truth_answer},
                "metadata": {
                    "source_article": p.source_article,
                    "cluster": p.cluster,
                },
            })

        resp = requests.post(
            f"{phoenix_url}/v1/datasets/{dataset_id}/examples",
            json={"examples": examples},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()

        return dataset_id
