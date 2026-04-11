"""
Deterministic Code Evaluators
==============================
Instant, free retrieval and answer quality metrics.
All functions are pure — no LLM calls, no network.
"""

from __future__ import annotations


def hit_rate_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 5,
) -> float:
    """1.0 if any relevant doc appears in top-K, else 0.0."""
    top_k = set(retrieved_ids[:k])
    return 1.0 if top_k & set(relevant_ids) else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Mean Reciprocal Rank — 1/rank of first relevant doc, or 0.0."""
    relevant_set = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 5,
) -> float:
    """Fraction of top-K results that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant_ids)
    return sum(1 for d in top_k if d in relevant_set) / len(top_k)


def contains_expected(answer: str, expected: str) -> float:
    """Keyword overlap between answer and expected answer (Jaccard similarity)."""
    def tokenize(text: str) -> set[str]:
        return {w.lower() for w in text.split() if len(w) >= 2}

    answer_tokens = tokenize(answer)
    expected_tokens = tokenize(expected)
    if not expected_tokens:
        return 0.0
    intersection = answer_tokens & expected_tokens
    union = answer_tokens | expected_tokens
    return len(intersection) / len(union) if union else 0.0


def exact_match(answer: str, expected: str) -> float:
    """1.0 if normalized answer matches expected, else 0.0."""
    def normalize(text: str) -> str:
        return " ".join(text.lower().split())
    return 1.0 if normalize(answer) == normalize(expected) else 0.0


def answer_length_ok(answer: str, min_words: int = 10, max_words: int = 500) -> float:
    """1.0 if answer word count is within bounds, else 0.0."""
    word_count = len(answer.split())
    return 1.0 if min_words <= word_count <= max_words else 0.0


def context_precision(answer: str, context_chunks: list[str]) -> float:
    """Approximate context precision: fraction of retrieved chunks whose tokens
    overlap meaningfully with the answer.

    A chunk is considered 'used' if ≥15% of its unique content words appear in
    the answer. This is a heuristic — a true precision score would require the
    LLM to cite sources explicitly.

    Returns a float in [0, 1]. Lower = the model ignored most of what was
    retrieved (noise in the prompt). Higher = retrieved chunks were on-topic.
    """
    if not context_chunks or not answer:
        return 0.0

    def content_tokens(text: str) -> set[str]:
        stopwords = {
            "the", "a", "an", "is", "in", "on", "at", "of", "to", "and",
            "or", "it", "its", "be", "was", "were", "are", "by", "for",
            "as", "with", "this", "that", "from", "have", "has", "had",
            "not", "but", "also", "than", "so", "if", "he", "she", "they",
            "we", "you", "i", "do", "did", "will", "can", "about",
        }
        return {
            w.lower().strip(".,;:\"'()[]")
            for w in text.split()
            if len(w) >= 3 and w.lower() not in stopwords
        }

    answer_tokens = content_tokens(answer)
    if not answer_tokens:
        return 0.0

    used = 0
    for chunk in context_chunks:
        chunk_tokens = content_tokens(chunk)
        if not chunk_tokens:
            continue
        overlap = len(chunk_tokens & answer_tokens) / len(chunk_tokens)
        if overlap >= 0.15:
            used += 1

    return used / len(context_chunks)
