"""
LLM-as-Judge Evaluators
========================
Hallucination, relevance, and correctness scoring using local Qwen.
Uses phoenix.evals.create_evaluator for compatibility with evaluate_dataframe.
"""

from __future__ import annotations

import asyncio
import re
import time

from openai import OpenAI
from phoenix.evals import create_evaluator

from rag.config import (
    JUDGE_MAX_TOKENS,
    JUDGE_TEMPERATURE,
    LLM_API_KEY,
    LLM_BASE_URL,
    MAX_CONCURRENT_JUDGE,
    MODEL_NAME,
)

_client = None
_sem = asyncio.Semaphore(MAX_CONCURRENT_JUDGE)


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    return _client


def _llm_judge_call(prompt: str, max_tokens: int = JUDGE_MAX_TOKENS) -> str:
    """Synchronous judge call with Qwen think-tag stripping."""
    client = _get_client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt + " /no_think"}],
        max_tokens=max_tokens,
        temperature=JUDGE_TEMPERATURE,
    )
    msg = resp.choices[0].message
    raw = msg.content or ""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if not raw:
        extra = getattr(msg, "model_extra", {}) or {}
        raw = extra.get("reasoning_content", "")
    return raw


@create_evaluator(name="hallucination", kind="llm")
def hallucination_eval(context: str, generated_answer: str) -> dict:
    """Binary: is the answer faithful to the context?"""
    prompt = (
        "You are a hallucination detector. Given a context and a response, determine "
        "if the response contains ONLY information supported by the context.\n\n"
        "Examples:\n"
        "Context: 'The Eiffel Tower is 330m tall.' Response: 'The Eiffel Tower is 330 meters.' -> FAITHFUL\n"
        "Context: 'The Eiffel Tower is 330m tall.' Response: 'The Eiffel Tower is 330m and was built in 1889.' -> HALLUCINATED\n\n"
        f"Context: {context}\n\n"
        f"Response: {generated_answer}\n\n"
        "Reply with exactly one word: FAITHFUL or HALLUCINATED."
    )
    raw = _llm_judge_call(prompt, max_tokens=32).strip().upper()
    if "FAITHFUL" in raw and "HALLUCINATED" not in raw:
        return {"label": "FAITHFUL", "score": 1.0}
    elif "HALLUCINATED" in raw:
        return {"label": "HALLUCINATED", "score": 0.0}
    return {"label": "UNKNOWN", "score": 0.5}


@create_evaluator(name="relevance", kind="llm")
def relevance_eval(question: str, generated_answer: str) -> dict:
    """Categorical: how relevant is the answer to the question?"""
    prompt = (
        "You are evaluating answer relevance.\n\n"
        f"Question: {question}\n\n"
        f"Answer: {generated_answer}\n\n"
        "How relevant is the answer to the question?\n"
        "Reply with exactly one word: HIGH, MEDIUM, or LOW."
    )
    raw = _llm_judge_call(prompt, max_tokens=32).strip().upper()
    if "HIGH" in raw:
        return {"label": "HIGH", "score": 1.0}
    elif "MEDIUM" in raw:
        return {"label": "MEDIUM", "score": 0.5}
    elif "LOW" in raw:
        return {"label": "LOW", "score": 0.0}
    return {"label": "UNKNOWN", "score": 0.5}


@create_evaluator(name="correctness", kind="llm")
def correctness_eval(expected_answer: str, generated_answer: str) -> dict:
    """Continuous 0-1: how correct is the answer vs ground truth?"""
    prompt = (
        "You are evaluating answer correctness.\n\n"
        f"Ground truth: {expected_answer}\n\n"
        f"Generated answer: {generated_answer}\n\n"
        "How correct is the generated answer compared to the ground truth? "
        "Judge semantic equivalence, not word-for-word match.\n"
        "Reply with ONLY a score from 0.0 to 1.0."
    )
    raw = _llm_judge_call(prompt, max_tokens=64)
    score = _parse_score(raw)
    return {"score": score}


def _parse_score(text: str) -> float:
    """Extract a float 0-1 from LLM output."""
    text = text.strip().rstrip(".")

    # JSON pattern
    m = re.search(r'"score"\s*:\s*([\d.]+)', text)
    if m:
        return max(0.0, min(1.0, float(m.group(1).rstrip("."))))

    # "Score: X" pattern
    m = re.search(r'[Ss]core\s*[:=]\s*([\d.]+)', text)
    if m:
        return max(0.0, min(1.0, float(m.group(1).rstrip("."))))

    # Any float 0-1
    floats = re.findall(r'\b(0\.\d+|1\.0|0|1)\b', text)
    if floats:
        return max(0.0, min(1.0, float(floats[0])))

    # Fallback: any number
    nums = re.findall(r'(\d+\.?\d*)', text)
    for n in nums:
        val = float(n)
        if 0 <= val <= 1:
            return val
        elif 1 < val <= 10:
            return val / 10.0
        elif 10 < val <= 100:
            return val / 100.0
    return 0.0
