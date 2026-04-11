"""
Agent Evaluators
=================
Evaluation metrics specific to the ReAct agent:
tool selection accuracy, step efficiency, final answer correctness.
"""

from __future__ import annotations

from phoenix.evals import create_evaluator


@create_evaluator(name="tool_accuracy", kind="code")
def tool_accuracy(optimal_tool: str, trajectory: list) -> float:
    """1.0 if the optimal tool was used in the trajectory, else 0.0."""
    tools_used = [t.get("tool") or getattr(t, "tool", None) for t in trajectory]
    tools_used = [t for t in tools_used if t]
    if not tools_used:
        return 0.0
    return 1.0 if optimal_tool in tools_used else 0.0


@create_evaluator(name="step_efficiency", kind="code")
def step_efficiency(trajectory: list) -> float:
    """Score based on number of steps taken (fewer = better)."""
    total = len(trajectory)
    if total <= 2:
        return 1.0
    elif total <= 3:
        return 0.8
    elif total <= 5:
        return 0.5
    return 0.2


@create_evaluator(name="final_answer_correct", kind="code")
def final_answer_correct(expected: str, generated_answer: str) -> float:
    """Keyword overlap between agent's final answer and expected answer."""
    def tokenize(text: str) -> set[str]:
        return {w.lower() for w in str(text).split() if len(w) >= 2}

    expected_tokens = tokenize(expected)
    answer_tokens = tokenize(generated_answer)
    if not expected_tokens:
        return 0.0
    intersection = expected_tokens & answer_tokens
    return len(intersection) / len(expected_tokens)
