"""
=============================================================================
PARADIGM 1: SINGLE AGENT
=============================================================================
The simplest possible approach — one LLM call with a well-crafted prompt.

This is the baseline. Every other paradigm should beat it on quality, but
this one wins on latency and simplicity. It's also the control: if the
multi-agent or swarm approach scores *lower* than a single call, something
is wrong with the more complex system.

Architecture: User → [System Prompt + Backlog] → LLM → Structured Output

When to use this pattern in production:
  - The problem is well-defined and doesn't benefit from decomposition
  - Latency matters more than thoroughness
  - You need a quick first draft before deeper analysis
=============================================================================
"""

import time
import json
from dataclasses import dataclass, field

from openai import OpenAI

from observability.logger import ObservabilityLogger, RunEvent, generate_run_id


@dataclass
class ParadigmResult:
    """Standardised result object returned by every paradigm."""
    output: str
    paradigm: str
    run_id: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    metadata: dict = field(default_factory=dict)


# System prompt — crafted to elicit structured, actionable output
SYSTEM_PROMPT = """You are an experienced engineering manager with 15 years of experience leading software teams.

You will be given a backlog of items that a software team is considering for next quarter.

Your task: Recommend the top 3 priorities with clear reasoning.

For each priority, explain:
1. WHY it should be prioritized (business impact, technical debt reduction, team velocity)
2. What RISKS exist if it's deprioritized
3. Rough T-shirt size estimate (S/M/L/XL)

Also briefly explain why the remaining items should wait.

Be specific and actionable — a PM should be able to take your output directly into sprint planning."""


def run(
    backlog: list[str],
    model: str,
    api_base: str,
    obs: ObservabilityLogger,
) -> ParadigmResult:
    """
    Run the single-agent paradigm.

    This is intentionally simple — one prompt, one call, one result.
    The quality comes entirely from prompt engineering.

    Args:
        backlog:  List of backlog item descriptions
        model:    Model ID for the LLM
        api_base: OpenAI-compatible API base URL
        obs:      Observability logger instance

    Returns:
        ParadigmResult with the recommendation
    """
    run_id = generate_run_id("single")
    client = OpenAI(base_url=api_base, api_key="not-needed")

    # Format the backlog into the user message
    backlog_text = "\n".join(f"{i+1}. {item}" for i, item in enumerate(backlog))
    user_message = f"""Here is the team's backlog for next quarter prioritization:

{backlog_text}

What should we prioritize and why?"""

    start = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=4096,
    )

    latency_ms = (time.time() - start) * 1000
    output = response.choices[0].message.content

    # Extract token counts from response
    prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0

    # Log the call
    obs.log_call(RunEvent(
        run_id=run_id,
        paradigm="single",
        agent_id="main",
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        input_text=user_message[:500],
        output_text=output[:500],
    ))

    return ParadigmResult(
        output=output,
        paradigm="single",
        run_id=run_id,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=model,
    )
