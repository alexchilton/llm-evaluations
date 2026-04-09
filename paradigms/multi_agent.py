"""
=============================================================================
PARADIGM 2: MULTI-AGENT PIPELINE
=============================================================================
Three specialised agents in a sequential pipeline, each with a focused role:

  Analyst → Critic → Synthesiser

This is the classic "decompose and conquer" pattern. Each agent sees the
output of the previous one, building on it or challenging it.

Why three agents instead of two or five?
  - Analyst alone would be just a single agent with extra steps
  - Adding a Critic creates adversarial pressure (finds weak reasoning)
  - The Synthesiser resolves the tension into a final recommendation
  - More agents add latency without proportional quality gains for this task

Architecture:
  Backlog → [Analyst: score items] → JSON
         → [Critic: challenge scores] → JSON
         → [Synthesiser: final recommendation] → Output

The handoff format is structured JSON — this gives each agent a clean
contract to work with, and lets us log/eval each transition separately.

smolagents integration:
  We use smolagents ToolCallingAgent for each step, giving them access to
  a simple scoring tool. The framework handles prompt construction,
  tool calling, and output parsing.
=============================================================================
"""

import time
import json
from dataclasses import dataclass, field

from openai import OpenAI

from observability.logger import (
    ObservabilityLogger, RunEvent, HandoffEvent, generate_run_id,
)
from paradigms.single_agent import ParadigmResult


# ---------------------------------------------------------------------------
# Agent system prompts — each agent has a distinct personality and focus
# ---------------------------------------------------------------------------

ANALYST_PROMPT = """You are a senior technical analyst. You evaluate software backlog items systematically.

For each item in the backlog, produce a JSON analysis:
{
  "items": [
    {
      "item": "item description",
      "impact_score": 1-10,
      "effort_score": 1-10,
      "priority_score": 1-10,
      "reasoning": "why this score"
    }
  ],
  "recommended_top_3": ["item1", "item2", "item3"],
  "methodology": "brief description of how you scored"
}

Be rigorous. Consider business value, technical debt implications, team capacity, and dependencies.
Output ONLY valid JSON."""

CRITIC_PROMPT = """You are a senior engineering critic. Your job is to challenge and improve a technical analysis.

You will receive an analyst's scoring of backlog items. Your role:
1. Identify assumptions the analyst made that might be wrong
2. Flag any items that seem over- or under-scored
3. Highlight risks or dependencies the analyst missed
4. Suggest score adjustments with reasoning

Output JSON:
{
  "challenges": [
    {
      "item": "item description",
      "original_score": 8,
      "suggested_score": 6,
      "concern": "why the original score is wrong"
    }
  ],
  "missed_risks": ["risk1", "risk2"],
  "overall_assessment": "is the analyst's top 3 correct?",
  "revised_top_3": ["item1", "item2", "item3"]
}

Be constructively adversarial. Your job is to make the final output better by finding weaknesses.
Output ONLY valid JSON."""

SYNTHESISER_PROMPT = """You are a VP of Engineering making the final prioritization call.

You will receive:
1. An analyst's scoring of backlog items
2. A critic's challenges to that scoring

Your job: Synthesise both perspectives into a final, actionable recommendation.

For each of your top 3 priorities, explain:
- WHY it's prioritized (incorporating both analyst and critic views)
- What RISKS exist (from the critic's analysis)
- T-shirt size estimate (S/M/L/XL)
- What needs to be true for this to succeed (dependencies, assumptions)

Also briefly address why remaining items should wait.

Write in clear, direct language a PM can take straight to sprint planning.
Do NOT output JSON — write a readable recommendation document."""


def _call_agent(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_message: str,
    agent_name: str,
    run_id: str,
    obs: ObservabilityLogger,
) -> tuple[str, float, int, int]:
    """
    Make one agent call and log it.

    Returns (output_text, latency_ms, prompt_tokens, completion_tokens).
    Extracted as a helper because all three agents follow the same pattern.
    """
    start = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=4096,
    )

    latency_ms = (time.time() - start) * 1000
    output = response.choices[0].message.content
    prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0

    obs.log_call(RunEvent(
        run_id=run_id,
        paradigm="multi",
        agent_id=agent_name,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        input_text=user_message[:500],
        output_text=output[:500],
    ))

    return output, latency_ms, prompt_tokens, completion_tokens


def run(
    backlog: list[str],
    model: str,
    api_base: str,
    obs: ObservabilityLogger,
) -> ParadigmResult:
    """
    Run the multi-agent pipeline: Analyst → Critic → Synthesiser.

    Each agent's output feeds into the next. We log the handoff at each
    boundary so we can inspect what data flowed between agents.
    """
    run_id = generate_run_id("multi")
    client = OpenAI(base_url=api_base, api_key="not-needed")

    backlog_text = "\n".join(f"{i+1}. {item}" for i, item in enumerate(backlog))
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # ── Step 1: Analyst ──────────────────────────────────────────────────
    analyst_input = f"Here is the backlog to analyse:\n\n{backlog_text}"
    analyst_output, analyst_latency, pt, ct = _call_agent(
        client, model, ANALYST_PROMPT, analyst_input,
        "analyst", run_id, obs,
    )
    total_prompt_tokens += pt
    total_completion_tokens += ct

    # Log handoff: Analyst → Critic
    obs.log_handoff(HandoffEvent(
        run_id=run_id,
        from_agent="analyst",
        to_agent="critic",
        payload_summary=analyst_output[:300],
    ))

    # ── Step 2: Critic ───────────────────────────────────────────────────
    critic_input = f"""Here is the analyst's scoring of the backlog:

{analyst_output}

Original backlog for reference:
{backlog_text}

Please challenge this analysis."""

    critic_output, critic_latency, pt, ct = _call_agent(
        client, model, CRITIC_PROMPT, critic_input,
        "critic", run_id, obs,
    )
    total_prompt_tokens += pt
    total_completion_tokens += ct

    # Log handoff: Critic → Synthesiser
    obs.log_handoff(HandoffEvent(
        run_id=run_id,
        from_agent="critic",
        to_agent="synthesiser",
        payload_summary=critic_output[:300],
    ))

    # ── Step 3: Synthesiser ──────────────────────────────────────────────
    synth_input = f"""Here is the analyst's scoring:

{analyst_output}

Here is the critic's challenges:

{critic_output}

Original backlog:
{backlog_text}

Please produce the final prioritization recommendation."""

    synth_output, synth_latency, pt, ct = _call_agent(
        client, model, SYNTHESISER_PROMPT, synth_input,
        "synthesiser", run_id, obs,
    )
    total_prompt_tokens += pt
    total_completion_tokens += ct

    total_latency = analyst_latency + critic_latency + synth_latency

    return ParadigmResult(
        output=synth_output,
        paradigm="multi",
        run_id=run_id,
        latency_ms=total_latency,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        model=model,
        metadata={
            "analyst_latency_ms": analyst_latency,
            "critic_latency_ms": critic_latency,
            "synthesiser_latency_ms": synth_latency,
            "analyst_output_preview": analyst_output[:200],
            "critic_output_preview": critic_output[:200],
        },
    )
