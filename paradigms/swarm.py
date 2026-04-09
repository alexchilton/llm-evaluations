"""
=============================================================================
PARADIGM 3: SWARM (Stigmergic Communication)
=============================================================================
No coordinator. No message passing between agents. Instead, agents
communicate indirectly through a shared "pheromone board" (SQLite database).

This is inspired by ant colony optimisation (ACO):
  - Ants don't talk to each other
  - They leave pheromone trails on paths they find successful
  - Other ants follow stronger trails → positive feedback loop
  - Trails evaporate over time → bad paths fade

In our version:
  - Each "ant" is an LLM call that reads the board and makes a decision
  - Decisions: propose a NEW hypothesis, REINFORCE an existing one, or
    CONTRADICT one that seems wrong
  - The board's score distribution converges over generations
  - We measure convergence via Shannon entropy

This is the most visually compelling paradigm — watching the entropy
collapse in real time as the swarm reaches consensus is mesmerising.

When to use swarm patterns in production:
  - Problems where you want diverse perspectives without a fixed structure
  - When the optimal decomposition isn't known upfront
  - Brainstorming and ideation tasks
  - Any problem where "wisdom of crowds" might outperform expert opinion
=============================================================================
"""

import time
import json
import uuid
from dataclasses import dataclass, field

from openai import OpenAI
import structlog

from observability.logger import (
    ObservabilityLogger, RunEvent, SwarmEvent, generate_run_id,
)
from observability.swarm_db import SwarmDB, Hypothesis, BoardEvent
from paradigms.single_agent import ParadigmResult

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Swarm agent prompt — each agent independently reads the board and acts
# ---------------------------------------------------------------------------

SWARM_AGENT_PROMPT = """You are one agent in a swarm solving a prioritization problem.

THE PROBLEM:
A software team needs to decide what to prioritize next quarter.

BACKLOG ITEMS:
{backlog}

CURRENT BOARD STATE (hypotheses from other agents, sorted by score):
{board_state}

YOUR TASK:
Choose ONE action:
1. NEW — propose a new prioritization hypothesis not already on the board
2. REINFORCE — agree with an existing hypothesis (specify which by ID)
3. CONTRADICT — disagree with an existing hypothesis (specify which by ID, explain why)

Rules:
- If the board is empty or has few hypotheses, prefer NEW
- If you strongly agree with a top hypothesis, REINFORCE it
- If you think a hypothesis is wrong, CONTRADICT it with reasoning
- Each hypothesis should be a clear recommendation like "Priority 1: X because Y"

Output ONLY valid JSON:
{{"action": "new|reinforce|contradict", "hypothesis_id": "hyp-xxx or null for new", "text": "your hypothesis text (required for new, optional reasoning for reinforce/contradict)"}}"""


def _format_board_state(hypotheses: list[Hypothesis]) -> str:
    """Format the board state for inclusion in the agent prompt."""
    if not hypotheses:
        return "(Board is empty — no hypotheses yet. Please propose one.)"

    lines = []
    for h in hypotheses:
        lines.append(f"  [{h.id}] (score: {h.score:.1f}) {h.text}")
    return "\n".join(lines)


def _parse_agent_response(raw: str) -> dict:
    """
    Parse the agent's JSON response, handling common LLM output quirks.

    LLMs sometimes wrap JSON in markdown code blocks or add commentary
    before/after. Qwen3 models wrap output in <think>...</think> tags.
    We try to extract the JSON object robustly.
    """
    text = raw.strip()

    # Remove Qwen3 <think>...</think> reasoning blocks
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Remove markdown code blocks
    if "```" in text:
        lines = text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                json_lines.append(line)
        if json_lines:
            text = "\n".join(json_lines)

    # Find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    try:
        parsed = json.loads(text)
        # Validate required fields
        action = parsed.get("action", "new").lower()
        if action not in ("new", "reinforce", "contradict"):
            action = "new"
        return {
            "action": action,
            "hypothesis_id": parsed.get("hypothesis_id"),
            "text": parsed.get("text", ""),
        }
    except json.JSONDecodeError:
        # If parsing fails, treat the entire response as a new hypothesis
        return {"action": "new", "hypothesis_id": None, "text": raw[:200]}


def run(
    backlog: list[str],
    model: str,
    api_base: str,
    obs: ObservabilityLogger,
    num_agents: int = 8,
    num_generations: int = 6,
    convergence_threshold: float = 0.8,
    top_k: int = 5,
    db_path: str = "./swarm_state.db",
    on_generation_complete=None,
) -> ParadigmResult:
    """
    Run the swarm paradigm.

    Args:
        backlog:                 List of backlog items
        model:                   Model ID for agent LLM calls
        api_base:                OpenAI-compatible API base URL
        obs:                     Observability logger
        num_agents:              Agents per generation
        num_generations:         Max generations before forced stop
        convergence_threshold:   Stop early if entropy drops below this
        top_k:                   How many hypotheses each agent sees
        db_path:                 Path for the SQLite pheromone board
        on_generation_complete:  Callback(generation, entropy, hypotheses) for live UI

    Returns:
        ParadigmResult with the swarm's consensus recommendation
    """
    run_id = generate_run_id("swarm")
    client = OpenAI(base_url=api_base, api_key="not-needed")
    db = SwarmDB(db_path=db_path)

    backlog_text = "\n".join(f"  - {item}" for item in backlog)
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    entropy_history = []

    try:
        for generation in range(num_generations):
            gen_start = time.time()

            for agent_idx in range(num_agents):
                agent_id = f"swarm-agent-{agent_idx}"

                # 1. Read current board state
                top_hypotheses = db.get_top_hypotheses(k=top_k)
                board_state = _format_board_state(top_hypotheses)

                # 2. Ask the agent to decide
                prompt = SWARM_AGENT_PROMPT.format(
                    backlog=backlog_text,
                    board_state=board_state,
                )

                call_start = time.time()
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a collaborative swarm agent. Output only valid JSON."},
                            {"role": "user", "content": prompt + "\n/no_think"},
                        ],
                        temperature=0.9,  # High temp for diversity
                        max_tokens=2048,
                    )
                    call_latency = (time.time() - call_start) * 1000
                    raw_output = response.choices[0].message.content
                    pt = getattr(response.usage, "prompt_tokens", 0) or 0
                    ct = getattr(response.usage, "completion_tokens", 0) or 0
                except Exception as e:
                    logger.warning("swarm_agent_call_failed", agent=agent_id, error=str(e))
                    continue

                total_latency += call_latency
                total_prompt_tokens += pt
                total_completion_tokens += ct

                # 3. Parse and execute the agent's decision
                decision = _parse_agent_response(raw_output)
                action = decision["action"]
                score_before = 0.0
                score_after = 0.0
                hyp_id = decision.get("hypothesis_id", "")

                if action == "new":
                    hyp = db.add_hypothesis(
                        text=decision["text"] or f"Auto-generated by {agent_id}",
                        agent_id=agent_id,
                        generation=generation,
                    )
                    hyp_id = hyp.id
                    score_after = 1.0

                elif action == "reinforce" and hyp_id:
                    score_before, score_after = db.reinforce_hypothesis(hyp_id)
                    if score_before == 0.0 and score_after == 0.0:
                        # Hypothesis not found — create a new one instead
                        hyp = db.add_hypothesis(
                            text=decision.get("text", f"Fallback from {agent_id}"),
                            agent_id=agent_id,
                            generation=generation,
                        )
                        hyp_id = hyp.id
                        action = "new"
                        score_after = 1.0

                elif action == "contradict" and hyp_id:
                    score_before, score_after = db.contradict_hypothesis(hyp_id)
                    if score_before == 0.0 and score_after == 0.0:
                        # Hypothesis not found — skip
                        continue

                else:
                    # Fallback: treat as new hypothesis
                    hyp = db.add_hypothesis(
                        text=decision.get("text", f"Fallback from {agent_id}"),
                        agent_id=agent_id,
                        generation=generation,
                    )
                    hyp_id = hyp.id
                    action = "new"
                    score_after = 1.0

                # 4. Compute board entropy after this action
                entropy = db.compute_entropy()

                # 5. Log the event
                board_event = BoardEvent(
                    generation=generation,
                    agent_id=agent_id,
                    action=action,
                    hypothesis_id=hyp_id,
                    score_before=score_before,
                    score_after=score_after,
                    board_entropy=entropy,
                )
                db.log_event(board_event)

                obs.log_swarm_event(SwarmEvent(
                    run_id=run_id,
                    generation=generation,
                    agent_id=agent_id,
                    action=action,
                    hypothesis_id=hyp_id,
                    score_before=score_before,
                    score_after=score_after,
                    board_entropy=entropy,
                ))

                obs.log_call(RunEvent(
                    run_id=run_id,
                    paradigm="swarm",
                    agent_id=agent_id,
                    model=model,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    latency_ms=call_latency,
                    input_text=f"gen={generation} action={action}",
                    output_text=raw_output[:200],
                    metadata={"generation": generation, "action": action},
                ))

            # End of generation — compute final entropy
            gen_entropy = db.compute_entropy()
            entropy_history.append((generation, gen_entropy))

            # Callback for live UI updates
            if on_generation_complete:
                on_generation_complete(
                    generation, gen_entropy, db.get_top_hypotheses(k=10),
                )

            logger.info(
                "swarm_generation_complete",
                generation=generation,
                entropy=f"{gen_entropy:.3f}",
                num_hypotheses=len(db.get_all_hypotheses()),
            )

            # Check convergence
            if gen_entropy < convergence_threshold and generation > 0:
                logger.info("swarm_converged", generation=generation, entropy=gen_entropy)
                break

        # ── Build final output from top hypotheses ────────────────────────
        final_hypotheses = db.get_top_hypotheses(k=5)
        output_lines = ["## Swarm Consensus (Stigmergic Prioritization)\n"]
        for i, h in enumerate(final_hypotheses, 1):
            output_lines.append(f"**#{i}** (score: {h.score:.1f}) {h.text}\n")

        output_lines.append(f"\n**Convergence:** {len(entropy_history)} generations")
        output_lines.append(f"**Final entropy:** {entropy_history[-1][1]:.3f}" if entropy_history else "")
        output = "\n".join(output_lines)

        return ParadigmResult(
            output=output,
            paradigm="swarm",
            run_id=run_id,
            latency_ms=total_latency,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            model=model,
            metadata={
                "entropy_history": entropy_history,
                "num_hypotheses": len(db.get_all_hypotheses()),
                "final_entropy": entropy_history[-1][1] if entropy_history else 0.0,
                "convergence_generation": len(entropy_history) - 1,
                "top_hypotheses": [
                    {"id": h.id, "text": h.text[:100], "score": h.score}
                    for h in final_hypotheses
                ],
            },
        )

    finally:
        db.close()
