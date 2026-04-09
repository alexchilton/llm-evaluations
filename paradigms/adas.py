"""
=============================================================================
PARADIGM 4: ADAS — Automated Design of Agentic Systems
=============================================================================
The most meta paradigm: a "meta-agent" reads the problem and DESIGNS the
agent system before running it. The architecture is dynamic — different
problems produce different agent configurations.

Two phases:
  Phase 1: Meta-agent reads the problem → outputs architecture spec (JSON)
  Phase 2: Runtime parses the spec → instantiates agents → runs the pipeline

This is powerful because:
  - The system adapts to the problem (not a fixed pipeline)
  - You can inspect the meta-agent's reasoning about WHY it chose each agent
  - It surfaces the implicit design decisions that humans usually make manually
  - Novel problems get novel architectures

The "wow moment" is printing the designed architecture before running it —
the audience sees the AI deciding what kind of AI system to build.

When to use ADAS in production:
  - Problems where the optimal decomposition isn't obvious
  - When you want to A/B test different architectures automatically
  - Meta-learning scenarios: "what kind of system works best for this class of problems?"

Risks:
  - Meta-agent errors cascade (bad architecture → bad results)
  - Higher latency (two phases instead of one)
  - Harder to debug (dynamic structure means dynamic failure modes)
=============================================================================
"""

import time
import json
from dataclasses import dataclass, field

from openai import OpenAI
import structlog

from observability.logger import ObservabilityLogger, RunEvent, generate_run_id
from paradigms.single_agent import ParadigmResult

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Meta-agent prompt — designs the agent system
# ---------------------------------------------------------------------------

META_PROMPT = """You are an agentic systems architect with deep expertise in multi-agent system design.

Given this problem, design the optimal agent system to solve it.

PROBLEM:
A software team needs to prioritize their next-quarter backlog. The backlog items are:
{backlog}

CONSTRAINTS:
- Maximum {max_agents} agents
- Each agent makes one LLM call
- Agents can be arranged in: pipeline (sequential), parallel (independent then merge), or hybrid
- Each agent needs a focused role and clear system prompt

Design the system. Output ONLY valid JSON:
{{
  "reasoning": "2-3 sentences explaining why you chose this architecture",
  "topology": "pipeline|parallel|hybrid",
  "agents": [
    {{
      "name": "descriptive_agent_name",
      "role": "one sentence role description",
      "system_prompt": "full system prompt for this agent — be specific and detailed",
      "receives_from": ["agent_name"] or [],
      "sends_to": ["agent_name"] or []
    }}
  ]
}}

Think carefully about:
- What perspectives are needed to prioritize well?
- What ordering maximises information flow?
- Where does adversarial thinking add value?
- What's the minimum number of agents to cover the problem space?"""


def _parse_architecture(raw: str) -> dict:
    """
    Parse the meta-agent's architecture spec from its response.
    Handles markdown code blocks, Qwen3 <think> tags, and other LLM quirks.
    """
    import re
    text = raw.strip()

    # Remove Qwen3 <think>...</think> reasoning blocks
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

    # Find JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    try:
        spec = json.loads(text)
        # Validate structure
        if "agents" not in spec or not isinstance(spec["agents"], list):
            raise ValueError("Missing or invalid 'agents' field")
        return spec
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("adas_architecture_parse_failed", error=str(e))
        # Return a sensible fallback architecture
        return {
            "reasoning": "Fallback architecture due to parse error",
            "topology": "pipeline",
            "agents": [
                {
                    "name": "analyst",
                    "role": "Analyse and prioritize the backlog",
                    "system_prompt": "You are a senior engineering manager. Analyse the backlog and recommend top 3 priorities with reasoning.",
                    "receives_from": [],
                    "sends_to": ["recommender"],
                },
                {
                    "name": "recommender",
                    "role": "Produce final recommendation",
                    "system_prompt": "You are a VP of Engineering. Given the analysis, produce a final actionable recommendation for the team.",
                    "receives_from": ["analyst"],
                    "sends_to": [],
                },
            ],
        }


def _execute_architecture(
    spec: dict,
    backlog: list[str],
    model: str,
    api_base: str,
    run_id: str,
    obs: ObservabilityLogger,
) -> tuple[str, float, int, int]:
    """
    Execute the meta-agent-designed architecture.

    This is the runtime that takes a JSON spec and turns it into actual
    LLM calls wired together according to the topology.

    For simplicity, we handle three topologies:
      - pipeline: agents run sequentially, each getting previous output
      - parallel: agents run independently, outputs merged at the end
      - hybrid: first batch runs in parallel, final agent synthesises

    Returns (final_output, total_latency_ms, total_prompt_tokens, total_completion_tokens)
    """
    client = OpenAI(base_url=api_base, api_key="not-needed")
    backlog_text = "\n".join(f"  - {item}" for item in backlog)

    agents = spec.get("agents", [])
    topology = spec.get("topology", "pipeline")

    total_latency = 0.0
    total_pt = 0
    total_ct = 0
    agent_outputs = {}  # name → output

    if topology == "pipeline":
        # Sequential: each agent gets the previous agent's output
        previous_output = ""
        for agent_spec in agents:
            name = agent_spec["name"]
            sys_prompt = agent_spec.get("system_prompt", f"You are {name}.")

            # Build input: backlog + any previous agent output
            if previous_output:
                user_msg = f"Backlog:\n{backlog_text}\n\nPrevious agent output:\n{previous_output}\n\nPlease do your analysis."
            else:
                user_msg = f"Backlog:\n{backlog_text}\n\nPlease do your analysis."

            start = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                max_tokens=4096,
            )
            latency = (time.time() - start) * 1000
            output = response.choices[0].message.content
            pt = getattr(response.usage, "prompt_tokens", 0) or 0
            ct = getattr(response.usage, "completion_tokens", 0) or 0

            total_latency += latency
            total_pt += pt
            total_ct += ct

            agent_outputs[name] = output
            previous_output = output

            obs.log_call(RunEvent(
                run_id=run_id,
                paradigm="adas",
                agent_id=name,
                model=model,
                prompt_tokens=pt,
                completion_tokens=ct,
                latency_ms=latency,
                input_text=user_msg[:300],
                output_text=output[:300],
                metadata={"meta_designed": True, "topology": topology},
            ))

        final_output = previous_output

    elif topology == "parallel":
        # All agents run independently, then we merge outputs
        outputs = []
        for agent_spec in agents:
            name = agent_spec["name"]
            sys_prompt = agent_spec.get("system_prompt", f"You are {name}.")
            user_msg = f"Backlog:\n{backlog_text}\n\nPlease do your analysis."

            start = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                max_tokens=4096,
            )
            latency = (time.time() - start) * 1000
            output = response.choices[0].message.content
            pt = getattr(response.usage, "prompt_tokens", 0) or 0
            ct = getattr(response.usage, "completion_tokens", 0) or 0

            total_latency += latency
            total_pt += pt
            total_ct += ct
            outputs.append(f"=== {name} ===\n{output}")

            agent_outputs[name] = output
            obs.log_call(RunEvent(
                run_id=run_id,
                paradigm="adas",
                agent_id=name,
                model=model,
                prompt_tokens=pt,
                completion_tokens=ct,
                latency_ms=latency,
                input_text=user_msg[:300],
                output_text=output[:300],
                metadata={"meta_designed": True, "topology": topology},
            ))

        # Merge with a final synthesis call
        merge_prompt = "You are a decision-maker. Synthesise these independent analyses into one final recommendation."
        merge_input = f"Backlog:\n{backlog_text}\n\nAgent outputs:\n\n" + "\n\n".join(outputs)

        start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": merge_prompt},
                {"role": "user", "content": merge_input},
            ],
            temperature=0.5,
            max_tokens=4096,
        )
        latency = (time.time() - start) * 1000
        final_output = response.choices[0].message.content
        pt = getattr(response.usage, "prompt_tokens", 0) or 0
        ct = getattr(response.usage, "completion_tokens", 0) or 0
        total_latency += latency
        total_pt += pt
        total_ct += ct

        obs.log_call(RunEvent(
            run_id=run_id, paradigm="adas", agent_id="merger",
            model=model, prompt_tokens=pt, completion_tokens=ct,
            latency_ms=latency,
            input_text="merge of parallel outputs",
            output_text=final_output[:300],
            metadata={"meta_designed": True, "topology": topology},
        ))

    else:
        # Hybrid: split agents into those with no receives_from (parallel batch)
        # and those that receive from others (sequential after)
        parallel_agents = [a for a in agents if not a.get("receives_from")]
        sequential_agents = [a for a in agents if a.get("receives_from")]

        # Run parallel batch
        for agent_spec in parallel_agents:
            name = agent_spec["name"]
            sys_prompt = agent_spec.get("system_prompt", f"You are {name}.")
            user_msg = f"Backlog:\n{backlog_text}\n\nPlease do your analysis."

            start = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                max_tokens=4096,
            )
            latency = (time.time() - start) * 1000
            output = response.choices[0].message.content
            pt = getattr(response.usage, "prompt_tokens", 0) or 0
            ct = getattr(response.usage, "completion_tokens", 0) or 0

            total_latency += latency
            total_pt += pt
            total_ct += ct
            agent_outputs[name] = output

            obs.log_call(RunEvent(
                run_id=run_id, paradigm="adas", agent_id=name,
                model=model, prompt_tokens=pt, completion_tokens=ct,
                latency_ms=latency,
                input_text=user_msg[:300], output_text=output[:300],
                metadata={"meta_designed": True, "topology": "hybrid-parallel"},
            ))

        # Run sequential agents with their inputs
        for agent_spec in sequential_agents:
            name = agent_spec["name"]
            sys_prompt = agent_spec.get("system_prompt", f"You are {name}.")
            inputs_from = agent_spec.get("receives_from", [])
            prev_outputs = "\n\n".join(
                f"=== {src} ===\n{agent_outputs.get(src, '(no output)')}"
                for src in inputs_from
            )
            user_msg = f"Backlog:\n{backlog_text}\n\nInputs from other agents:\n{prev_outputs}\n\nPlease produce your analysis."

            start = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.5,
                max_tokens=4096,
            )
            latency = (time.time() - start) * 1000
            output = response.choices[0].message.content
            pt = getattr(response.usage, "prompt_tokens", 0) or 0
            ct = getattr(response.usage, "completion_tokens", 0) or 0

            total_latency += latency
            total_pt += pt
            total_ct += ct
            agent_outputs[name] = output

            obs.log_call(RunEvent(
                run_id=run_id, paradigm="adas", agent_id=name,
                model=model, prompt_tokens=pt, completion_tokens=ct,
                latency_ms=latency,
                input_text=user_msg[:300], output_text=output[:300],
                metadata={"meta_designed": True, "topology": "hybrid-sequential"},
            ))

        # The last sequential agent's output is the final output
        final_output = output if sequential_agents else "\n\n".join(agent_outputs.values())

    return final_output, total_latency, total_pt, total_ct


def run(
    backlog: list[str],
    model: str,
    api_base: str,
    obs: ObservabilityLogger,
    max_agents: int = 5,
    on_architecture_designed=None,
) -> ParadigmResult:
    """
    Run the ADAS paradigm: meta-agent designs, then runtime executes.

    Args:
        backlog:                  List of backlog items
        model:                    Model ID for LLM calls
        api_base:                 OpenAI-compatible API base URL
        obs:                      Observability logger
        max_agents:               Max agents the meta-agent can design
        on_architecture_designed: Callback(spec_dict) for UI display

    Returns:
        ParadigmResult with the final recommendation
    """
    run_id = generate_run_id("adas")
    client = OpenAI(base_url=api_base, api_key="not-needed")

    backlog_text = "\n".join(f"  - {item}" for item in backlog)

    # ── Phase 1: Meta-agent designs the architecture ─────────────────────
    meta_prompt = META_PROMPT.format(backlog=backlog_text, max_agents=max_agents)

    meta_start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in multi-agent system design. Output only valid JSON."},
            {"role": "user", "content": meta_prompt + "\n/no_think"},
        ],
        temperature=0.7,
        max_tokens=4096,
    )
    meta_latency = (time.time() - meta_start) * 1000
    meta_output = response.choices[0].message.content
    meta_pt = getattr(response.usage, "prompt_tokens", 0) or 0
    meta_ct = getattr(response.usage, "completion_tokens", 0) or 0

    obs.log_call(RunEvent(
        run_id=run_id,
        paradigm="adas",
        agent_id="meta_architect",
        model=model,
        prompt_tokens=meta_pt,
        completion_tokens=meta_ct,
        latency_ms=meta_latency,
        input_text=meta_prompt[:300],
        output_text=meta_output[:300],
    ))

    # Parse the architecture spec
    architecture = _parse_architecture(meta_output)
    obs.log_architecture(run_id, architecture)

    # Notify the UI about the designed architecture
    if on_architecture_designed:
        on_architecture_designed(architecture)

    # ── Phase 2: Execute the designed architecture ────────────────────────
    exec_output, exec_latency, exec_pt, exec_ct = _execute_architecture(
        architecture, backlog, model, api_base, run_id, obs,
    )

    total_latency = meta_latency + exec_latency
    total_pt = meta_pt + exec_pt
    total_ct = meta_ct + exec_ct

    return ParadigmResult(
        output=exec_output,
        paradigm="adas",
        run_id=run_id,
        latency_ms=total_latency,
        prompt_tokens=total_pt,
        completion_tokens=total_ct,
        model=model,
        metadata={
            "architecture": architecture,
            "meta_latency_ms": meta_latency,
            "exec_latency_ms": exec_latency,
            "topology": architecture.get("topology", "unknown"),
            "num_agents_designed": len(architecture.get("agents", [])),
            "meta_reasoning": architecture.get("reasoning", ""),
        },
    )
