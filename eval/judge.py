"""
=============================================================================
LLM-AS-JUDGE SCORER
=============================================================================
After each paradigm completes, we score the output using the same LLM as an
impartial judge. This gives us comparable quality metrics across paradigms.

Scoring dimensions:
  - clarity:      Is the recommendation clear and actionable?
  - reasoning:    Is the reasoning sound and well-supported?
  - completeness: Does it address the full backlog?
  - overall:      Weighted combination

Why LLM-as-judge instead of human eval?
  For a demo, we need automated scoring to show the comparison table
  immediately. LLM-as-judge is the standard approach (see: LMSYS Chatbot
  Arena, AlpacaEval). It correlates well with human preferences when the
  judge model is capable.

We use the same model for judging — in production you'd want a separate,
stronger model, but for a self-contained demo this works.
=============================================================================
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI
import structlog

logger = structlog.get_logger()


@dataclass
class EvalResult:
    """Structured eval scores from the judge."""
    clarity: float = 0.0
    reasoning: float = 0.0
    completeness: float = 0.0
    overall: float = 0.0
    raw_response: str = ""
    latency_ms: float = 0.0


# The judge prompt — carefully designed to produce consistent JSON scores
JUDGE_PROMPT = """You are evaluating the quality of a software team prioritization recommendation.

A team was given this backlog of items to prioritize:
{backlog}

Here is the recommendation that was produced:
{output}

Score the recommendation from 0.0 to 1.0 on each dimension:
- clarity: Is the recommendation clear, specific, and actionable? Would a PM know exactly what to do?
- reasoning: Is the reasoning sound? Are trade-offs acknowledged? Is the logic coherent?
- completeness: Does it address the full backlog? Are all items accounted for, even deprioritized ones?

Then compute an overall score as: (clarity * 0.3) + (reasoning * 0.4) + (completeness * 0.3)

Output ONLY valid JSON, no other text:
{{"clarity": 0.0, "reasoning": 0.0, "completeness": 0.0, "overall": 0.0}}"""


def score(
    backlog: list[str],
    output: str,
    model: str,
    api_base: str,
) -> EvalResult:
    """
    Score a paradigm's output using LLM-as-judge.

    Args:
        backlog:  The original backlog items
        output:   The paradigm's recommendation text
        model:    Model ID for the judge LLM
        api_base: OpenAI-compatible API base URL

    Returns:
        EvalResult with scores on each dimension

    The judge call uses low temperature (0.1) for consistency — we want
    the same output to get roughly the same score across runs.
    """
    client = OpenAI(base_url=api_base, api_key="not-needed")

    backlog_text = "\n".join(f"  - {item}" for item in backlog)
    prompt = JUDGE_PROMPT.format(backlog=backlog_text, output=output)

    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a fair and precise evaluator. Output only valid JSON."},
                {"role": "user", "content": prompt + "\n/no_think"},
            ],
            temperature=0.1,
            max_tokens=8192,
        )
        latency_ms = (time.time() - start) * 1000
        msg = response.choices[0].message
        raw = (msg.content or "").strip()
        
        # Qwen3 thinking models put output in reasoning_content when content is empty
        if not raw and hasattr(msg, 'model_extra') and msg.model_extra:
            rc = msg.model_extra.get('reasoning_content', '')
            if rc:
                logger.info("judge_using_reasoning_content")
                raw = rc

        # Remove Qwen3 <think>...</think> reasoning blocks
        import re
        json_str = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Parse the JSON response — handle markdown code blocks
        if "```" in json_str:
            # Extract JSON from code block
            lines = json_str.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block or (not json_lines and line.strip().startswith("{")):
                    json_lines.append(line)
            json_str = "\n".join(json_lines) if json_lines else json_str

        # Find JSON object in response
        start_idx = json_str.find("{")
        end_idx = json_str.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = json_str[start_idx:end_idx]

        scores = json.loads(json_str)

        return EvalResult(
            clarity=float(scores.get("clarity", 0.0)),
            reasoning=float(scores.get("reasoning", 0.0)),
            completeness=float(scores.get("completeness", 0.0)),
            overall=float(scores.get("overall", 0.0)),
            raw_response=raw,
            latency_ms=latency_ms,
        )

    except json.JSONDecodeError as e:
        logger.warning("judge_json_parse_failed", error=str(e), raw=raw[:200])
        return EvalResult(raw_response=raw if 'raw' in dir() else str(e), latency_ms=(time.time() - start) * 1000)

    except Exception as e:
        logger.error("judge_call_failed", error=str(e))
        return EvalResult(latency_ms=(time.time() - start) * 1000)
