"""
=============================================================================
UNIFIED OBSERVABILITY LOGGER
=============================================================================
Central logging hub for all four paradigms. Every LLM call, agent handoff,
swarm event, and architecture decision flows through here.

Design choices:
- Uses structlog for structured JSON logging (machine-parseable + human-readable)
- RunEvent dataclass captures the common denominator across paradigms
- Paradigm-specific events (handoffs, swarm, ADAS) extend with metadata dicts
- Thread-safe: each call creates its own event — no shared mutable state
- Falls back to JSON file logging if Langfuse is unavailable

Why structlog over stdlib logging?
  structlog gives us key=value structured output that's trivially parseable
  by log aggregators (ELK, Loki, etc.) while staying readable in a terminal.
  It also chains processors, so we can add context (run_id, paradigm) once
  and have it appear on every subsequent log line.
=============================================================================
"""

import time
import uuid
import json
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional

import structlog

# ---------------------------------------------------------------------------
# Configure structlog once at import time
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Data classes — the common event schema
# ---------------------------------------------------------------------------

@dataclass
class RunEvent:
    """
    Captures one LLM call or agent step.

    Every paradigm emits these. The `metadata` dict carries paradigm-specific
    fields (e.g. swarm generation number, ADAS architecture spec hash).
    """
    run_id: str
    paradigm: str               # single | multi | swarm | adas
    agent_id: str = ""          # which agent within the paradigm
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    input_text: str = ""
    output_text: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class HandoffEvent:
    """Captures data passing between agents in the multi-agent pipeline."""
    run_id: str
    from_agent: str
    to_agent: str
    payload_summary: str        # truncated preview of the handoff data
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class SwarmEvent:
    """Captures one swarm agent's action in one generation."""
    run_id: str
    generation: int
    agent_id: str
    action: str                 # 'new' | 'reinforce' | 'contradict'
    hypothesis_id: str
    score_before: float
    score_after: float
    board_entropy: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# ObservabilityLogger — the single interface all paradigms use
# ---------------------------------------------------------------------------

class ObservabilityLogger:
    """
    Unified logger that wraps structlog + optional Langfuse + optional JSON file.

    Usage:
        obs = ObservabilityLogger(langfuse_client=lf_client)
        obs.log_call(RunEvent(...))
        obs.log_handoff(HandoffEvent(...))
        obs.score("run-123", 0.85, "Good reasoning")

    Why a class rather than bare functions?
      We need to hold references to the Langfuse client and the JSON fallback
      file path. A class keeps this state tidy and testable.
    """

    def __init__(self, langfuse_client=None, json_log_path: str = "./run_log.jsonl"):
        self.langfuse = langfuse_client
        self.json_log_path = Path(json_log_path)
        self._log = structlog.get_logger()

    def _write_jsonl(self, event_dict: dict):
        """Append one JSON line to the fallback log file."""
        with open(self.json_log_path, "a") as f:
            f.write(json.dumps(event_dict, default=str) + "\n")

    def log_call(self, event: RunEvent):
        """
        Log an LLM call. This is the most common event — every paradigm
        emits one per LLM invocation.
        """
        d = asdict(event)
        self._log.info("llm_call", **d)
        self._write_jsonl({"type": "llm_call", **d})

        # If Langfuse is available, create a generation span
        if self.langfuse:
            try:
                self.langfuse.log_generation(event)
            except Exception as e:
                self._log.warning("langfuse_log_failed", error=str(e))

    def log_handoff(self, event: HandoffEvent):
        """Log agent-to-agent data handoff (multi-agent pipeline)."""
        d = asdict(event)
        self._log.info("handoff", **d)
        self._write_jsonl({"type": "handoff", **d})

    def log_swarm_event(self, event: SwarmEvent):
        """Log one swarm agent action within a generation."""
        d = asdict(event)
        self._log.info("swarm_event", **d)
        self._write_jsonl({"type": "swarm_event", **d})

    def log_architecture(self, run_id: str, spec: dict):
        """Log the ADAS-designed architecture spec as a JSON artifact."""
        self._log.info("adas_architecture", run_id=run_id, spec=spec)
        self._write_jsonl({"type": "adas_architecture", "run_id": run_id, "spec": spec})

        if self.langfuse:
            try:
                self.langfuse.log_architecture(run_id, spec)
            except Exception as e:
                self._log.warning("langfuse_architecture_log_failed", error=str(e))

    def score(self, run_id: str, scores: dict, reason: str = ""):
        """
        Record eval scores for a paradigm run.
        `scores` is a dict like {"clarity": 0.8, "reasoning": 0.9, ...}
        """
        self._log.info("eval_score", run_id=run_id, scores=scores, reason=reason)
        self._write_jsonl({
            "type": "eval_score",
            "run_id": run_id,
            "scores": scores,
            "reason": reason,
        })

        if self.langfuse:
            try:
                self.langfuse.log_scores(run_id, scores)
            except Exception as e:
                self._log.warning("langfuse_score_failed", error=str(e))


def generate_run_id(paradigm: str) -> str:
    """Generate a unique run ID with paradigm prefix for easy filtering."""
    short_uuid = uuid.uuid4().hex[:8]
    return f"{paradigm}-{short_uuid}"
