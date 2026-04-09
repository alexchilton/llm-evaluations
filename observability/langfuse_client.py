"""
=============================================================================
LANGFUSE CLIENT WRAPPER
=============================================================================
Thin wrapper around the Langfuse Python SDK. Provides trace/span/score
helpers that the ObservabilityLogger calls into.

Key design decision: GRACEFUL DEGRADATION
  If Langfuse isn't running (no Docker, wrong keys, etc.), this module
  returns a NullClient that silently no-ops every call. The demo never
  crashes because of missing observability infrastructure.

Why wrap at all instead of using Langfuse directly?
  1. Centralise connection setup and error handling
  2. Provide a clean interface that matches our RunEvent/SwarmEvent schema
  3. Make it trivial to swap to another observability backend later
=============================================================================
"""

import os
from typing import Optional
from dataclasses import asdict

import structlog

logger = structlog.get_logger()


class NullLangfuseClient:
    """
    Drop-in replacement when Langfuse is unavailable.
    Every method is a no-op — the demo keeps running, just without
    remote trace collection.
    """
    def log_generation(self, event): pass
    def log_architecture(self, run_id, spec): pass
    def log_scores(self, run_id, scores): pass
    def create_trace(self, **kwargs): return NullTrace()
    def flush(self): pass


class NullTrace:
    """No-op trace object returned by NullLangfuseClient."""
    def span(self, **kwargs): return NullSpan()
    def score(self, **kwargs): pass
    def update(self, **kwargs): pass

    @property
    def id(self):
        return "null-trace"


class NullSpan:
    """No-op span object."""
    def end(self, **kwargs): pass
    def update(self, **kwargs): pass
    def score(self, **kwargs): pass

    @property
    def id(self):
        return "null-span"


class LangfuseClient:
    """
    Real Langfuse client wrapper.

    Reads connection details from environment variables (loaded from .env
    by the main entrypoint). If connection fails at init time, degrades
    to NullLangfuseClient automatically.
    """

    def __init__(self, host: Optional[str] = None):
        self._client = None
        self._available = False

        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
                host=host or os.getenv("LANGFUSE_HOST", "http://localhost:3001"),
            )
            # Quick health check — if keys are placeholder values, skip
            pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
            if pk and "REPLACE" not in pk and "..." not in pk:
                self._available = True
                logger.info("langfuse_connected", host=host or os.getenv("LANGFUSE_HOST"))
            else:
                logger.info("langfuse_skipped", reason="placeholder keys detected")

        except Exception as e:
            logger.warning("langfuse_init_failed", error=str(e))

    @property
    def available(self) -> bool:
        return self._available

    def create_trace(self, **kwargs):
        """Create a Langfuse trace. Returns NullTrace if unavailable."""
        if not self._available:
            return NullTrace()
        try:
            return self._client.trace(**kwargs)
        except Exception as e:
            logger.warning("langfuse_trace_failed", error=str(e))
            return NullTrace()

    def log_generation(self, event):
        """Log an LLM generation event as a Langfuse generation."""
        if not self._available:
            return
        try:
            d = asdict(event) if hasattr(event, '__dataclass_fields__') else event
            self._client.generation(
                name=f"{d.get('paradigm', 'unknown')}:{d.get('agent_id', 'main')}",
                model=d.get("model", ""),
                input=d.get("input_text", ""),
                output=d.get("output_text", ""),
                usage={
                    "prompt_tokens": d.get("prompt_tokens", 0),
                    "completion_tokens": d.get("completion_tokens", 0),
                },
                metadata=d.get("metadata", {}),
            )
        except Exception as e:
            logger.warning("langfuse_generation_failed", error=str(e))

    def log_architecture(self, run_id: str, spec: dict):
        """Log an ADAS architecture spec as a Langfuse event."""
        if not self._available:
            return
        try:
            self._client.event(
                name="adas_architecture",
                metadata={"run_id": run_id, "spec": spec},
            )
        except Exception as e:
            logger.warning("langfuse_architecture_failed", error=str(e))

    def log_scores(self, run_id: str, scores: dict):
        """Log eval scores to Langfuse."""
        if not self._available:
            return
        try:
            for name, value in scores.items():
                self._client.score(
                    name=name,
                    value=float(value),
                    trace_id=run_id,
                )
        except Exception as e:
            logger.warning("langfuse_score_failed", error=str(e))

    def flush(self):
        """Flush any buffered events to Langfuse."""
        if self._available and self._client:
            try:
                self._client.flush()
            except Exception:
                pass


def get_langfuse_client(host: Optional[str] = None) -> LangfuseClient:
    """
    Factory function. Returns a real or null client depending on availability.
    Call this once at startup and pass the result to ObservabilityLogger.
    """
    client = LangfuseClient(host=host)
    if not client.available:
        logger.info("langfuse_fallback", message="Using JSON file logging only")
    return client
