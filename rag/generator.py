"""
Generator
=========
LLM generation inside an OpenTelemetry span. Returns answer + trace context.
Strips Qwen's <think> tags from output.
"""

from __future__ import annotations

import re
import time

from openai import OpenAI
from opentelemetry import trace

from rag.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    LLM_API_KEY,
    LLM_BASE_URL,
    MODEL_NAME,
)

tracer = trace.get_tracer("rag.generator")


def _strip_think_tags(text: str) -> str:
    """Remove Qwen's <think>...</think> reasoning blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class Generator:
    """LLM generation with OpenTelemetry instrumentation."""

    def __init__(
        self,
        llm_base_url: str = LLM_BASE_URL,
        model_name: str = MODEL_NAME,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.llm_base_url = llm_base_url
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(base_url=self.llm_base_url, api_key=LLM_API_KEY)
        return self._client

    def generate(self, query: str, context: str) -> tuple[str, float]:
        """
        Send query + context to local Qwen inside an OTel span.
        Returns (answer, latency_ms).
        """
        with tracer.start_as_current_span("llm_generate") as span:
            span.set_attribute("llm.model", self.model_name)
            span.set_attribute("llm.temperature", self.temperature)
            span.set_attribute("llm.max_tokens", self.max_tokens)

            user_msg = (
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer concisely based on the context. /no_think"
            )

            start = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            latency_ms = (time.time() - start) * 1000

            answer = response.choices[0].message.content or ""
            answer = _strip_think_tags(answer)

            # Fallback: check reasoning_content if content is empty
            if not answer:
                extra = getattr(response.choices[0].message, "model_extra", {}) or {}
                reasoning = extra.get("reasoning_content", "")
                if reasoning:
                    answer = reasoning.split("</think>")[-1].strip() if "</think>" in reasoning else reasoning[-500:]

            usage = getattr(response, "usage", None)
            if usage:
                span.set_attribute("llm.tokens.prompt", usage.prompt_tokens)
                span.set_attribute("llm.tokens.completion", usage.completion_tokens)
                span.set_attribute("llm.tokens.total", usage.total_tokens)

            span.set_attribute("llm.latency_ms", latency_ms)
            span.set_attribute("llm.answer_length", len(answer))

        return answer, latency_ms
