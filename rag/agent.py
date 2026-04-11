"""
RAG Agent
=========
ReAct-style tool-using agent wrapping RAGPipeline as one of several tools.
Useful for questions requiring calculation, date logic, or multi-hop retrieval.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from openai import OpenAI
from opentelemetry import trace

from rag.config import LLM_API_KEY, LLM_BASE_URL, MODEL_NAME
from rag.pipeline import RAGPipeline, RAGResult

tracer = trace.get_tracer("rag.agent")


@dataclass
class AgentStep:
    """One step in the agent's trajectory."""
    step: int
    thought: str
    tool: Optional[str] = None
    tool_input: Optional[str] = None
    tool_output: Optional[str] = None
    is_final: bool = False


@dataclass
class AgentResult:
    """Complete output from an agent run."""
    question: str
    final_answer: str
    trajectory: list[AgentStep]
    trace_id: str
    latency_ms: float
    rag_result: Optional[RAGResult] = None  # set if rag_tool was used


def _safe_calc(expr: str) -> str:
    """Evaluate a math expression safely."""
    try:
        expr = re.sub(
            r'(\d+(?:\.\d+)?)%\s*of\s*(\d+(?:\.\d+)?)',
            lambda m: f"({m.group(1)}/100)*{m.group(2)}",
            expr,
        )
        expr = re.split(r'[^0-9+\-*/().% ]', expr)[0].strip()
        return str(eval(expr))  # noqa: S307
    except Exception:
        return f"Cannot evaluate: {expr!r}"


def _date_tool(query: str) -> str:
    """Answer basic date/time questions."""
    now = datetime.now()
    q = query.lower()
    if "today" in q or "date" in q:
        return now.strftime("%Y-%m-%d")
    if "year" in q:
        return str(now.year)
    if "time" in q:
        return now.strftime("%H:%M:%S")
    if "day" in q:
        return now.strftime("%A")
    return now.isoformat()


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


REACT_SYSTEM = """You are a helpful agent with access to tools.

Available tools:
- rag_query("question"): Search a knowledge base and get an answer
- calculator("expression"): Evaluate a math expression
- date("query"): Get current date/time info

To use a tool, reply EXACTLY: ACTION: tool_name("argument")
To give your final answer, reply: FINAL: your answer

Think step by step. Use tools when needed."""


class RAGAgent:
    """ReAct agent with rag_query, calculator, and date tools."""

    def __init__(
        self,
        pipeline: RAGPipeline,
        llm_base_url: str = LLM_BASE_URL,
        model_name: str = MODEL_NAME,
        max_steps: int = 5,
    ):
        self.pipeline = pipeline
        self.model_name = model_name
        self.max_steps = max_steps
        self._client = None
        self._llm_base_url = llm_base_url

        self.tools = {
            "rag_query": self._rag_tool,
            "calculator": _safe_calc,
            "date": _date_tool,
        }

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(base_url=self._llm_base_url, api_key=LLM_API_KEY)
        return self._client

    def _rag_tool(self, question: str) -> str:
        """Invoke the RAG pipeline and return the answer."""
        result = self.pipeline.query(question)
        self._last_rag_result = result
        return result.answer

    def run(self, question: str) -> AgentResult:
        """Execute the ReAct loop. Returns AgentResult with trajectory and trace_id."""
        start = time.time()
        trajectory: list[AgentStep] = []
        self._last_rag_result = None
        context_log = ""

        with tracer.start_as_current_span("agent_run") as root_span:
            root_span.set_attribute("agent.question", question[:200])
            root_span.set_attribute("agent.max_steps", self.max_steps)

            for step_num in range(self.max_steps):
                with tracer.start_as_current_span(f"agent_step_{step_num}") as step_span:
                    step_span.set_attribute("agent.step", step_num)

                    user_content = (
                        f"Question: {question}\n"
                        f"{context_log}\n"
                        "What's your next step? /no_think"
                    )

                    resp = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": REACT_SYSTEM},
                            {"role": "user", "content": user_content},
                        ],
                        max_tokens=512,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                    response_text = _strip_think(resp.choices[0].message.content or "")

                    # Check for final answer
                    if response_text.strip().upper().startswith("FINAL:"):
                        final_answer = response_text.split(":", 1)[1].strip()
                        step = AgentStep(
                            step=step_num,
                            thought=response_text,
                            is_final=True,
                        )
                        trajectory.append(step)
                        step_span.set_attribute("agent.action", "final_answer")

                        total_latency = (time.time() - start) * 1000
                        root_span.set_attribute("agent.steps_taken", step_num + 1)
                        root_span.set_attribute("agent.latency_ms", total_latency)
                        trace_id = format(root_span.get_span_context().trace_id, "032x")

                        return AgentResult(
                            question=question,
                            final_answer=final_answer,
                            trajectory=trajectory,
                            trace_id=trace_id,
                            latency_ms=total_latency,
                            rag_result=self._last_rag_result,
                        )

                    # Parse tool call
                    action_match = re.search(r'ACTION:\s*(\w+)\(["\']?([^"\'()]*)["\']?\)', response_text)
                    if action_match:
                        tool_name = action_match.group(1)
                        tool_arg = action_match.group(2)
                        step_span.set_attribute("agent.tool", tool_name)
                        step_span.set_attribute("agent.tool_input", tool_arg[:200])

                        if tool_name in self.tools:
                            tool_result = self.tools[tool_name](tool_arg)
                            step = AgentStep(
                                step=step_num,
                                thought=response_text,
                                tool=tool_name,
                                tool_input=tool_arg,
                                tool_output=str(tool_result)[:500],
                            )
                            context_log += f"\nStep {step_num}: Used {tool_name}('{tool_arg}') -> {str(tool_result)[:300]}"
                        else:
                            step = AgentStep(
                                step=step_num,
                                thought=response_text,
                                tool=tool_name,
                                tool_input=tool_arg,
                                tool_output=f"Error: unknown tool '{tool_name}'",
                            )
                            context_log += f"\nStep {step_num}: Tool '{tool_name}' not found."
                    else:
                        step = AgentStep(step=step_num, thought=response_text)
                        context_log += f"\nStep {step_num}: Could not parse action. Try again."

                    trajectory.append(step)

            # Exceeded max steps
            total_latency = (time.time() - start) * 1000
            root_span.set_attribute("agent.steps_taken", self.max_steps)
            root_span.set_attribute("agent.exceeded_max_steps", True)
            trace_id = format(root_span.get_span_context().trace_id, "032x")

        return AgentResult(
            question=question,
            final_answer="Agent exceeded maximum steps without reaching a final answer.",
            trajectory=trajectory,
            trace_id=trace_id,
            latency_ms=total_latency,
            rag_result=self._last_rag_result,
        )
