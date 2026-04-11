# Prompt: Build a Production RAG Pipeline with Phoenix Evaluation

## Context

This project already has:
- `rag_real.ipynb` — a working RAG pipeline (FAISS + local Qwen via llama.cpp) but notebook-only
- `observability/phoenix_tutorial.ipynb` — Phoenix tracing/evaluation tutorial
- Phoenix running at `http://localhost:6006` (launchd service)
- Local Qwen model: `qwen3-30b-instruct` at `http://127.0.0.1:8002/v1` (llama.cpp)
- Phoenix annotation configs registered: `hallucination` (CATEGORICAL), `relevance` (CATEGORICAL), `correctness` (CONTINUOUS)

The goal is to build a **proper Python package** (not notebooks) implementing a RAG pipeline with an agent wrapper, drift detection, and first-class Phoenix observability and evaluation — following Phoenix best practices.

---

## What to Build

### 1. Package Structure

```
rag/
├── __init__.py
├── config.py            # LLM_BASE, PHOENIX_URL, MODEL_NAME, INDEX_DIR constants
├── pipeline.py          # RAGPipeline — retrieve + generate, returns trace_id
├── indexer.py           # DocumentIndexer — fetch, chunk, embed, FAISS index
├── retriever.py         # Retriever — FAISS search, returns chunks with scores + doc_ids
├── generator.py         # Generator — LLM call inside OTel span, returns answer + trace_id
├── agent.py             # RAGAgent — ReAct tool-using agent wrapping RAGPipeline
├── drift.py             # DriftDetector — statistical drift detection over eval scores
├── dataset.py           # QADataset — load/save QA pairs, upload to Phoenix datasets
├── runner.py            # EvaluationRunner — orchestrates pipeline + evals + posting
└── evaluators/
    ├── __init__.py
    ├── code_evals.py    # Deterministic evaluators (hit rate, MRR, exact match)
    ├── llm_evals.py     # LLM-as-judge (hallucination, relevance, correctness)
    ├── agent_evals.py   # Agent-specific (tool accuracy, step efficiency)
    └── poster.py        # post_eval_scores() — POST results to Phoenix trace annotations
```

---

### 2. Data

The pipeline uses **real Wikipedia documents** (same source as `rag_real.ipynb`):
- 30 articles across 3 topic clusters (AI/ML, History, Science)
- Chunked into ~500-token passages with 50-token overlap
- Embedded with `all-MiniLM-L6-v2` and indexed in FAISS

**QA pairs** are generated from article titles (one question per article, no LLM needed), saved to `qa_pairs.json`. Each QA pair has:
```python
{
  "question": "What is the transformer architecture?",
  "expected_answer": "...",          # opening sentences of the article
  "relevant_doc_ids": ["doc_7"],     # ground truth for retrieval eval
  "cluster": "AI/ML"
}
```

Start with the existing `qa_pairs.json` from `rag_real.ipynb` — do not regenerate.

---

### 3. RAGPipeline

Fully instrumented with OpenTelemetry from the start. Every query produces a trace with nested spans:

```
Trace: rag_query
  ├── Span: embed_query      (latency, embedding_model)
  ├── Span: faiss_retrieve   (latency, top_k, doc_ids, scores)
  └── Span: llm_generate     (latency, model, tokens, answer)
```

```python
class RAGPipeline:
    def query(self, question: str) -> RAGResult:
        """Returns answer, retrieved_chunks, trace_id, latency_ms."""
```

**Capture and return `trace_id`** from every query — this is the link between pipeline output and Phoenix evaluation scores.

---

### 4. RAGAgent

A ReAct-style tool-using agent that wraps `RAGPipeline` as one of several tools. Useful for questions that need calculation, date logic, or multi-hop retrieval.

```
Trace: agent_run
  ├── Span: agent_step_1  (thought, tool=rag_query, tool_input)
  │   └── Span: rag_query (full RAG sub-trace nested inside)
  ├── Span: agent_step_2  (thought, tool=calculator, tool_input)
  └── Span: agent_final   (final_answer, steps_taken)
```

```python
class RAGAgent:
    tools = [rag_tool, calculator_tool, date_tool]

    def run(self, question: str) -> AgentResult:
        """Returns final_answer, trajectory (list of steps), trace_id."""
```

---

### 5. Evaluators — Three Tiers

#### Tier 1: Code Evaluators (`code_evals.py`) — instant, free

```python
hit_rate_at_k(retrieved_ids, relevant_ids, k=5) -> float
mrr(retrieved_ids, relevant_ids)                -> float
precision_at_k(retrieved_ids, relevant_ids, k=5) -> float
contains_expected(answer, expected)             -> float  # keyword overlap
exact_match(answer, expected)                   -> float
answer_length_ok(answer, min=10, max=500)       -> float  # sanity check
```

#### Tier 2: LLM-as-Judge (`llm_evals.py`) — slow, powerful

Using `phoenix.evals.create_evaluator` + local Qwen. Match the annotation configs already in Phoenix:

```python
hallucination_eval(context, generated_answer)
    -> {"label": "FAITHFUL"|"HALLUCINATED", "score": 1.0|0.0}

relevance_eval(question, generated_answer)
    -> {"label": "HIGH"|"MEDIUM"|"LOW", "score": 1.0|0.5|0.0}

correctness_eval(expected, generated_answer)
    -> {"score": 0.0-1.0}
```

Use `asyncio.Semaphore(2)` to limit concurrent LLM judge calls (30B model, 7-30s per call).

#### Tier 3: Agent Evaluators (`agent_evals.py`)

```python
tool_accuracy(optimal_tool, trajectory)   -> float  # right tool chosen?
step_efficiency(trajectory)               -> float  # steps vs optimal
final_answer_correct(expected, answer)    -> float
```

---

### 6. Drift Detection (`drift.py`)

Compares a **baseline** evaluation run against a **current** run and flags regressions. After the first clean run, save baseline stats to `baseline.json`. On subsequent runs, load and compare.

```python
class DriftDetector:
    def __init__(self, baseline_path: str = "baseline.json"):
        ...

    def fit(self, results: pd.DataFrame) -> None:
        """Save mean/std per metric as baseline."""

    def check(self, current: pd.DataFrame) -> DriftReport:
        """
        Per metric: mean delta, z-score, is_drifted (|z| > 2.0).
        Posts a drift_check span to Phoenix — drift events appear in Tracing timeline.
        """
```

Metrics tracked: `hit_rate`, `mrr`, `hallucination_score`, `relevance_score`, `correctness_score`, `latency_ms`.

If `DriftReport.any_drift` is True, post a warning span to Phoenix so it appears alongside regular query traces.

---

### 7. Evaluation Runner (`runner.py`)

Orchestrates pipeline queries, evaluations, and Phoenix posting in one call:

```python
class EvaluationRunner:
    def run(
        self,
        qa_pairs: list[QAPair],
        pipeline: RAGPipeline | RAGAgent,
        evaluators: list,                # mix of code + LLM evaluators
        post_to_phoenix: bool = True,
        drift_detector: DriftDetector | None = None,
    ) -> pd.DataFrame:
        """
        For each QA pair:
          1. pipeline.query(question) -> answer + trace_id
          2. run code evaluators (all rows)
          3. run LLM evaluators (only rows where code evals flag issues)
          4. post_to_phoenix: POST scores to /v1/trace_annotations
          5. if drift_detector: check for drift, post warning span
        Returns full results DataFrame.
        """
```

**Strategy:** run cheap code evals on everything first, then only run expensive LLM evals on rows where code evals flag problems (low hit rate, short answer, etc.). This avoids 30 × 30s = 15 minutes of judge calls for a healthy pipeline.

---

### 8. Phoenix Integration — Correct Pattern

The working pattern (proven in this project):

```python
from phoenix.evals import evaluate_dataframe

results_df = evaluate_dataframe(dataframe=df, evaluators=[hallucination_eval, relevance_eval])

# Post back to Phoenix as trace annotations
requests.post(
    "http://localhost:6006/v1/trace_annotations",
    json={"data": [
        {
            "trace_id": trace_id,          # hex, from pipeline.query()
            "name": "hallucination",
            "annotator_kind": "LLM",
            "result": {"label": "FAITHFUL", "score": 1.0}
        }
    ]}
)
```

**Do not** use the Phoenix UI evaluator feature for local models — it has timeout and custom provider limitations that make it unreliable with llama.cpp.

---

### 9. Dataset Upload to Phoenix

Upload QA pairs as a Phoenix dataset so evaluation runs appear in the Datasets & Experiments tab:

```python
# POST /v1/datasets  → create dataset
# POST /v1/datasets/{id}/examples  → upload QA pairs as examples
# Each experiment run then references this dataset by ID
```

---

## Key Design Decisions

1. **Trace ID is mandatory** — every `pipeline.query()` must return the trace_id. Without it you cannot link eval scores to traces in Phoenix.

2. **Code evals gate LLM evals** — run cheap evals on everything, expensive LLM judge only on flagged rows. Avoids unnecessary inference cost.

3. **Indexing is one-time** — `DocumentIndexer.build()` saves to disk. `RAGPipeline` loads the saved index at startup. Never re-index on every run.

4. **No notebooks in the package** — the `rag/` package is pure Python. Notebooks (or a CLI script) import from it; the package has no notebook dependency.

5. **Drift baseline is persistent** — saved to `baseline.json` after first clean run, loaded on all subsequent runs. The baseline is metrics from the first 30-query evaluation.

---

## Phoenix Annotation Configs Already Registered

No need to re-register these — they exist in Phoenix:

| Name | Type | Values |
|------|------|--------|
| `hallucination` | CATEGORICAL | FAITHFUL (1.0), HALLUCINATED (0.0) |
| `relevance` | CATEGORICAL | HIGH (1.0), MEDIUM (0.5), LOW (0.0) |
| `correctness` | CONTINUOUS | 0.0 – 1.0 |

---

## Implementation Order

1. `config.py` — constants only
2. `indexer.py` — extract from `rag_real.ipynb` (DocumentIndexer already there)
3. `retriever.py` — FAISS search, return chunks with scores and doc_ids
4. `generator.py` — LLM call inside OTel span, return answer + trace_id
5. `pipeline.py` — wire indexer + retriever + generator, single `query()` method
6. `evaluators/code_evals.py` — pure functions, no dependencies
7. `evaluators/llm_evals.py` — `create_evaluator` decorators, local Qwen judge
8. `evaluators/poster.py` — `post_eval_scores(eval_df, eval_name, trace_ids)`
9. `runner.py` — `EvaluationRunner` with code-first gating strategy
10. `drift.py` — `DriftDetector` with fit/check/post-span
11. `agent.py` — `RAGAgent` ReAct loop with tool spans
12. `evaluators/agent_evals.py` — tool accuracy, step efficiency
13. `dataset.py` — load `qa_pairs.json`, upload to Phoenix dataset

Validate each step with a 3-5 query smoke test before moving to the next.