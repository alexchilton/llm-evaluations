# RAG Testing & Drift Detection Guide

## Quick Start

```bash
cd /path/to/evaluations
source .venv/bin/activate
jupyter notebook rag_testing.ipynb
```

Run all cells. The notebook takes about 5 minutes (mostly waiting for LLM calls in Phase 5).

## What This System Does

Three things:

1. **Measures** 5 quality metrics on every RAG query
2. **Detects** when metrics drift from your established baseline
3. **Simulates** 7 failure modes so you can validate detection works before production breaks

## The 5 Metrics

| Metric | What It Tells You | Healthy Range |
|--------|------------------|--------------|
| **Contextual Precision** | Are the retrieved chunks relevant to the query? | > 0.8 |
| **Contextual Recall** | Did you retrieve ALL the relevant information? | > 0.6 |
| **Faithfulness** | Is the answer grounded in the retrieved context? | > 0.5 |
| **Answer Relevance** | Does the answer actually address the question? | > 0.3 |
| **Answer Correctness** | Does the answer match ground truth? | > 0.5 |

**Faithfulness is the most important.** Low faithfulness = hallucination = your RAG is making things up.

## How Drift Detection Works

### The Statistical Approach

1. You establish a **baseline** from 30-50 known-good queries
2. The system calculates mean and standard deviation for each metric
3. As new queries come in, it maintains a **rolling window** of recent results
4. It compares the rolling window against the baseline using:
   - **Welch's t-test**: Is the difference statistically significant?
   - **Cohen's d effect size**: Is the difference practically meaningful?

### Alert Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Warning** | Cohen's d > 0.8 OR mean < baseline - 2σ | Something changed, investigate |
| **Critical** | Cohen's d > 1.5 OR mean < baseline - 3σ | Pipeline is broken, act now |

### Why Cohen's d and Not Just P-Values?

P-values shrink with sample size. With enough queries, even a tiny irrelevant fluctuation becomes "statistically significant." Cohen's d measures the actual size of the difference:

- d < 0.2: trivial (ignore)
- d = 0.5: medium (investigate)
- d > 0.8: large (alert)
- d > 1.5: severe (something is very wrong)

## The 7 Failure Modes

### Retrieval Failures

| Mode | What It Simulates | Real-World Cause |
|------|------------------|-----------------|
| `embedding_drift` | Similarity scores degrade, ranking shuffled | Embedding model retrained, distribution shift |
| `retrieval_noise` | Irrelevant chunks get high scores | Index poisoning, new document types confusing retriever |
| `context_truncation` | Chunks removed before LLM sees them | Context window overflow, chunking bug |
| `index_corruption` | Chunk content replaced with garbage | Storage failure, encoding issue, migration bug |
| `staleness` | Chunks marked as outdated | Knowledge base not refreshed, temporal drift |

### Generation Failures

| Mode | What It Simulates | Real-World Cause |
|------|------------------|-----------------|
| `model_degradation` | LLM ignores context, gives generic answers | Model update, system prompt change, context too long |
| `latency_spike` | Response time multiplied up to 10x | Overloaded GPU, network issues, rate limiting |

## Reading the Dashboard

### When Everything Is Working

```
         metric  baseline_mean  baseline_std  current_mean  current_n  delta status
contextual_precision          0.89          0.15          0.87        20  -0.02     OK
   contextual_recall          0.74          0.10          0.73        20  -0.01     OK
        faithfulness          0.68          0.08          0.67        20  -0.01     OK
    answer_relevance          0.42          0.12          0.41        20  -0.01     OK
   answer_correctness          0.79          0.14          0.78        20  -0.01     OK
```

Small deltas, all "OK". This is healthy.

### When Retrieval Is Broken

```
         metric  baseline_mean  baseline_std  current_mean  current_n  delta  status
contextual_precision          0.89          0.15          0.45        15  -0.44  DRIFT  ← THIS
   contextual_recall          0.74          0.10          0.50        15  -0.24  DRIFT  ← THIS
        faithfulness          0.68          0.08          0.60        15  -0.08     OK
```

Precision and recall dropped but faithfulness is OK. **The retriever is broken but the LLM is fine.** Fix your embeddings/index.

### When the Model Is Broken

```
         metric  baseline_mean  baseline_std  current_mean  current_n  delta  status
contextual_precision          0.89          0.15          0.88        15  -0.01     OK
        faithfulness          0.68          0.08          0.20        15  -0.48  DRIFT  ← THIS
   answer_correctness          0.79          0.14          0.15        15  -0.64  DRIFT  ← THIS
```

Precision is fine but faithfulness collapsed. **The retriever works but the LLM is ignoring the context.** Check the model, system prompt, or context window.

## Phoenix Tracing

When Phoenix is running (http://localhost:6006), the notebook traces all LLM calls in Phase 5.

**What to look for in Phoenix:**
- **Latency distribution**: Are some queries much slower than others?
- **Token usage**: Is the context getting too long?
- **Error rate**: Are any calls failing silently?
- **Input/Output**: What exactly is the LLM seeing and producing?

## Integration with Your RAG Pipeline

### Minimum Integration (5 lines)

```python
from rag_testing_system import RAGTestingSystem, RetrievedChunk

# At startup
system = RAGTestingSystem()
system.establish_baseline(your_test_results)

# Per query
result, alerts = system.test_query(query, chunks, answer)
if alerts:
    logger.warning(f"RAG drift: {alerts}")
```

### With Phoenix Tracing

```python
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor

tracer_provider = register(
    project_name="my-rag",
    endpoint="http://localhost:6006/v1/traces",
)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
```

### With LLM-as-Judge (Upgrading from Keyword Overlap)

Subclass `MetricsCalculator` and override individual metrics:

```python
class LLMJudgeCalculator(MetricsCalculator):
    def faithfulness(self, answer, chunks):
        # Use your local Qwen to judge faithfulness
        prompt = f"Does this answer only contain info from the context? Score 0-1."
        score = call_llm(prompt)
        return float(score)
```

## File Structure

```
rag_testing_system.py    # Core: MetricsCalculator, DriftDetector, FailureSimulator, RAGTestingSystem
rag_testing.ipynb        # Interactive notebook: 6 phases, live LLM, plots
rag_testing_guide.md     # This file
```

## Dependencies

All in `requirements.txt`:
- numpy, scipy (statistics)
- pandas (DataFrames)
- matplotlib (plots)
- openai (LLM calls)
- arize-phoenix-otel, openinference-instrumentation-openai (tracing)

## Hardware Notes

Tested with:
- Qwen 3.5 35B via llama.cpp at `http://127.0.0.1:8001/v1`
- Phoenix at `http://localhost:6006`
- Python 3.14, macOS (M4 48GB)

The synthetic phases (1-4) run instantly (no LLM calls). Phase 5 takes ~60s per query with the 35B model.
