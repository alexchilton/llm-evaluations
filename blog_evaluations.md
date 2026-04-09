# Beyond Vibes: A Practical Guide to LLM Evaluation

*How we built reproducible evaluation pipelines for DNN models and RAG systems — and what the numbers actually told us.*

---

## The Problem with "It Seems to Work"

Every team building with LLMs has the same moment: the demo looks great, someone asks "how do we know it's actually good?" — and the room goes quiet.

Evaluation is the unglamorous work that separates prototypes from production systems. Without it, you're shipping on vibes. We built two evaluation pipelines — one for DNN model quality and one for RAG retrieval accuracy — and learned some uncomfortable truths along the way.

This post covers what we measured, how we measured it, and what surprised us.

---

## Two Evaluation Domains, One Philosophy

We tackled evaluation across two very different contexts:

**DNN Model Evaluation** — assessing language model capabilities across reasoning, function calling, domain-specific benchmarks, and custom task suites. The question: *which model variant actually performs best for our use case?*

**RAG System Evaluation** — measuring retrieval quality, answer faithfulness, and detecting drift over time. The question: *is our RAG pipeline degrading, and where?*

Both share the same core principle: **measure what matters, automate the measurement, and track it over time.**

---

## DNN Evaluation: What We Measured

### Standard Benchmarks Are a Starting Point, Not an Answer

We used established frameworks — lm-evaluation-harness for general capabilities, NVIDIA NeMo for structured evaluation, and Berkeley Function Calling Leaderboard (BFCL) for tool-use accuracy.

The standard benchmarks tell you how a model performs *in general*. They don't tell you how it performs *on your tasks*. We learned this the hard way: a model that scored well on MMLU underperformed on our domain-specific extraction tasks.

### Custom Benchmarks Fill the Gap

The most valuable evaluations were custom ones:

- **Domain extraction accuracy** — can the model reliably extract structured data from our specific document types?
- **Instruction following precision** — does it respect output format constraints (JSON schema, field ordering)?
- **Latency under load** — benchmark scores mean nothing if inference takes 30 seconds per request

We built these as parameterised test suites that run in CI. Every model candidate goes through the same battery before anyone looks at the results.

### The CI Pipeline

The evaluation pipeline runs automatically:

1. Pull the candidate model
2. Run standard benchmarks (lm-eval, NeMo, BFCL)
3. Run custom task suites
4. Generate a comparison dashboard
5. Flag regressions against the current production model

No human in the loop until the dashboard is ready. This matters — manual evaluation doesn't scale and introduces selection bias.

---

## RAG Evaluation: Measuring Retrieval Health

RAG systems fail in subtle ways. The retriever returns *something*, the generator produces *something* — and it all looks plausible until a user points out the answer is wrong.

### The Three Failure Modes

1. **Retrieval failure** — the right chunks aren't in the context window
2. **Synthesis failure** — the right chunks are there, but the model ignores or misinterprets them
3. **Drift** — the system worked last month, but embedding quality or chunk boundaries have degraded

### What We Measured

| Metric | What It Catches |
|--------|----------------|
| **Context Precision** | Are relevant chunks ranked highly? |
| **Context Recall** | Are all relevant chunks retrieved? |
| **Faithfulness** | Does the answer actually follow from the retrieved context? |
| **Answer Relevance** | Does the answer address the question? |
| **Semantic Similarity** | Does the answer match the reference? |

### Drift Detection

The most operationally valuable metric was **drift tracking**. We run the same evaluation suite weekly against a fixed test set. When scores drop, we investigate before users complain.

Three drift patterns we found:

- **Embedding drift** — after a model update, cosine similarity distributions shifted, changing what gets retrieved
- **Chunk boundary drift** — document preprocessing changes silently broke context that used to be in single chunks
- **Score distribution drift** — overall accuracy held steady, but variance increased — the system became less predictable

Each of these would have been invisible without continuous measurement.

---

## LLM-as-Judge: Using Models to Evaluate Models

For subjective quality dimensions — clarity, helpfulness, tone — we use LLM-as-judge evaluation. A separate model scores the output on defined criteria.

### The Setup

```python
JUDGE_PROMPT = """
Score this response on:
- clarity (0-1): Is it clear and actionable?
- reasoning (0-1): Is the logic sound?
- completeness (0-1): Does it address the full question?

Output JSON: {"clarity": 0.0, "reasoning": 0.0, "completeness": 0.0}
"""
```

We use a different (often smaller) model as the judge to avoid self-evaluation bias. The judge model doesn't need to be powerful — it needs to be consistent.

### Calibration

LLM judges have biases. We calibrated ours by:

1. Having 3 humans score 50 examples independently
2. Running the LLM judge on the same examples
3. Measuring correlation (Spearman rank) between human and LLM scores
4. Adjusting the rubric until correlation exceeded 0.75

Without calibration, LLM-as-judge scores are just numbers. With calibration, they're useful proxies.

---

## What We Learned

### 1. Benchmark Scores Lie (Sometimes)

A model's published benchmark score tells you how it performs on that benchmark's distribution. Your data has a different distribution. Always build custom evaluations for your actual use case.

### 2. Continuous Evaluation Catches What Spot Checks Miss

Running evaluations once during development is table stakes. Running them weekly in production is where the real value lives. Drift is slow, silent, and expensive when it reaches users.

### 3. The Dashboard Is the Product

Nobody reads evaluation logs. A well-designed dashboard that shows trends, flags regressions, and compares candidates — that's what actually changes decisions. We invested heavily in the visualisation layer and it paid off immediately.

### 4. Evaluation Itself Has a Cost

Running full evaluation suites takes compute time and LLM tokens. We budget for it explicitly: ~5% of our inference spend goes to evaluation. This feels expensive until you consider the cost of shipping a regression.

---

## Try It Yourself

Both evaluation pipelines are available as interactive Jupyter notebooks:

- **[DNN Evaluation Notebook](nvidia_dnn_eval.ipynb)** — complete pipeline from standard benchmarks through custom suites to CI integration
- **[RAG Evaluation Notebook](rag_eval.ipynb)** — retrieval metrics, faithfulness scoring, and drift detection

The notebooks run locally against any OpenAI-compatible endpoint. Start with the RAG notebook — the drift simulation sections work without any API keys.

---

## Bottom Line

Evaluation isn't optional and it isn't a one-time task. Build it into your pipeline from day one, measure what matters to your users (not what's easy to measure), and track it continuously.

The models will keep getting better. Your ability to *know* they're getting better — and catch when they're not — is the competitive advantage.

---

*Alex Chilton is a software engineer exploring the intersection of AI evaluation and engineering practices. The notebooks referenced in this post are available on [GitHub](https://github.com/alexchilton).*
