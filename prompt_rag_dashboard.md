# Prompt: Real RAG Evaluation Dashboard Notebook

## What This Is

A single Jupyter notebook — `rag_dashboard.ipynb` — that serves as the **live control panel** for the RAG system. It runs real queries through a real pipeline, evaluates them, detects drift, tests the agent, and visualises everything in one place. No synthetic data, no placeholder numbers.

The notebook imports from the `rag/` Python package (described in `prompt2to.md`). Do not inline the pipeline logic inside the notebook — keep it clean by calling `from rag import RAGPipeline, RAGAgent, EvaluationRunner, DriftDetector`.

---

## Infrastructure

- **LLM:** `qwen3-30b-instruct` at `http://127.0.0.1:8002/v1` (llama.cpp, OpenAI-compatible)
- **Phoenix:** `http://localhost:6006` — all traces and eval scores posted here
- **Embeddings:** `all-MiniLM-L6-v2` (sentence-transformers, local, no API key)
- **Index:** FAISS, pre-built from 30 Wikipedia articles, loaded from disk
- **QA pairs:** loaded from `qa_pairs.json` (pre-generated, do not regenerate)
- Add `/no_think` to all LLM prompts and use `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` to suppress Qwen3 thinking tokens

---

## Notebook Structure

### Section 0 — Setup & Health Check

- Import `rag.*`, connect to Phoenix, load FAISS index
- Print a health table: Phoenix reachable ✓, LLM reachable ✓, index loaded (N docs) ✓, QA pairs loaded (N pairs) ✓
- Show the 30 QA pairs in a DataFrame so the reader knows what data is being used

---

### Section 1 — Run the Pipeline (Real Queries)

Run all 30 QA pairs through `RAGPipeline`. Show a live progress bar.

For each query, capture: `question`, `answer`, `retrieved_doc_ids`, `retrieved_scores`, `trace_id`, `latency_ms`.

Display results as a scrollable DataFrame. Each row is one query. Include a Phoenix link per row: `http://localhost:6006/traces/{trace_id}`.

---

### Section 2 — Retrieval Quality

Code evaluators only (instant). For each query, compute:

- `hit_rate@5` — did the right doc appear in top 5?
- `mrr` — where did it appear?
- `precision@5` — what fraction of top 5 were relevant?

**Visualisations:**

1. **Bar chart** — hit rate per topic cluster (AI/ML vs History vs Science). Which cluster retrieves worst?
2. **Scatter plot** — MRR vs retrieval score (does a higher FAISS score mean better ranking?)
3. **Failure table** — queries where `hit_rate@5 = 0`. These are the retrieval failures. Show question + what was actually retrieved vs what was expected.

---

### Section 3 — Answer Quality (LLM-as-Judge)

Run `hallucination_eval`, `relevance_eval`, `correctness_eval` via `phoenix.evals.evaluate_dataframe`. Use `asyncio.Semaphore(2)` to limit concurrency. Show a progress bar — this takes ~5 minutes for 30 queries.

Post all scores back to Phoenix as trace annotations via `POST /v1/trace_annotations`.

**Visualisations:**

1. **Stacked bar** — hallucination breakdown: FAITHFUL vs HALLUCINATED per cluster
2. **Distribution plot** — correctness scores (0.0–1.0) as a histogram. Where does the pipeline cluster?
3. **Heatmap** — question × metric (hit_rate, mrr, hallucination, relevance, correctness). One row per query, colour = score. Instantly shows which queries are failing across all dimensions.
4. **Correlation plot** — does retrieval quality (MRR) correlate with answer quality (correctness)? Scatter with regression line.
5. **Worst 5 table** — bottom 5 queries by overall score (mean across all metrics). Show question, answer, expected, and which metrics failed.

---

### Section 4 — Baseline & Drift Detection

After Section 3, save the current run as the baseline if no `baseline.json` exists yet. If it does exist, load it and run `DriftDetector.check()`.

**Visualisations:**

1. **Metric comparison bar chart** — current vs baseline, side by side, for each metric. Green = within 1 std dev, amber = 1-2 std dev, red = drifted (>2 std dev).
2. **Drift timeline** — if multiple runs exist (load from a `run_history.jsonl` log), plot each metric over time as a line chart. Horizontal band = baseline ± 1 std dev.
3. **Drift report table** — one row per metric: baseline mean, current mean, delta, z-score, drifted (yes/no).

Save the current run to `run_history.jsonl` (append, one JSON line per run with timestamp + metric means).

---

### Section 5 — Agent Evaluation

Run 10 questions through `RAGAgent` — a mix of:
- Questions that need RAG (factual lookup)
- Questions that need the calculator tool (e.g. "How many years between X and Y?")
- Questions that need both

Show the full trajectory for each run: thought → tool → observation → final answer.

**Visualisations:**

1. **Trajectory table** — one row per agent step across all runs. Columns: run_id, step, thought (truncated), tool_used, tool_input, observation (truncated), correct.
2. **Tool usage bar chart** — how many times each tool was called. Did the agent overuse RAG?
3. **Step count distribution** — histogram of steps per question. Optimal = 2-3. Flag any that took 6+.
4. **Agent vs Direct RAG table** — for the questions that could be answered by RAG alone, compare: agent answer vs direct pipeline answer vs expected. Did the agent add value or just add latency?

Post agent eval scores to Phoenix (`tool_accuracy`, `step_efficiency`, `final_answer_correct`) as trace annotations.

---

### Section 6 — End-to-End Summary

A single summary section pulling everything together.

**Visualisations:**

1. **Radar chart** — one axis per metric (hit_rate, mrr, hallucination, relevance, correctness, agent_accuracy). Shows overall pipeline health at a glance.
2. **Summary table** — one row per section, green/amber/red RAG status:

| Component | Score | Status |
|-----------|-------|--------|
| Retrieval (hit@5) | 0.73 | OK |
| Retrieval (MRR) | 0.61 | OK |
| Hallucination | 87% faithful | OK |
| Relevance | 0.82 avg | OK |
| Correctness | 0.71 avg | WARN |
| Drift | none detected | OK |
| Agent accuracy | 0.80 | OK |

3. **Phoenix link block** — direct URLs to:
   - All traces from this run: `http://localhost:6006` → phoenix-tutorial project
   - This run's evaluation scores in the annotation panel

---

## Style & Code Conventions

- Use `matplotlib` and `pandas` — already installed, no new dependencies
- Each section starts with a markdown cell explaining what is being measured and why it matters (like the phoenix_tutorial.ipynb teaching style)
- Progress bars via `tqdm` for any loop that calls the LLM
- All LLM calls go through `rag.pipeline` or `rag.agent` — no inline `client.chat.completions.create()` in the notebook
- Cells should be runnable independently where possible (load from saved files rather than re-running the whole pipeline)
- Save pipeline results to `pipeline_results.csv` after Section 1 so later sections can reload without re-running queries
- Save eval results to `eval_results.csv` after Section 3 so drift and summary sections can reload

---

## What NOT to Do

- Do not use synthetic/fake data anywhere
- Do not re-generate QA pairs — load from `qa_pairs.json`
- Do not re-build the FAISS index — load from disk
- Do not inline `DocumentIndexer`, `RAGPipeline` etc. inside the notebook — import from `rag/`
- Do not use the Phoenix UI evaluator feature for LLM evals — use `phoenix.evals.evaluate_dataframe` + `post_eval_scores()` (see `prompt2to.md` for the correct pattern)
- Do not add `temperature` to invocation parameters for Phoenix custom providers (known bug in Phoenix 14.x)
