# LLM Evaluations

A collection of notebooks covering LLM evaluation, RAG testing, drift detection, agentic paradigms, and production observability — all running locally on llama.cpp with no cloud API dependencies for the core workflows.

**Stack:** Python · llama.cpp (Qwen 3.5 35B) · FAISS · sentence-transformers · Arize Phoenix · Langfuse · MLflow

---

## Notebooks

### 1. `agents_paradigms.ipynb` — When to Use Which Agent

Each agentic paradigm gets a different task designed to show where it genuinely excels, with a single-agent baseline for comparison so you can see the measurable difference rather than just the outputs.

| Paradigm | Task | What to look for |
|----------|------|-----------------|
| Single Agent | Classify a support ticket | Baseline latency and quality ceiling |
| Multi-Agent Pipeline | Architecture review | How decomposition + adversarial critique improves reasoning |
| Swarm | Open-ended brainstorm | Entropy curve: watch consensus form across generations |
| ADAS | Novel problem | The meta-agent designing its own architecture before running |
| ReAct | Multi-step tool use | Thought → action → observation loops in the trace |

**What to expect:** Rich terminal output with a comparison table (quality score, latency, tokens, estimated cost). Langfuse tracing optional — traces show up at `http://localhost:3001` if running.

---

### 2. `nvidia_dnn_eval.ipynb` — DNN & Agentic Evaluation Patterns

Maps evaluation techniques to production requirements: academic benchmarks, enterprise evaluation APIs, agentic trajectory scoring, and custom judge design.

| Section | What it covers |
|---------|---------------|
| EleutherAI LM Eval Harness | Running standard benchmarks and writing custom YAML tasks |
| NVIDIA NeMo Evaluator | Enterprise evaluation API patterns (microservice style) |
| Agentic Pipeline Eval | BFCL, tool-use scoring, trajectory evaluation |
| Custom Benchmarks | Building domain-specific evaluators from scratch |
| LLM-as-Judge | Designing, calibrating, and auditing judge prompts |

**What to expect:** Some sections use the Anthropic API (flagged in the notebook). The local-only sections run entirely via llama.cpp.

---

### 3. `rag_eval.ipynb` — RAG Quality Benchmarking

A comprehensive RAG evaluation framework covering established benchmark methodologies, drift simulation, and production monitoring patterns.

| Section | What it covers |
|---------|---------------|
| CRAG-style evaluation | Meta's Comprehensive RAG benchmark methodology |
| Drift simulation | Four drift types with statistical detection |
| Bloom-style behavioural eval | Testing for robustness, consistency, edge cases |
| Kaggle SDK patterns | Benchmark submission and scoring workflows |
| CI regression gate | Golden dataset pass/fail gate for continuous evaluation |

**What to expect:** Several sections call the Anthropic API — check the table in the notebook header. The drift detection and local evaluation sections run without API keys.

---

### 4. `rag_testing.ipynb` — RAG Monitoring with Drift Detection

A focused walkthrough of the monitoring and alerting layer for a RAG pipeline. Everything is synthetic (no real vector DB or embeddings) so phases run instantly — the point is the detection framework itself.

**Six phases:**

1. **Baseline** — 50 synthetic queries establish normal metric distributions
2. **Normal traffic** — 20 queries, confirms no false alerts
3. **Retrieval noise** — Irrelevant chunks injected; precision and recall drop, faithfulness holds
4. **Model degradation** — LLM ignores context; faithfulness collapses while precision stays healthy
5. **Live RAG** — Real queries to local Qwen with Phoenix tracing
6. **Dashboard** — Metric timeline plot with baseline band and drift alerts

**Appendix** demos all 7 failure modes (embedding drift, retrieval noise, context truncation, index corruption, staleness, model degradation, latency spike) with scores for each.

**What to expect:** Phases 1–4 complete in seconds (no LLM calls). Phase 5 takes ~60s per query with the 35B model. Key insight: different failures break different metrics — retrieval problems tank precision/recall, generation problems tank faithfulness. See `rag_testing_guide.md` for the full interpretation guide.

> **Note:** Metrics in this notebook use keyword overlap (Jaccard similarity), not semantic similarity. It's a monitoring harness demo, not a complete RAG system. See `rag_real.ipynb` for the real thing.

---

### 5. `rag_real.ipynb` — Real RAG Pipeline with Monitoring

The version of `rag_testing.ipynb` where nothing is faked. Real documents, real embeddings, real retrieval, real LLM generation, real LLM-as-judge metrics.

**What's real:**
- 30 Wikipedia articles across 3 topic clusters (AI/ML, History, Science), fetched and chunked with LangChain's `RecursiveCharacterTextSplitter`
- Embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`) stored in a FAISS index
- Retrieval via vector similarity search
- Answers generated by local Qwen via llama.cpp
- All 5 metrics scored by Qwen-as-judge (not keyword overlap)

**Six phases:**

1. **Build the index** — Fetch articles, chunk, embed, save to `rag_index/`; generate 30 QA pairs
2. **Establish baseline** — Run all 30 QA pairs through the full pipeline
3. **Normal traffic** — Confirm no false drift alerts
4. **Inject retrieval failure** — Corrupt the FAISS index; watch precision/recall break
5. **Inject generation failure** — Override system prompt; faithfulness collapses
6. **Recovery** — Restore and confirm metrics return to baseline band

**What to expect:** Phase 0 (index build + QA generation) takes several minutes on first run, then caches to `rag_index/`. The 3-cluster document design means you can test cross-topic retrieval confusion. The LLM-as-judge calls add latency — budget ~10–15 minutes for a full run.

---

### 6. `observability/phoenix_tutorial.ipynb` — Production LLM Observability with Phoenix

A complete tutorial from basic tracing to production-grade monitoring, using entirely local infrastructure. Runs against Phoenix at `http://localhost:6006`.

| Section | What it covers |
|---------|---------------|
| 0 | The Phoenix philosophy — observability vs. logging vs. evaluation |
| 1 | Instrumentation — auto-capture LLM calls with OpenTelemetry |
| 2 | Datasets & experiments — running A/B tests via the Phoenix client |
| 3 | Evaluation hierarchy — deterministic → heuristic → LLM-as-judge |
| 4 | RAG evaluation — retrieval metrics separate from generation metrics |
| 5 | Agent evaluation — multi-step trajectory scoring |
| 6 | Hidden features — span-level eval, human-in-the-loop export, trace → fine-tuning data |
| 7 | Meta-evaluation — testing the judge itself for consistency and bias |
| 8 | Prompt versioning — iterating prompts via experiment metadata |
| 9 | Prompt management — create, version, and tag prompts via the Phoenix client API |
| 10 | Span annotations — attaching human and LLM-judge feedback to traces programmatically |
| 11 | Native experiments — datasets and experiment runs that appear in the Phoenix UI |

**What to expect:** Sections 1–8 are mostly self-contained code with Rich output. Sections 9–11 interact with the Phoenix UI — open `http://localhost:6006` alongside the notebook to see prompts, annotations, and experiment comparisons appear in real time. LLM judge cells (Sections 3, 4, 7) take several minutes at 35B scale. The toxicity evaluator (`detoxify`) downloads a BERT model on first run.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy env template (only needed for API-dependent notebook sections)
cp .env.example .env

# Ensure llama.cpp is serving Qwen at http://127.0.0.1:8001/v1
# Ensure Phoenix is running at http://localhost:6006

# Launch any notebook
jupyter notebook
```

## Hardware

Tested on M4 Mac, 48GB RAM. llama.cpp serving `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` fully GPU-offloaded at `http://127.0.0.1:8001/v1` (alias: `unsloth/Qwen3.5-35B-A3B`).

For smaller hardware, reduce swarm parameters in `config.yaml` and expect longer inference times in judge-heavy cells.

## Project Structure

```
agents_paradigms.ipynb       # Agentic paradigm comparison (5 paradigms)
nvidia_dnn_eval.ipynb        # DNN & agentic evaluation patterns
rag_eval.ipynb               # RAG quality benchmarking framework
rag_testing.ipynb            # RAG drift detection (synthetic, fast)
rag_testing_system.py        # Drift detection + failure simulation classes
rag_testing_guide.md         # Interpretation guide for rag_testing.ipynb
rag_real.ipynb               # Real RAG pipeline with FAISS + LLM judge
rag_real_system.py           # RAG pipeline classes for rag_real.ipynb
rag_qa_pairs.jsonl           # Ground truth QA pairs (generated at runtime)
observability/
└── phoenix_tutorial.ipynb   # Full Phoenix observability + evaluation tutorial
paradigms/                   # Agent paradigm implementations
eval/                        # LLM-as-judge scorer
config.yaml                  # Runtime configuration
docker-compose.yml           # Langfuse stack
.env.example                 # Environment variable template
```
