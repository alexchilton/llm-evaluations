# Agentic Paradigms Demo

Four agentic paradigms solve the same problem — **"What should a software team prioritize next quarter?"** — so you can compare their approaches, quality, and cost side-by-side.

## Quick Start

```bash
# 1. Start Langfuse (optional — demo works without it)
docker compose up -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Make sure your LLM is running (llama.cpp / Ollama / etc.)
#    Default: http://127.0.0.1:8001/v1

# 4. Run the demo
python demo.py
```

## The Four Paradigms

| # | Paradigm | How It Works | When to Use |
|---|----------|-------------|-------------|
| 1 | **Single Agent** | One LLM call with a well-crafted prompt | Simple, well-defined problems; latency-sensitive |
| 2 | **Multi-Agent Pipeline** | Analyst → Critic → Synthesiser chain | Problems that benefit from decomposition and adversarial review |
| 3 | **Swarm (Stigmergic)** | 8 agents communicate via shared pheromone board, no coordinator | Brainstorming; when optimal structure is unknown; "wisdom of crowds" |
| 4 | **ADAS** | Meta-agent designs the agent architecture, then runtime executes it | Novel problems; when the right decomposition isn't obvious |

## CLI Options

```bash
python demo.py                           # Run all 4 paradigms
python demo.py -p single                 # Run just one
python demo.py -p swarm --swarm-agents 12 --swarm-generations 8
python demo.py --no-eval                 # Skip LLM-as-judge scoring
python demo.py --config my_config.yaml   # Custom config
```

## Reading the Output

### Comparison Table
After all paradigms complete, a table shows:
- **Quality score** (0–1, from LLM-as-judge)
- **Latency** (wall-clock seconds)
- **Token count** (prompt + completion)
- **Estimated cost** (simulated cloud pricing)

### Swarm Entropy Curve
The swarm displays an entropy value per generation:
- **High entropy** (>1.5): agents disagree, many competing hypotheses
- **Falling entropy**: consensus is forming
- **Low entropy** (<0.5): strong convergence on dominant hypotheses
- **Convergence generation**: when entropy drops below threshold (default 0.8)

### ADAS Architecture
Before running, ADAS prints the architecture it designed — including topology (pipeline/parallel/hybrid) and each agent's role. This is the "wow moment."

## Reading the Langfuse Dashboard

If Langfuse is running (`docker compose up -d`), open http://localhost:3001:

| Paradigm | What to Look For |
|----------|-----------------|
| Single | One trace, one generation span — check latency and token usage |
| Multi-Agent | Three nested spans (analyst, critic, synthesiser) — look for bottlenecks |
| Swarm | Many generation spans — look for convergence patterns in metadata |
| ADAS | Meta-agent span + dynamically spawned agent spans — inspect architecture artifact |

## Configuration

Edit `config.yaml` to change:
- Model IDs and LLM endpoint
- Swarm parameters (agents, generations, convergence threshold)
- ADAS max agent count
- Langfuse connection details

Copy `.env.example` to `.env` and fill in Langfuse keys if using Langfuse.

## Project Structure

```
demo.py                          # Main entrypoint, CLI + Rich UI
├── paradigms/
│   ├── single_agent.py          # Paradigm 1: one LLM call
│   ├── multi_agent.py           # Paradigm 2: 3-agent pipeline
│   ├── swarm.py                 # Paradigm 3: stigmergic swarm
│   └── adas.py                  # Paradigm 4: meta-agent designs + runs
├── observability/
│   ├── logger.py                # Unified structured logger
│   ├── langfuse_client.py       # Langfuse wrapper (graceful degradation)
│   └── swarm_db.py              # SQLite pheromone board
├── eval/
│   └── judge.py                 # LLM-as-judge scorer
├── config.yaml                  # Runtime configuration
├── docker-compose.yml           # Langfuse stack
└── .env.example                 # Environment variable template
```

## Hardware Note

Tested on M4 Mac with 48GB RAM running Qwen 3.5 35B (A3B variant, Q4_K_M) via llama.cpp.

Paradigms run sequentially to stay within memory budget. The swarm is the most LLM-intensive (agents × generations calls). Reduce `swarm.num_agents` or `swarm.num_generations` in config.yaml if you're running a smaller model.
