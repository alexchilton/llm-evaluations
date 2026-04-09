# Four Ways to Build an AI Agent (And When Each One Wins)

*We solved the same problem with four different agentic paradigms. Here's what we learned about when simplicity beats sophistication — and when it doesn't.*

---

## The Experiment

There's a growing zoo of agentic architectures: single agents, multi-agent pipelines, swarms, self-designing systems. Each comes with bold claims. We wanted data, not claims.

So we took one problem — *"What should a software team prioritize next quarter?"* — and solved it four ways. Same input, same LLM, same evaluation criteria. The only variable: the architecture.

The results were instructive. And occasionally surprising.

---

## The Setup

All four paradigms received the same input: a realistic backlog of 8 items ranging from database upgrades to billing refactors. Each paradigm produced a prioritised recommendation. An LLM-as-judge scored every output on clarity, reasoning, and completeness.

| Component | Choice |
|-----------|--------|
| LLM | Qwen 3.5 35B (local, via llama.cpp) |
| Agent Framework | smolagents (HuggingFace) |
| Swarm State | SQLite |
| Evaluation | LLM-as-judge |

Everything runs locally on a single machine. No cloud APIs, no rate limits, no costs beyond electricity.

---

## Paradigm 1: Single Agent 🎯

**One expert. One prompt. One answer.**

The simplest possible architecture: a well-crafted system prompt ("You are an experienced engineering manager…") and a single LLM call.

```
Input → LLM → Output
```

This is the baseline every other paradigm must beat. If your multi-agent system doesn't outperform a good prompt, you're adding complexity for nothing.

**Result:** Consistently solid. The single agent produced clear, actionable recommendations. Latency was lowest. Token cost was lowest. Quality was… good enough for most purposes.

**When to use:** Known problem domains, latency-sensitive applications, quick first drafts, and as a baseline for evaluating fancier approaches.

---

## Paradigm 2: Multi-Agent Pipeline 🔗

**Analyst → Critic → Synthesiser**

Three agents in sequence, each with a focused role:

1. **Analyst** — scores every backlog item on impact and effort
2. **Critic** — challenges the analyst's assumptions, flags risks
3. **Synthesiser** — resolves the tension into a final recommendation

The handoff between agents uses structured JSON. This creates inspectable boundaries — you can see exactly what the analyst proposed and exactly how the critic pushed back.

**Result:** The critic caught things the single agent missed. In one run, it flagged that the analyst overweighted technical debt items because the prompt biased toward engineering concerns over business impact. The synthesiser produced a more balanced recommendation as a result.

**The cost:** 3x the latency, 3x the tokens. Worth it when the problem has genuine tension between perspectives.

**When to use:** Problems that benefit from adversarial review. Compliance-sensitive domains where you need an audit trail of reasoning. Anywhere a second opinion adds value.

---

## Paradigm 3: Swarm (Stigmergic) 🐝

**No coordinator. No message passing. Just a shared board.**

This is the paradigm that surprises people. Inspired by ant colony optimisation:

- 5 agents, running independently
- Each reads a shared "pheromone board" (SQLite database) of hypotheses
- Each agent either proposes a NEW hypothesis, REINFORCES an existing one, or CONTRADICTS one
- Repeat for several generations

There is no coordinator. No agent knows what the others are doing. Consensus emerges from indirect communication through the shared state — exactly how ant colonies find optimal paths without central planning.

### The Entropy Story

The key metric is **Shannon entropy** of the hypothesis score distribution:

- **High entropy** = many competing hypotheses, no consensus
- **Low entropy** = convergence on dominant hypotheses

Watching the entropy curve collapse in real time is the most visually compelling part of the demo. In a typical run:

- Generation 0: entropy ~2.1 (agents exploring, many new hypotheses)
- Generation 2: entropy ~1.4 (strong hypotheses getting reinforced)
- Generation 4: entropy ~0.8 (near-consensus)

The swarm doesn't just converge — it converges *on good answers*, because agents that reinforce bad hypotheses see those hypotheses get contradicted by later agents.

**Result:** The swarm produced the most diverse initial exploration and often surfaced edge cases that the pipeline missed. Final quality was competitive with the multi-agent pipeline but with a very different reasoning path.

**When to use:** Brainstorming and ideation. Problems where you want diverse perspectives. Situations where the "right answer" isn't obvious and you want to explore the solution space.

---

## Paradigm 4: ADAS — Automated Design of Agentic Systems 🧬

**A meta-agent that designs the agent system before running it.**

This is the most meta paradigm. It works in two phases:

1. **Design phase** — a meta-agent reads the problem and outputs an architecture specification: how many agents, what roles, what topology (pipeline, parallel, or hybrid)
2. **Execution phase** — the runtime instantiates the designed system and runs it

The "wow moment" is seeing what the meta-agent designs. Given our prioritisation problem, it typically creates something like:

```json
{
  "topology": "pipeline",
  "agents": [
    {"name": "stakeholder_analyst", "role": "Maps each item to business stakeholders"},
    {"name": "technical_assessor", "role": "Evaluates implementation complexity"},
    {"name": "risk_evaluator", "role": "Identifies cross-cutting risks"},
    {"name": "priority_synthesiser", "role": "Produces final ranked recommendation"}
  ]
}
```

It doesn't just copy the multi-agent pipeline — it *designs* a pipeline tailored to this specific problem. A different problem gets a different architecture.

**Result:** Quality was often the highest, because the meta-agent made structural decisions that a human designer might take hours to consider. But it added significant latency (the design call itself uses substantial tokens) and the architecture occasionally over-engineered simple problems.

**When to use:** Novel problems where you don't know the optimal agent structure. R&D exploration. Situations where the meta-cost is justified by the improved output quality.

---

## The Comparison

After running all four paradigms with the same input, here's the pattern we consistently observed:

| Paradigm | Quality | Latency | Tokens | Best For |
|----------|---------|---------|--------|----------|
| Single Agent | ★★★☆ | Fastest | Lowest | Known problems, quick drafts |
| Multi-Agent | ★★★★ | 3x base | 3x base | Review-sensitive, audit trails |
| Swarm | ★★★★ | 5x base | 5x base | Brainstorming, diverse exploration |
| ADAS | ★★★★★ | 4x base | 4x base | Novel problems, max quality |

The single agent is *never terrible*. The multi-agent pipeline is the most predictable improvement. The swarm is the most creative. ADAS is the most adaptive.

---

## What We Learned

### 1. Start Simple, Prove You Need Complexity

The single agent baseline is your null hypothesis. If you can't show measurable improvement from a multi-agent approach, you're adding cost and latency for nothing. We found that for ~60% of straightforward tasks, the single agent was sufficient.

### 2. The Critic Is the Most Valuable Agent

In the multi-agent pipeline, the critic consistently added the most value. Analysts tend to confirm their own assumptions. Critics surface blind spots. If you only have budget for two agents, make the second one adversarial.

### 3. Swarm Entropy Is a Genuinely Useful Signal

The entropy curve isn't just pretty — it tells you *how confident the system is*. High final entropy means the swarm couldn't converge, which is itself valuable information. Low entropy with high scores means strong consensus. We started using entropy as a confidence metric in production.

### 4. ADAS Is Overpowered for Simple Problems

The meta-agent adds real value on novel or complex problems. For routine tasks, it typically designs something very similar to our hand-crafted multi-agent pipeline — but takes extra tokens to get there. Match the paradigm to the problem complexity.

### 5. Observability Is Non-Negotiable

Without structured logging and tracing, multi-agent systems are black boxes. We log every handoff, every swarm event, every architecture decision. When something goes wrong — and it will — you need to know *where* in the chain it went wrong.

---

## Try It Yourself

The complete implementation is available as an interactive notebook:

- **[Agentic Paradigms Notebook](agents_paradigms.ipynb)** — each paradigm on a *different* task designed to highlight its strengths, with baseline comparisons and LLM-as-judge scoring

Each paradigm gets the task where it genuinely shines: ticket classification (single agent), architecture review (multi-agent), investment calculations (tool-use), product naming (swarm), and market entry planning (ADAS). The eval proves the difference quantitatively.

Requirements: Python 3.11+, an OpenAI-compatible LLM endpoint (local or remote). The notebook uses Qwen 3.5 via llama.cpp, but any model works.

For the full demo with Rich terminal UI, run:

```bash
python demo.py --all
```

---

## Bottom Line

There's no universally best agentic architecture. There's the right architecture for your problem, your latency budget, and your quality requirements.

Start with a single agent. Measure it. Add complexity only when the measurements justify it. And whatever you build — make it observable.

The future of AI engineering isn't about which paradigm wins. It's about knowing which one to reach for, and having the evaluation infrastructure to prove you chose correctly.

---

*Alex Chilton is a software engineer working at the intersection of AI systems design and practical engineering. The notebooks and demo code referenced in this post are available on [GitHub](https://github.com/alexchilton).*
