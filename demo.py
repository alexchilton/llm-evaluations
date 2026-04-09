#!/usr/bin/env python3
"""
=============================================================================
AGENTIC PARADIGMS DEMO — Main Entrypoint
=============================================================================
Demonstrates four agentic paradigms solving the same problem side-by-side:

  1. Single Agent    — one LLM call, baseline
  2. Multi-Agent     — Analyst → Critic → Synthesiser pipeline
  3. Swarm           — stigmergic communication via shared pheromone board
  4. ADAS            — meta-agent designs the architecture, then runs it

Usage:
  python demo.py              # run all paradigms sequentially
  python demo.py --paradigm single   # run just one
  python demo.py --paradigm swarm --swarm-agents 12 --swarm-generations 8

All output goes to a Rich terminal UI with live panels, progress bars,
and a final comparison table. Observability data goes to Langfuse (if
running) and always to a local JSON log file.
=============================================================================
"""

import os
import sys
import time
import argparse
from pathlib import Path
from dataclasses import asdict

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.markdown import Markdown
from rich.rule import Rule
from rich import box

# Load environment variables before any other imports that might need them
load_dotenv()

# Local imports
from observability.logger import ObservabilityLogger
from observability.langfuse_client import get_langfuse_client
from eval.judge import score as judge_score
from paradigms import single_agent, multi_agent, swarm, adas

console = Console()

# ---------------------------------------------------------------------------
# Default backlog — the problem all paradigms solve
# ---------------------------------------------------------------------------

DEFAULT_BACKLOG = [
    "Migrate authentication service from monolith to microservice (reduces deploy coupling)",
    "Build real-time analytics dashboard for customer engagement metrics",
    "Upgrade PostgreSQL from v12 to v16 (EOL risk, missing partitioning features)",
    "Implement feature flags system to enable trunk-based development",
    "Reduce API p99 latency from 800ms to 200ms (top customer complaint)",
    "Add comprehensive E2E test suite (currently 12% coverage, blocking CI/CD)",
    "Build self-service data export tool (eliminates 40 hrs/month of manual work)",
    "Refactor billing module to support multi-currency (EU expansion blocked on this)",
]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file with sensible defaults.

    Why YAML over JSON or TOML?
      - Human-readable and editable with comments
      - Standard in ML/AI tooling (HuggingFace, MLflow, etc.)
      - Python's PyYAML is ubiquitous
    """
    defaults = {
        "model": {
            "primary": "unsloth/Qwen3.5-35B-A3B",
            "judge": "unsloth/Qwen3.5-35B-A3B",
            "llm_base": "http://127.0.0.1:8001/v1",
        },
        "swarm": {
            "num_agents": 8,
            "num_generations": 6,
            "convergence_threshold": 0.8,
            "top_k_context": 5,
        },
        "multi_agent": {
            "enable_handoff_eval": True,
        },
        "adas": {
            "max_agents": 5,
        },
        "observability": {
            "langfuse_host": "http://localhost:3000",
            "swarm_db_path": "./swarm_state.db",
            "log_level": "INFO",
        },
    }

    if Path(config_path).exists():
        with open(config_path) as f:
            file_config = yaml.safe_load(f) or {}
        # Merge file config over defaults (shallow per top-level key)
        for key in defaults:
            if key in file_config:
                defaults[key].update(file_config[key])

    return defaults


# ---------------------------------------------------------------------------
# Model availability check
# ---------------------------------------------------------------------------

def check_model_available(api_base: str, model: str) -> tuple[bool, str]:
    """
    Check if the configured model is available and find a fallback if not.

    Returns (available, model_to_use). If the primary model isn't available,
    we try to find any model that is — the demo should always be runnable.
    """
    from openai import OpenAI

    try:
        client = OpenAI(base_url=api_base, api_key="not-needed")
        models = client.models.list()
        available_ids = [m.id for m in models.data]

        if model in available_ids:
            return True, model

        # Try the first available model as fallback
        if available_ids:
            fallback = available_ids[0]
            console.print(
                f"[yellow]⚠ Model '{model}' not found. "
                f"Using fallback: '{fallback}'[/yellow]"
            )
            return True, fallback

        return False, model

    except Exception as e:
        console.print(f"[red]✗ Cannot connect to LLM at {api_base}: {e}[/red]")
        return False, model


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def print_header(backlog: list[str]):
    """Print the task description and backlog items."""
    console.print()
    console.print(Panel(
        "[bold cyan]Agentic Paradigms Demo[/bold cyan]\n\n"
        "[dim]Four paradigms solve the same problem. Watch them work, then compare.[/dim]",
        box=box.DOUBLE,
        border_style="cyan",
    ))

    table = Table(
        title="📋 Backlog Items",
        box=box.SIMPLE_HEAVY,
        show_lines=True,
        title_style="bold yellow",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Item", style="white")

    for i, item in enumerate(backlog, 1):
        table.add_row(str(i), item)

    console.print(table)
    console.print()


def print_paradigm_result(result, eval_result=None):
    """Print one paradigm's result in a styled panel."""
    paradigm_colours = {
        "single": "green",
        "multi": "blue",
        "swarm": "magenta",
        "adas": "yellow",
    }
    paradigm_names = {
        "single": "🎯 Paradigm 1: Single Agent",
        "multi": "🔗 Paradigm 2: Multi-Agent Pipeline",
        "swarm": "🐝 Paradigm 3: Swarm (Stigmergic)",
        "adas": "🧬 Paradigm 4: ADAS (Auto-Designed)",
    }

    colour = paradigm_colours.get(result.paradigm, "white")
    name = paradigm_names.get(result.paradigm, result.paradigm)

    # Build stats line
    stats = (
        f"⏱ {result.latency_ms/1000:.1f}s  "
        f"📊 {result.prompt_tokens + result.completion_tokens} tokens  "
        f"🔑 {result.run_id}"
    )

    # Truncate output for display (full output is in logs)
    display_output = result.output
    if len(display_output) > 2000:
        display_output = display_output[:2000] + "\n\n[dim]... (truncated, see logs for full output)[/dim]"

    content = f"[dim]{stats}[/dim]\n\n{display_output}"

    if eval_result:
        scores_line = (
            f"\n\n[bold]Eval Scores:[/bold] "
            f"clarity={eval_result.clarity:.2f}  "
            f"reasoning={eval_result.reasoning:.2f}  "
            f"completeness={eval_result.completeness:.2f}  "
            f"[bold]overall={eval_result.overall:.2f}[/bold]"
        )
        content += scores_line

    console.print(Panel(
        content,
        title=f"[bold {colour}]{name}[/bold {colour}]",
        border_style=colour,
        box=box.ROUNDED,
    ))


def print_swarm_entropy(entropy_history: list[tuple[int, float]]):
    """Print the swarm entropy curve as an ASCII sparkline table."""
    if not entropy_history:
        return

    table = Table(
        title="🐝 Swarm Entropy Curve",
        box=box.SIMPLE_HEAVY,
        title_style="bold magenta",
    )
    table.add_column("Gen", style="dim", justify="center", width=5)
    table.add_column("Entropy", justify="right", width=8)
    table.add_column("Visualisation", width=40)

    max_entropy = max(e for _, e in entropy_history) if entropy_history else 1.0
    if max_entropy == 0:
        max_entropy = 1.0

    for gen, entropy in entropy_history:
        bar_length = int((entropy / max_entropy) * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        table.add_row(str(gen), f"{entropy:.3f}", f"[magenta]{bar}[/magenta]")

    console.print(table)


def print_adas_architecture(spec: dict):
    """Print the ADAS-designed architecture in a readable format."""
    console.print(Panel(
        f"[bold]Reasoning:[/bold] {spec.get('reasoning', 'N/A')}\n\n"
        f"[bold]Topology:[/bold] {spec.get('topology', 'N/A')}\n\n"
        + "\n".join(
            f"  [cyan]{a.get('name', '?')}[/cyan]: {a.get('role', '?')}"
            for a in spec.get("agents", [])
        ),
        title="[bold yellow]🧬 ADAS-Designed Architecture[/bold yellow]",
        border_style="yellow",
        box=box.ROUNDED,
    ))


def print_comparison_table(results: list[tuple], backlog: list[str]):
    """Print the final side-by-side comparison table."""
    table = Table(
        title="📊 Paradigm Comparison",
        box=box.DOUBLE_EDGE,
        show_lines=True,
        title_style="bold white",
    )
    table.add_column("Paradigm", style="bold", width=20)
    table.add_column("Quality", justify="center", width=10)
    table.add_column("Latency", justify="right", width=12)
    table.add_column("Tokens", justify="right", width=10)
    table.add_column("Est. Cost", justify="right", width=10)

    paradigm_names = {
        "single": "🎯 Single",
        "multi": "🔗 Multi-Agent",
        "swarm": "🐝 Swarm",
        "adas": "🧬 ADAS",
    }

    best_quality = 0.0
    best_paradigm = ""

    for result, eval_result in results:
        name = paradigm_names.get(result.paradigm, result.paradigm)
        quality = f"{eval_result.overall:.2f}" if eval_result else "N/A"
        latency = f"{result.latency_ms/1000:.1f}s"
        tokens = str(result.prompt_tokens + result.completion_tokens)
        # Rough cost estimate (local = free, but simulate cloud pricing)
        total_tokens = result.prompt_tokens + result.completion_tokens
        est_cost = f"${total_tokens * 0.00001:.4f}"

        if eval_result and eval_result.overall > best_quality:
            best_quality = eval_result.overall
            best_paradigm = result.paradigm

        table.add_row(name, quality, latency, tokens, est_cost)

    console.print()
    console.print(table)

    # Verdict
    if best_paradigm:
        verdict_name = paradigm_names.get(best_paradigm, best_paradigm)
        console.print(Panel(
            f"[bold]For this task, {verdict_name} produced the best quality/cost tradeoff "
            f"(score: {best_quality:.2f})[/bold]",
            title="🏆 Verdict",
            border_style="green",
            box=box.HEAVY,
        ))


# ---------------------------------------------------------------------------
# Paradigm runners with progress display
# ---------------------------------------------------------------------------

def run_paradigm_with_progress(
    name: str,
    run_fn,
    backlog: list[str],
    config: dict,
    obs: ObservabilityLogger,
    **extra_kwargs,
):
    """Run a single paradigm with a spinner progress indicator."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Running {name}...", total=1)

        result = run_fn(
            backlog=backlog,
            model=config["model"]["primary"],
            api_base=config["model"]["llm_base"],
            obs=obs,
            **extra_kwargs,
        )

        progress.update(task, completed=1)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Agentic Paradigms Demo — four approaches, one problem",
    )
    parser.add_argument(
        "--paradigm", "-p",
        choices=["single", "multi", "swarm", "adas", "all"],
        default="all",
        help="Which paradigm to run (default: all)",
    )
    parser.add_argument(
        "--swarm-agents", type=int, default=None,
        help="Override number of swarm agents",
    )
    parser.add_argument(
        "--swarm-generations", type=int, default=None,
        help="Override number of swarm generations",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Skip LLM-as-judge evaluation",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Check model availability
    console.print("[dim]Checking LLM availability...[/dim]")
    available, model = check_model_available(
        config["model"]["llm_base"],
        config["model"]["primary"],
    )
    if not available:
        console.print("[red]✗ No LLM available. Start your LLM server and try again.[/red]")
        sys.exit(1)

    config["model"]["primary"] = model
    config["model"]["judge"] = model
    console.print(f"[green]✓ Using model: {model}[/green]")

    # Initialise observability
    lf_client = get_langfuse_client(config["observability"]["langfuse_host"])
    obs = ObservabilityLogger(langfuse_client=lf_client)

    backlog = DEFAULT_BACKLOG
    print_header(backlog)

    # Determine which paradigms to run
    paradigms_to_run = (
        ["single", "multi", "swarm", "adas"]
        if args.paradigm == "all"
        else [args.paradigm]
    )

    all_results = []  # list of (ParadigmResult, EvalResult)

    for paradigm_name in paradigms_to_run:
        console.print(Rule(f"[bold]Running: {paradigm_name.upper()}[/bold]"))

        try:
            if paradigm_name == "single":
                result = run_paradigm_with_progress(
                    "Single Agent", single_agent.run,
                    backlog, config, obs,
                )

            elif paradigm_name == "multi":
                result = run_paradigm_with_progress(
                    "Multi-Agent Pipeline", multi_agent.run,
                    backlog, config, obs,
                )

            elif paradigm_name == "swarm":
                swarm_config = config["swarm"]
                num_agents = args.swarm_agents or swarm_config["num_agents"]
                num_gens = args.swarm_generations or swarm_config["num_generations"]

                # Callback for live entropy display
                entropy_data = []

                def on_gen_complete(gen, entropy, hypotheses):
                    entropy_data.append((gen, entropy))
                    top = hypotheses[:3] if hypotheses else []
                    top_text = " | ".join(
                        f"[{h.score:.1f}] {h.text[:50]}" for h in top
                    )
                    console.print(
                        f"  [magenta]Gen {gen}[/magenta] "
                        f"entropy={entropy:.3f}  "
                        f"top: {top_text}"
                    )

                result = run_paradigm_with_progress(
                    "Swarm", swarm.run,
                    backlog, config, obs,
                    num_agents=num_agents,
                    num_generations=num_gens,
                    convergence_threshold=swarm_config["convergence_threshold"],
                    top_k=swarm_config["top_k_context"],
                    db_path=config["observability"]["swarm_db_path"],
                    on_generation_complete=on_gen_complete,
                )

                # Show entropy curve
                entropy_history = result.metadata.get("entropy_history", [])
                if entropy_history:
                    print_swarm_entropy(entropy_history)

            elif paradigm_name == "adas":
                def on_arch_designed(spec):
                    print_adas_architecture(spec)

                result = run_paradigm_with_progress(
                    "ADAS", adas.run,
                    backlog, config, obs,
                    max_agents=config["adas"]["max_agents"],
                    on_architecture_designed=on_arch_designed,
                )

            # Evaluate the result
            eval_result = None
            if not args.no_eval:
                console.print(f"  [dim]Evaluating {paradigm_name} output...[/dim]")
                try:
                    eval_result = judge_score(
                        backlog=backlog,
                        output=result.output,
                        model=config["model"]["judge"],
                        api_base=config["model"]["llm_base"],
                    )
                    obs.score(
                        result.run_id,
                        {
                            "clarity": eval_result.clarity,
                            "reasoning": eval_result.reasoning,
                            "completeness": eval_result.completeness,
                            "overall": eval_result.overall,
                        },
                        reason=f"LLM-as-judge eval for {paradigm_name}",
                    )
                except Exception as e:
                    console.print(f"  [yellow]⚠ Eval failed: {e}[/yellow]")

            print_paradigm_result(result, eval_result)
            all_results.append((result, eval_result))

        except Exception as e:
            console.print(f"[red]✗ {paradigm_name} failed: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")

    # ── Final comparison ─────────────────────────────────────────────────
    if len(all_results) > 1:
        print_comparison_table(all_results, backlog)

    # Langfuse URL hint
    lf_host = config["observability"]["langfuse_host"]
    console.print(
        f"\n[dim]📊 Traces logged to: run_log.jsonl[/dim]"
    )
    if lf_client.available:
        console.print(f"[dim]🔍 Langfuse dashboard: {lf_host}[/dim]")

    console.print("\n[bold green]✓ Demo complete![/bold green]\n")

    # Flush Langfuse
    if lf_client:
        lf_client.flush()


if __name__ == "__main__":
    main()
