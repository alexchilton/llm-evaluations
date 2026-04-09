"""
=============================================================================
SWARM PHEROMONE BOARD — SQLite State Log
=============================================================================
The swarm paradigm uses stigmergic communication: agents don't talk to each
other directly. Instead, they read from and write to a shared "pheromone board"
(a SQLite database). This is inspired by ant colony optimisation.

Two tables:
  - hypotheses: the shared belief state (what the swarm thinks)
  - events:     append-only audit log (what each agent did and when)

Why SQLite?
  - Zero-config, serverless, file-based — perfect for a demo
  - ACID transactions mean agents can read-modify-write atomically
  - The events table gives us a full replay log for observability
  - Board entropy is computed from hypothesis scores at each generation

The entropy metric is key: it measures how "spread out" the swarm's belief
distribution is. High entropy = many competing hypotheses (disagreement).
Low entropy = convergence on a few dominant hypotheses (consensus).
=============================================================================
"""

import sqlite3
import uuid
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Data classes for type-safe board operations
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    """One hypothesis on the pheromone board."""
    id: str
    text: str
    score: float = 1.0
    created_by: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    generation: int = 0


@dataclass
class BoardEvent:
    """One agent action logged to the events table."""
    generation: int
    agent_id: str
    action: str             # 'new' | 'reinforce' | 'contradict'
    hypothesis_id: str
    score_before: float
    score_after: float
    board_entropy: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# SwarmDB — manages the SQLite pheromone board
# ---------------------------------------------------------------------------

class SwarmDB:
    """
    Manages the swarm's shared state via SQLite.

    Each run creates a fresh database (or clears existing tables) to avoid
    cross-run contamination. The events table is append-only — we never
    delete from it, giving us a full audit trail.
    """

    def __init__(self, db_path: str = "./swarm_state.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        """Create tables if they don't exist, clear data for fresh run."""
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hypotheses (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                score REAL DEFAULT 1.0,
                created_by TEXT,
                created_at TIMESTAMP,
                generation INTEGER
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                generation INTEGER,
                agent_id TEXT,
                action TEXT,
                hypothesis_id TEXT,
                score_before REAL,
                score_after REAL,
                board_entropy REAL
            )
        """)

        # Clear for fresh run
        cursor.execute("DELETE FROM hypotheses")
        cursor.execute("DELETE FROM events")
        self.conn.commit()

    def get_top_hypotheses(self, k: int = 5) -> list[Hypothesis]:
        """
        Read the top-k hypotheses by score.

        This is what each swarm agent "sees" — a limited window into the
        board's current state. The k parameter controls how much context
        each agent gets, affecting convergence speed vs diversity.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM hypotheses ORDER BY score DESC LIMIT ?", (k,)
        )
        rows = cursor.fetchall()
        return [
            Hypothesis(
                id=r["id"],
                text=r["text"],
                score=r["score"],
                created_by=r["created_by"] or "",
                created_at=r["created_at"] or "",
                generation=r["generation"] or 0,
            )
            for r in rows
        ]

    def get_all_hypotheses(self) -> list[Hypothesis]:
        """Get all hypotheses, ordered by score descending."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM hypotheses ORDER BY score DESC")
        rows = cursor.fetchall()
        return [
            Hypothesis(
                id=r["id"],
                text=r["text"],
                score=r["score"],
                created_by=r["created_by"] or "",
                created_at=r["created_at"] or "",
                generation=r["generation"] or 0,
            )
            for r in rows
        ]

    def add_hypothesis(self, text: str, agent_id: str, generation: int) -> Hypothesis:
        """
        Add a new hypothesis to the board.
        Returns the created Hypothesis with its generated ID.
        """
        hyp_id = f"hyp-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO hypotheses (id, text, score, created_by, created_at, generation)
               VALUES (?, ?, 1.0, ?, ?, ?)""",
            (hyp_id, text, agent_id, now, generation),
        )
        self.conn.commit()

        return Hypothesis(
            id=hyp_id, text=text, score=1.0,
            created_by=agent_id, created_at=now, generation=generation,
        )

    def reinforce_hypothesis(self, hyp_id: str, boost: float = 0.5) -> tuple[float, float]:
        """
        Increase a hypothesis's score (another agent agrees with it).

        Returns (score_before, score_after) for logging.
        Reinforcement is additive — this creates a positive feedback loop
        similar to ant pheromone trails getting stronger with more ants.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT score FROM hypotheses WHERE id = ?", (hyp_id,))
        row = cursor.fetchone()
        if not row:
            return (0.0, 0.0)

        score_before = row["score"]
        score_after = score_before + boost

        cursor.execute(
            "UPDATE hypotheses SET score = ? WHERE id = ?",
            (score_after, hyp_id),
        )
        self.conn.commit()
        return (score_before, score_after)

    def contradict_hypothesis(self, hyp_id: str, penalty: float = 0.3) -> tuple[float, float]:
        """
        Decrease a hypothesis's score (another agent disagrees).

        Returns (score_before, score_after). Score floors at 0.1 to prevent
        hypotheses from being completely eliminated — even weak ideas might
        get reinforced later. This mirrors pheromone evaporation in ACO.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT score FROM hypotheses WHERE id = ?", (hyp_id,))
        row = cursor.fetchone()
        if not row:
            return (0.0, 0.0)

        score_before = row["score"]
        score_after = max(0.1, score_before - penalty)

        cursor.execute(
            "UPDATE hypotheses SET score = ? WHERE id = ?",
            (score_after, hyp_id),
        )
        self.conn.commit()
        return (score_before, score_after)

    def log_event(self, event: BoardEvent):
        """Append an event to the audit log."""
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO events (timestamp, generation, agent_id, action,
               hypothesis_id, score_before, score_after, board_entropy)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.timestamp, event.generation, event.agent_id,
                event.action, event.hypothesis_id, event.score_before,
                event.score_after, event.board_entropy,
            ),
        )
        self.conn.commit()

    def compute_entropy(self) -> float:
        """
        Compute Shannon entropy of the hypothesis score distribution.

        Entropy = -Σ p_i * log(p_i) where p_i = score_i / Σ scores

        High entropy → many hypotheses with similar scores (disagreement)
        Low entropy  → one or few hypotheses dominate (consensus)

        We use scipy's entropy function which handles edge cases (zero probs).
        """
        hypotheses = self.get_all_hypotheses()
        if not hypotheses:
            return 0.0

        scores = np.array([h.score for h in hypotheses])
        if scores.sum() == 0:
            return 0.0

        # Normalise to probability distribution
        probs = scores / scores.sum()

        # Shannon entropy (base e, natural log)
        # scipy.stats.entropy handles zeros gracefully, but we use numpy
        # to avoid the scipy import just for this one function
        probs = probs[probs > 0]  # remove zeros to avoid log(0)
        return float(-np.sum(probs * np.log(probs)))

    def get_entropy_history(self) -> list[tuple[int, float]]:
        """
        Get entropy value at the end of each generation.
        Returns list of (generation, entropy) tuples.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT generation, board_entropy
            FROM events
            WHERE id IN (
                SELECT MAX(id) FROM events GROUP BY generation
            )
            ORDER BY generation
        """)
        return [(r["generation"], r["board_entropy"]) for r in cursor.fetchall()]

    def close(self):
        """Close the database connection."""
        self.conn.close()
