"""Typed events flowing from the engine thread to the UI thread.

The engine never touches widgets and the UI never touches LangGraph:
every fact the dashboard displays arrives as one of these frozen-shape
dataclasses. This keeps the two layers independently testable and makes
the threading boundary explicit (events are created on the engine thread
and consumed on the UI thread via ``App.call_from_thread``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EngineEvent:
    """Base class. ``ts`` is stamped on the engine thread at creation."""
    ts: float = field(default_factory=time.time, init=False)


# --- Run lifecycle -----------------------------------------------------------

@dataclass
class RunInfo(EngineEvent):
    """Emitted once, after the ARES engine imports successfully."""
    provider: str
    model: str
    thread_id: str
    topic: str


@dataclass
class StatusChanged(EngineEvent):
    """Backend connection status: 'connecting' | 'connected' | 'error'."""
    status: str
    detail: str = ""


@dataclass
class PhaseChanged(EngineEvent):
    """Top-level run phase:
    'analysts' | 'feedback' | 'interviews' | 'synthesis' | 'done' | 'error'."""
    phase: str


@dataclass
class RunError(EngineEvent):
    """Fatal engine error, with an optional human-friendly hint."""
    message: str
    hint: str = ""


# --- Phase 1: analyst personas ----------------------------------------------

@dataclass
class AnalystsGenerated(EngineEvent):
    """The latest batch of personas (list of plain dicts, UI-safe)."""
    analysts: List[Dict[str, str]]


@dataclass
class AwaitingFeedback(EngineEvent):
    """The graph is paused at the human-in-the-loop checkpoint."""


# --- Phase 2/3: interviews & synthesis ----------------------------------------

@dataclass
class InterviewRegistered(EngineEvent):
    """A parallel interview branch was seen for the first time."""
    interview_id: str
    analyst_name: str
    color: str


@dataclass
class StageChanged(EngineEvent):
    """One interview moved to a new pipeline stage:
    'questioning' | 'retrieving' | 'answering' | 'writing' | 'done'."""
    interview_id: str
    stage: str


@dataclass
class TokenBurst(EngineEvent):
    """A coalesced burst of streamed LLM tokens (~60 ms of output)."""
    interview_id: str
    node: str            # 'ask_question' | 'answer_question'
    text: str


@dataclass
class QuestionComplete(EngineEvent):
    interview_id: str
    text: str


@dataclass
class AnswerComplete(EngineEvent):
    interview_id: str
    text: str


# --- Telemetry ----------------------------------------------------------------

@dataclass
class LogLine(EngineEvent):
    """One activity-log entry.
    category: 'system' | 'ai' | 'retriever' | 'graph'
    level:    'info' | 'success' | 'warning' | 'error'
    """
    category: str
    level: str
    message: str


@dataclass
class MetricsSnapshot(EngineEvent):
    """Rolling performance numbers, refreshed on every graph update."""
    latencies: Dict[str, Dict[str, float]]  # stage -> {last, avg, count} (ms)
    history: List[float]                    # recent answer latencies (ms)
    tokens: int                             # provider-reported tokens (0 if n/a)
    chunks: int                             # streamed chunk count (~tokens)
    interviews_done: int
    interviews_total: int
    memory_mb: float
    elapsed_s: float


@dataclass
class ReportReady(EngineEvent):
    """The final report was assembled and saved to disk."""
    markdown: str
    path: str
    error: Optional[str] = None             # set if saving to disk failed
