"""Reusable dashboard widgets.

Every widget here is deliberately *dumb*: it renders state that the app
pushes into it via typed methods and never talks to the engine, the graph,
or other widgets. That keeps the UI composable and trivially testable.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import RichLog, Sparkline, Static

from . import events as ev

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

STAGE_ICONS = {
    "questioning": "🎤",
    "retrieving": "🔍",
    "answering": "🧠",
    "writing": "📝",
    "done": "✅",
}

PHASE_BADGES = {
    "analysts": ("PERSONAS", "bold black on dodger_blue2"),
    "feedback": ("REVIEW", "bold black on yellow"),
    "interviews": ("INTERVIEWS", "bold black on cyan"),
    "synthesis": ("SYNTHESIS", "bold black on magenta"),
    "done": ("COMPLETE", "bold black on green"),
    "error": ("ERROR", "bold white on red"),
}

STATUS_DOTS = {
    "connecting": ("●", "yellow", "connecting"),
    "connected": ("●", "green", "connected"),
    "error": ("●", "red", "error"),
}

LEVEL_MARKS = {
    "info": ("·", "bright_black"),
    "success": ("✔", "green"),
    "warning": ("▲", "yellow"),
    "error": ("✖", "red"),
}

CATEGORY_BADGES = {
    "system": ("SYS ", "bright_black"),
    "ai": ("AI  ", "cyan"),
    "retriever": ("RETR", "yellow"),
    "graph": ("GRPH", "magenta"),
}


class StatusHeader(Static):
    """Top bar: app mark · topic · phase badge · connection · model · clock."""

    def __init__(self) -> None:
        super().__init__(id="status-header")
        self._topic = ""
        self._phase = "analysts"
        self._status = "connecting"
        self._provider = ""
        self._model = ""

    def on_mount(self) -> None:
        self.set_interval(1.0, self._redraw)
        self._redraw()

    # -- state setters (called by the app) --------------------------------

    def set_run_info(self, provider: str, model: str, topic: str) -> None:
        self._provider, self._model, self._topic = provider, model, topic
        self._redraw()

    def set_phase(self, phase: str) -> None:
        self._phase = phase
        self._redraw()

    def set_status(self, status: str) -> None:
        self._status = status
        self._redraw()

    # -- rendering ----------------------------------------------------------

    def _redraw(self) -> None:
        badge_text, badge_style = PHASE_BADGES.get(
            self._phase, (self._phase.upper(), "bold"))
        dot, dot_style, dot_label = STATUS_DOTS.get(
            self._status, ("●", "bright_black", self._status))
        parts = Text()
        parts.append(" ⬢ ARES ", style="bold cyan")
        parts.append("│ ", style="bright_black")
        if self._topic:
            topic = self._topic if len(self._topic) <= 40 else self._topic[:39] + "…"
            parts.append(f"{topic} ", style="italic")
            parts.append("│ ", style="bright_black")
        parts.append(f" {badge_text} ", style=badge_style)
        parts.append(" │ ", style="bright_black")
        parts.append(dot, style=dot_style)
        parts.append(f" {dot_label} ", style="bright_black")
        if self._model:
            parts.append("│ ", style="bright_black")
            parts.append(f"{self._provider}·{self._model} ", style="bright_black")
        parts.append("│ ", style="bright_black")
        parts.append(time.strftime("%H:%M:%S"), style="bright_black")
        self.update(parts)


class PipelinePanel(Static):
    """Left column: global phase checklist + one live row per interview."""

    _GLOBAL_STAGES = [
        ("🧑‍🔬", "Personas"),
        ("🎙", "Interviews"),
        ("🧵", "Synthesis"),
        ("📊", "Report"),
    ]
    _PHASE_RANK = {"analysts": 0, "feedback": 0, "interviews": 1,
                   "synthesis": 2, "done": 4, "error": -1}

    def __init__(self) -> None:
        super().__init__(id="pipeline")
        self.border_title = "AGENT PIPELINE"
        self._phase = "analysts"
        self._frame = 0
        self._interviews: Dict[str, Tuple[str, str, str]] = {}  # iid -> (name, color, stage)
        self._done = 0
        self._total = 0

    def on_mount(self) -> None:
        self.set_interval(0.12, self._tick)
        self._redraw()

    def _tick(self) -> None:
        if self._phase not in ("done", "error"):
            self._frame = (self._frame + 1) % len(SPINNER_FRAMES)
            self._redraw()

    # -- state setters ---------------------------------------------------------

    def set_phase(self, phase: str) -> None:
        self._phase = phase
        self._redraw()

    def register_interview(self, iid: str, name: str, color: str) -> None:
        self._interviews[iid] = (name, color, "questioning")
        self._redraw()

    def set_stage(self, iid: str, stage: str) -> None:
        if iid in self._interviews:
            name, color, _ = self._interviews[iid]
            self._interviews[iid] = (name, color, stage)
            self._redraw()

    def set_progress(self, done: int, total: int) -> None:
        self._done, self._total = done, total
        self._redraw()

    # -- rendering ----------------------------------------------------------------

    def _redraw(self) -> None:
        spinner = SPINNER_FRAMES[self._frame]
        rank = self._PHASE_RANK.get(self._phase, 0)
        table = Table.grid(padding=(0, 1))
        table.add_column(width=2)
        table.add_column()
        table.add_column(justify="right")

        for index, (icon, label) in enumerate(self._GLOBAL_STAGES):
            if rank == -1:
                mark, style, note = ("✖", "red", "") if index <= 1 else ("○", "bright_black", "")
            elif index < rank or rank >= 4:
                mark, style, note = "✔", "green", "done"
            elif index == rank:
                mark, style = spinner, "cyan"
                note = (f"{self._done}/{self._total}"
                        if index == 1 and self._total else "active")
            else:
                mark, style, note = "○", "bright_black", ""
            table.add_row(
                Text(mark, style=style),
                Text(f"{icon} {label}", style="bold" if index == rank else "bright_black"),
                Text(note, style="bright_black"),
            )

        if self._interviews:
            table.add_row("", Text("─" * 20, style="bright_black"), "")
            for name, color, stage in self._interviews.values():
                icon = STAGE_ICONS.get(stage, "·")
                active = stage != "done"
                mark = Text(spinner if active else "✔",
                            style=color if active else "green")
                display = name if len(name) <= 16 else name[:15] + "…"
                table.add_row(
                    mark,
                    Text(display, style=f"bold {color}"),
                    Text(f"{icon} {stage}", style="bright_black"),
                )
        self.update(table)


class QuestionPanel(Static):
    """Prominent display of the question currently being asked (streams live)."""

    def __init__(self) -> None:
        super().__init__(id="question")
        self.border_title = "🎤 CURRENT QUESTION"
        self._iid: Optional[str] = None
        self._name = ""
        self._color = "cyan"
        self._text = ""

    def stream(self, iid: str, name: str, color: str, text: str) -> None:
        """Append streamed tokens; switches focus if a new interview speaks."""
        if iid != self._iid:
            self._iid, self._name, self._color, self._text = iid, name, color, ""
        self._text += text
        self._redraw()

    def complete(self, iid: str, name: str, color: str, text: str) -> None:
        self._iid, self._name, self._color, self._text = iid, name, color, text
        self._redraw()

    def _redraw(self) -> None:
        body = Text()
        if self._name:
            body.append(f"{self._name}  ", style=f"bold {self._color}")
        body.append(self._text or "Waiting for the first question…",
                    style="" if self._text else "italic bright_black")
        self.update(body)


class ResponsePanel(VerticalScroll):
    """Expert answer streamed token-by-token with a blinking cursor."""

    def __init__(self) -> None:
        super().__init__(id="response")
        self.border_title = "💬 EXPERT ANSWER"
        self._body = Static(id="response-body")
        self._iid: Optional[str] = None
        self._name = ""
        self._color = "green"
        self._text = ""
        self._streaming = False
        self._cursor_on = True

    def compose(self) -> ComposeResult:
        yield self._body

    def on_mount(self) -> None:
        self.set_interval(0.5, self._blink)
        self._redraw()

    def _blink(self) -> None:
        if self._streaming:
            self._cursor_on = not self._cursor_on
            self._redraw()

    def stream(self, iid: str, name: str, color: str, text: str) -> None:
        if iid != self._iid:
            self._iid, self._name, self._color, self._text = iid, name, color, ""
        self._streaming = True
        self._text += text
        self._redraw()
        self.scroll_end(animate=False)

    def complete(self, iid: str, name: str, color: str, text: str) -> None:
        self._iid, self._name, self._color, self._text = iid, name, color, text
        self._streaming = False
        self._redraw()
        self.scroll_end(animate=False)

    def _redraw(self) -> None:
        body = Text()
        if self._name:
            body.append(f"answering {self._name}\n", style=f"italic {self._color}")
        if self._text:
            body.append(self._text)
        elif not self._streaming:
            body.append("The expert's grounded, cited answer streams here.",
                        style="italic bright_black")
        if self._streaming and self._cursor_on:
            body.append("▊", style="bold cyan")
        self._body.update(body)


class TranscriptPanel(RichLog):
    """Scrolling history of every completed Q&A across all interviews."""

    def __init__(self) -> None:
        super().__init__(id="transcript", wrap=True, auto_scroll=True,
                         max_lines=2000)
        self.border_title = "LIVE TRANSCRIPT"

    def add_question(self, name: str, color: str, text: str) -> None:
        entry = Text()
        entry.append(f"🎤 {name}  ", style=f"bold {color}")
        entry.append(text)
        self.write(entry)

    def add_answer(self, name: str, color: str, text: str) -> None:
        entry = Text()
        entry.append("   💬 ", style=color)
        entry.append(text, style="bright_black")
        self.write(entry)
        self.write(Text(""))

    def add_system(self, text: str) -> None:
        self.write(Text(f"── {text} ──", style="italic bright_black"))


class ActivityLog(RichLog):
    """Compact, categorised event timeline (docked at the bottom)."""

    def __init__(self) -> None:
        super().__init__(id="activity", wrap=False, auto_scroll=True,
                         max_lines=500)
        self.border_title = "ACTIVITY"

    def write_event(self, event: ev.LogLine) -> None:
        mark, mark_style = LEVEL_MARKS.get(event.level, ("·", "bright_black"))
        badge, badge_style = CATEGORY_BADGES.get(
            event.category, ("MISC", "bright_black"))
        line = Text()
        line.append(time.strftime("%H:%M:%S", time.localtime(event.ts)) + " ",
                    style="bright_black")
        line.append(f"{badge} ", style=badge_style)
        line.append(f"{mark} ", style=mark_style)
        line.append(event.message,
                    style="red" if event.level == "error" else "")
        self.write(line)


class MetricsPanel(Widget):
    """Latency table + token/interview counters + answer-latency sparkline."""

    def __init__(self) -> None:
        super().__init__(id="metrics")
        self.border_title = "PERFORMANCE"
        self._table = Static(id="metrics-table")
        self._spark = Sparkline([0.0], summary_function=max, id="metrics-spark")
        self._spark_label = Static(
            Text("answer latency trend", style="italic bright_black"),
            id="metrics-spark-label",
        )

    def compose(self) -> ComposeResult:
        yield self._table
        yield self._spark_label
        yield self._spark

    def on_mount(self) -> None:
        self.update_snapshot(None)

    @staticmethod
    def _fmt_ms(value: float) -> str:
        return f"{value / 1000:.1f}s" if value >= 1000 else f"{value:.0f}ms"

    def update_snapshot(self, snap: Optional[ev.MetricsSnapshot]) -> None:
        table = Table.grid(padding=(0, 1))
        table.add_column()
        table.add_column(justify="right")
        table.add_column(justify="right", style="bright_black")

        rows: List[Tuple[str, str]] = [
            ("Question gen", "questioning"),
            ("Retrieval", "retrieving"),
            ("LLM answer", "answering"),
            ("Section write", "writing"),
        ]
        table.add_row(Text("stage", style="bright_black"),
                      Text("last", style="bright_black"),
                      Text("avg", style="bright_black"))
        for label, key in rows:
            stats = (snap.latencies.get(key) if snap else None) or {}
            last = self._fmt_ms(stats["last"]) if stats else "—"
            avg = self._fmt_ms(stats["avg"]) if stats else "—"
            table.add_row(label, Text(last, style="cyan"), avg)

        table.add_row("", "", "")
        tokens = (f"{snap.tokens:,}" if snap and snap.tokens
                  else f"~{snap.chunks:,}" if snap else "—")
        table.add_row("Tokens", Text(tokens, style="magenta"), "")
        interviews = (f"{snap.interviews_done}/{snap.interviews_total}"
                      if snap and snap.interviews_total else "—")
        table.add_row("Interviews", Text(interviews, style="cyan"), "")
        memory = f"{snap.memory_mb:.0f} MB" if snap and snap.memory_mb else "—"
        table.add_row("Memory", memory, "")
        elapsed = (time.strftime("%M:%S", time.gmtime(snap.elapsed_s))
                   if snap else "—")
        table.add_row("Elapsed", elapsed, "")

        self._table.update(table)
        if snap and snap.history:
            self._spark.data = snap.history
