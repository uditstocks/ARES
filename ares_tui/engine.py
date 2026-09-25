"""ResearchEngine — runs the ARES LangGraph pipeline for the dashboard.

Threading model
---------------
The compiled ``master_graph`` streams synchronously, so the engine runs it
on a **daemon thread** and pushes typed events (see ``events.py``) to the
UI through a single ``emit`` callback. The UI wraps that callback with
``App.call_from_thread``, so the Textual event loop is never blocked and
never touched from the wrong thread. Being a daemon thread also means
quitting the app never hangs on an in-flight LLM call.

What the engine translates
--------------------------
LangGraph is streamed with three modes at once (langgraph >= 1.0):

    'tasks'    — task start events; used to learn WHICH analyst belongs to
                 each parallel ``conduct_interview`` branch (the task input
                 carries the Analyst object).
    'messages' — token-by-token LLM output with the producing node in the
                 metadata; coalesced into ~60 ms TokenBursts.
    'updates'  — node completion deltas; used for stage transitions,
                 transcript entries, latency measurement and the report.

``print()`` calls inside ARES.py (search/checkpoint logs) are captured by
temporarily swapping ``sys.stdout`` for a parser that turns complete lines
into LogLine events, so engine internals land in the Activity Log instead
of corrupting the TUI.
"""

from __future__ import annotations

import io
import queue
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from . import events as ev

try:
    import psutil
    _PROCESS = psutil.Process()
except Exception:                                    # psutil is optional
    _PROCESS = None

#: Display colors assigned to parallel interviews, in first-seen order.
INTERVIEW_COLORS = ["cyan", "magenta", "yellow", "green", "bright_blue", "bright_red"]

#: Interview sub-graph node -> pipeline stage shown in the UI.
STAGE_FOR_NODE = {
    "ask_question": "questioning",
    "search_context": "retrieving",
    "answer_question": "answering",
    "save_interview": "writing",
    "write_section": "writing",
}

#: Minimum seconds between token flushes / metric snapshots (UI throttle).
_TOKEN_FLUSH_INTERVAL = 0.06
_SNAPSHOT_INTERVAL = 0.25


@dataclass
class EngineConfig:
    """Everything the engine needs to drive one research run."""
    topic: str
    max_analysts: int = 3
    max_turns: int = 3
    thread_id: Optional[str] = None
    output: str = "research_report.md"
    no_feedback: bool = False


def _chunk_text(chunk) -> str:
    """Extract plain text from a message chunk (str or content-parts list)."""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return ""


def _get(payload, key, default=None):
    """Read a key from a dict-or-object payload (defensive across versions)."""
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


class _StdoutRouter(io.TextIOBase):
    """Converts print() lines from ARES.py into LogLine events.

    ARES logs look like ``[Wikipedia] ✅ Found 2 docs for: '...'`` — the
    bracket label picks the category and the emoji picks the level.
    """

    _CATEGORY = {
        "Wikipedia": "retriever",
        "Web": "retriever",
        "Search": "retriever",
        "Checkpoint": "system",
    }

    def __init__(self, emit: Callable[[ev.EngineEvent], None]):
        self._emit = emit
        self._buffer = ""

    def write(self, s: str) -> int:  # noqa: D102 (io API)
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._route(line.strip())
        return len(s)

    def flush(self) -> None:  # noqa: D102 (io API)
        pass

    def _route(self, line: str) -> None:
        if not line:
            return
        level = "info"
        if "✅" in line:
            level = "success"
        elif "⚠️" in line or "⚠" in line:
            level = "warning"
        elif "❌" in line:
            level = "error"

        category, message = "system", line
        match = re.match(r"^\[([^\]]+)\]\s*(.*)$", line)
        if match:
            label, rest = match.group(1), match.group(2)
            category = self._CATEGORY.get(label, "ai")
            message = rest if label in self._CATEGORY else f"{label}: {rest}"
        for mark in ("✅ ", "⚠️ ", "❌ ", "✅", "⚠️", "❌"):
            message = message.replace(mark, "")
        self._emit(ev.LogLine(category, level, message.strip()))


class ResearchEngine:
    """Drives one full ARES research run on a background daemon thread."""

    def __init__(self, config: EngineConfig, emit: Callable[[ev.EngineEvent], None]):
        self.cfg = config
        self._emit_cb = emit
        self._feedback_q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Interview registry: LangGraph namespace -> stable interview id.
        self._iid_by_ns: Dict[str, str] = {}
        self._name_by_iid: Dict[str, str] = {}
        self._analyst_by_task: Dict[str, str] = {}
        self._stage_by_iid: Dict[str, str] = {}
        self._done_interviews: set = set()
        self._interviews_total = 0

        # Token coalescing.
        self._token_buf: Dict[Tuple[str, str], str] = {}
        self._last_flush = 0.0

        # Metrics.
        self._t0 = time.time()
        self._latencies: Dict[str, Dict[str, float]] = {}
        self._answer_history: List[float] = []
        self._stage_start: Dict[Tuple[str, str], float] = {}
        self._tokens = 0
        self._chunks = 0
        self._last_snapshot = 0.0
        self._phase = ""

    # -- public API (called from the UI thread) --------------------------------

    def start(self) -> None:
        """Spawn the engine thread. Returns immediately."""
        self._thread = threading.Thread(
            target=self._run, name="ares-engine", daemon=True
        )
        self._thread.start()

    def submit_feedback(self, feedback: Optional[str]) -> None:
        """Deliver the user's persona feedback (None/'' = accept)."""
        self._feedback_q.put((feedback or "").strip() or None)

    def stop(self) -> None:
        """Ask the engine to wind down (daemon thread dies with the app)."""
        self._stop.set()
        self._feedback_q.put(None)

    # -- internals (engine thread only) -----------------------------------------

    def _emit(self, event: ev.EngineEvent) -> None:
        if self._stop.is_set():
            return
        try:
            self._emit_cb(event)
        except Exception:
            # The app is shutting down; nothing useful left to do.
            self._stop.set()

    def _set_phase(self, phase: str) -> None:
        if phase != self._phase:
            self._phase = phase
            self._emit(ev.PhaseChanged(phase))

    def _run(self) -> None:
        router = _StdoutRouter(self._emit)
        previous_stdout = sys.stdout
        sys.stdout = router
        try:
            self._emit(ev.StatusChanged("connecting"))
            import ARES  # heavy: builds graphs + validates provider credentials
            self._ares = ARES

            thread_id = self.cfg.thread_id or uuid.uuid4().hex[:8]
            thread = {"configurable": {"thread_id": thread_id}}
            model = (getattr(ARES.llm, "model", None)
                     or getattr(ARES.llm, "model_name", None) or "?")
            self._emit(ev.RunInfo(
                provider=ARES.LLM_PROVIDER,
                model=str(model),
                thread_id=thread_id,
                topic=self.cfg.topic,
            ))
            self._emit(ev.LogLine(
                "system", "info",
                f"Run started · thread_id={thread_id} · "
                f"analysts={self.cfg.max_analysts} · turns={self.cfg.max_turns}",
            ))

            self._phase_analysts(ARES.master_graph, thread)
            if not self._stop.is_set():
                self._phase_interviews(ARES.master_graph, thread)
        except Exception as exc:  # noqa: BLE001 — surfaced to the UI
            hint = ""
            try:
                import ARES
                if ARES.is_connection_error(exc):
                    hint = {
                        "ollama": "Is Ollama running? Try `ollama serve`.",
                        "openrouter": "Check OPENROUTER_API_KEY and your network connection.",
                    }.get(ARES.LLM_PROVIDER,
                          "Check NVIDIA_API_KEY and your network connection.")
            except Exception:
                hint = "Check your .env configuration (provider credentials)."
            self._emit(ev.StatusChanged("error", str(exc)))
            self._emit(ev.RunError(str(exc), hint))
            self._set_phase("error")
        finally:
            sys.stdout = previous_stdout

    # -- phase 1: personas + human feedback loop ---------------------------------

    def _phase_analysts(self, graph, thread) -> None:
        self._set_phase("analysts")
        first_pass = True
        while not self._stop.is_set():
            stream_input = (
                {
                    "topic": self.cfg.topic,
                    "max_analysts": self.cfg.max_analysts,
                    "max_num_turns": self.cfg.max_turns,
                }
                if first_pass
                else None
            )
            first_pass = False

            self._emit(ev.LogLine("ai", "info", "Generating analyst personas…"))
            latest = None
            for state in graph.stream(stream_input, thread, stream_mode="values"):
                if self._stop.is_set():
                    return
                if state.get("analysts"):
                    latest = state["analysts"]

            self._emit(ev.StatusChanged("connected"))
            if latest:
                self._interviews_total = len(latest)
                self._emit(ev.AnalystsGenerated([
                    {
                        "name": a.name,
                        "role": a.role,
                        "affiliation": a.affiliation,
                        "description": a.description,
                    }
                    for a in latest
                ]))

            if self.cfg.no_feedback:
                feedback = None
                self._emit(ev.LogLine(
                    "system", "info", "--no-feedback: accepting personas."))
            else:
                self._set_phase("feedback")
                self._emit(ev.AwaitingFeedback())
                feedback = self._wait_for_feedback()
                if self._stop.is_set():
                    return

            # Recording the decision (even None) is what un-sticks the
            # interrupted human_feedback node — see docs/ARES.md §10.3.
            graph.update_state(
                thread, {"human_analyst_feedback": feedback}, as_node="human_feedback"
            )
            if not feedback:
                return
            self._set_phase("analysts")
            self._emit(ev.LogLine(
                "system", "info", f"Feedback received — regenerating: “{feedback}”"))

    def _wait_for_feedback(self) -> Optional[str]:
        while not self._stop.is_set():
            try:
                return self._feedback_q.get(timeout=0.25)
            except queue.Empty:
                continue
        return None

    # -- phases 2+3: parallel interviews & report synthesis -----------------------

    def _phase_interviews(self, graph, thread) -> None:
        self._set_phase("interviews")
        self._t0 = time.time()
        stream = graph.stream(
            None, thread,
            stream_mode=["updates", "messages", "tasks"],
            subgraphs=True,
        )
        for namespace, mode, payload in stream:
            if self._stop.is_set():
                return
            if mode == "tasks":
                self._on_task(payload)
            elif mode == "messages":
                self._on_message(namespace, payload)
            elif mode == "updates":
                self._on_update(namespace, payload)
        self._flush_tokens(force=True)
        self._snapshot(force=True)

    # -- 'tasks' mode: map parallel branches to analyst names ----------------------

    def _on_task(self, payload) -> None:
        try:
            if _get(payload, "name") != "conduct_interview":
                return
            task_id = str(_get(payload, "id", "") or "")
            task_input = _get(payload, "input") or {}
            analyst = _get(task_input, "analyst")
            name = getattr(analyst, "name", None)
            if task_id and name:
                self._analyst_by_task[task_id] = str(name)
        except Exception:
            pass  # attribution is cosmetic — never let it kill the run

    def _interview_id(self, namespace) -> Optional[str]:
        """Resolve a LangGraph namespace to a stable interview id, registering
        (and naming) the interview on first sight."""
        if not namespace:
            return None
        key = str(namespace[0])
        if not key.startswith("conduct_interview"):
            return None
        if key not in self._iid_by_ns:
            index = len(self._iid_by_ns) + 1
            iid = f"iv{index}"
            task_id = key.split(":", 1)[1] if ":" in key else ""
            name = self._analyst_by_task.get(task_id, f"Interview {index}")
            color = INTERVIEW_COLORS[(index - 1) % len(INTERVIEW_COLORS)]
            self._iid_by_ns[key] = iid
            self._name_by_iid[iid] = name
            self._interviews_total = max(self._interviews_total, index)
            self._emit(ev.InterviewRegistered(iid, name, color))
        return self._iid_by_ns[key]

    # -- 'messages' mode: live tokens ------------------------------------------------

    def _on_message(self, namespace, payload) -> None:
        try:
            chunk, metadata = payload
        except Exception:
            return
        node = (metadata or {}).get("langgraph_node", "")
        text = _chunk_text(chunk)
        if text:
            self._chunks += 1
        usage = getattr(chunk, "usage_metadata", None) or {}
        if isinstance(usage, dict):
            self._tokens += int(usage.get("total_tokens") or 0)

        iid = self._interview_id(namespace)
        if iid and node in STAGE_FOR_NODE:
            self._mark_stage(iid, node)
            if text and node in ("ask_question", "answer_question"):
                buf_key = (iid, node)
                self._token_buf[buf_key] = self._token_buf.get(buf_key, "") + text
                self._flush_tokens()
        elif not iid and node in ("write_report", "write_introduction", "write_conclusion"):
            self._set_phase("synthesis")

    def _mark_stage(self, iid: str, node: str) -> None:
        stage = STAGE_FOR_NODE[node]
        if self._stage_by_iid.get(iid) != stage:
            self._stage_by_iid[iid] = stage
            self._emit(ev.StageChanged(iid, stage))
        self._stage_start.setdefault((iid, node), time.time())

    def _flush_tokens(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_flush < _TOKEN_FLUSH_INTERVAL:
            return
        self._last_flush = now
        for (iid, node), text in list(self._token_buf.items()):
            if text:
                self._emit(ev.TokenBurst(iid, node, text))
        self._token_buf.clear()

    # -- 'updates' mode: node completions ----------------------------------------------

    def _on_update(self, namespace, payload) -> None:
        if not isinstance(payload, dict):
            return
        for node, value in payload.items():
            if node == "__interrupt__":
                continue
            iid = self._interview_id(namespace)
            if iid:
                self._on_interview_update(iid, node, value)
            else:
                self._on_parent_update(node, value)
        self._snapshot()

    def _on_interview_update(self, iid: str, node: str, value) -> None:
        self._finish_stage(iid, node)
        name = self._name_by_iid.get(iid, iid)
        value = value if isinstance(value, dict) else {}

        if node == "ask_question":
            self._flush_tokens(force=True)
            text = self._last_message_text(value)
            if text:
                self._emit(ev.QuestionComplete(iid, text))
        elif node == "answer_question":
            self._flush_tokens(force=True)
            text = self._last_message_text(value)
            if text:
                self._emit(ev.AnswerComplete(iid, text))
        elif node == "search_context":
            count = len(value.get("sources") or [])
            level = "success" if count else "warning"
            self._emit(ev.LogLine(
                "retriever", level, f"{name}: retrieved {count} source(s)"))
        elif node == "write_section":
            self._done_interviews.add(iid)
            self._stage_by_iid[iid] = "done"
            self._emit(ev.StageChanged(iid, "done"))
            self._emit(ev.LogLine("ai", "success", f"{name}: memo completed"))

    def _on_parent_update(self, node: str, value) -> None:
        value = value if isinstance(value, dict) else {}
        if node in ("write_report", "write_introduction", "write_conclusion"):
            self._set_phase("synthesis")
            self._emit(ev.LogLine(
                "ai", "success", f"{node.replace('_', ' ')} finished"))
        elif node == "finalize_report":
            markdown = value.get("final_report", "") or ""
            path, error = self._save_report(markdown)
            self._set_phase("done")
            self._snapshot(force=True)
            self._emit(ev.ReportReady(markdown, path, error))
            level = "error" if error else "success"
            message = (f"Failed to save report: {error}" if error
                       else f"Report saved to {path}")
            self._emit(ev.LogLine("system", level, message))

    @staticmethod
    def _last_message_text(value: dict) -> str:
        messages = value.get("messages") or []
        if not messages:
            return ""
        return _chunk_text(messages[-1]).strip()

    # -- metrics ----------------------------------------------------------------------

    def _finish_stage(self, iid: str, node: str) -> None:
        started = self._stage_start.pop((iid, node), None)
        if started is None:
            return
        elapsed_ms = (time.time() - started) * 1000.0
        bucket = STAGE_FOR_NODE.get(node, node)
        stats = self._latencies.setdefault(
            bucket, {"last": 0.0, "avg": 0.0, "count": 0})
        stats["count"] += 1
        stats["last"] = elapsed_ms
        stats["avg"] += (elapsed_ms - stats["avg"]) / stats["count"]
        if node == "answer_question":
            self._answer_history.append(elapsed_ms)
            del self._answer_history[:-30]

    def _snapshot(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_snapshot < _SNAPSHOT_INTERVAL:
            return
        self._last_snapshot = now
        memory_mb = 0.0
        if _PROCESS is not None:
            try:
                memory_mb = _PROCESS.memory_info().rss / (1024 * 1024)
            except Exception:
                memory_mb = 0.0
        self._emit(ev.MetricsSnapshot(
            latencies={k: dict(v) for k, v in self._latencies.items()},
            history=list(self._answer_history),
            tokens=self._tokens,
            chunks=self._chunks,
            interviews_done=len(self._done_interviews),
            interviews_total=self._interviews_total,
            memory_mb=memory_mb,
            elapsed_s=now - self._t0,
        ))

    # -- output -----------------------------------------------------------------------

    def _save_report(self, markdown: str) -> Tuple[str, Optional[str]]:
        import os
        path = os.path.abspath(self.cfg.output)
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(markdown)
            return path, None
        except OSError as exc:
            return path, str(exc)
