"""The ARES dashboard application: screens, layout, and event dispatch.

Structure
---------
    AresApp            owns config + the finished report; picks the first screen
    SetupScreen        topic / depth entry (skipped when --topic is given)
    DashboardScreen    the live dashboard; owns the ResearchEngine
    AnalystModal       human-in-the-loop persona review (accept / refine)
    ReportScreen       rendered markdown of the final report

All engine events arrive on the UI thread through exactly one funnel
(``DashboardScreen._handle``), which fans them out to dumb widgets.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, DataTable, Footer, Input, Label, Markdown, Static

from . import events as ev
from .engine import EngineConfig, ResearchEngine
from .widgets import (
    ActivityLog,
    MetricsPanel,
    PipelinePanel,
    QuestionPanel,
    ResponsePanel,
    StatusHeader,
    TranscriptPanel,
)

LOGO = """\
 █████╗ ██████╗ ███████╗███████╗
██╔══██╗██╔══██╗██╔════╝██╔════╝
███████║██████╔╝█████╗  ╚█████╗
██╔══██║██╔══██╗██╔══╝   ╚═══██╗
██║  ██║██║  ██║███████╗██████╔╝
╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═════╝"""

#: Sidebar auto-hides below this terminal width (responsive layout).
_SIDEBAR_MIN_WIDTH = 96


class SetupScreen(Screen):
    """Landing screen: collect the topic and research depth."""

    def compose(self) -> ComposeResult:
        with Vertical(id="setup-box"):
            yield Static(LOGO, id="logo")
            yield Static(
                Text("Autonomous Research & Multi-Agent Evaluation Engine",
                     style="italic bright_black"),
                id="tagline",
            )
            yield Input(placeholder="Research topic — press Enter to launch",
                        id="topic-input")
            with Horizontal(id="setup-row"):
                yield Label("Analysts", classes="setup-label")
                yield Input(value="3", type="integer", id="analysts-input",
                            classes="setup-small")
                yield Label("Turns", classes="setup-label")
                yield Input(value="3", type="integer", id="turns-input",
                            classes="setup-small")
                yield Button("Start research ▶", variant="primary", id="start")

    def on_mount(self) -> None:
        self.query_one("#topic-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "topic-input":
            self._launch()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self._launch()

    def _launch(self) -> None:
        app = self.app
        assert isinstance(app, AresApp)
        topic = self.query_one("#topic-input", Input).value.strip()
        if not topic:
            self.notify("Please enter a research topic.", severity="warning")
            return
        app.cfg.topic = topic
        for field, attr in (("#analysts-input", "max_analysts"),
                            ("#turns-input", "max_turns")):
            try:
                value = int(self.query_one(field, Input).value)
                setattr(app.cfg, attr, max(1, value))
            except ValueError:
                pass  # keep the default
        app.switch_screen(DashboardScreen())


class AnalystModal(ModalScreen[Optional[str]]):
    """Persona review checkpoint. Dismisses with feedback text, or None to accept."""

    BINDINGS = [Binding("escape", "accept", "Accept & continue")]

    def __init__(self, analysts: List[Dict[str, str]]) -> None:
        super().__init__()
        self._analysts = analysts

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Static(
                Text(f"🧑‍🔬 {len(self._analysts)} analyst persona(s) generated — "
                     "review before the interviews begin", style="bold"),
                id="modal-title",
            )
            yield DataTable(id="modal-table", cursor_type="row", zebra_stripes=True)
            yield Input(
                placeholder="Feedback to regenerate personas — leave empty and "
                            "press Enter to accept",
                id="modal-feedback",
            )
            with Horizontal(id="modal-actions"):
                yield Button("♻ Regenerate", id="regenerate")
                yield Button("Accept ▶", variant="success", id="accept")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("#", "Name", "Role", "Affiliation", "Focus")
        for index, analyst in enumerate(self._analysts, 1):
            focus = analyst.get("description", "")
            if len(focus) > 58:
                focus = focus[:57] + "…"
            table.add_row(str(index), analyst.get("name", ""),
                          analyst.get("role", ""), analyst.get("affiliation", ""),
                          focus)
        self.query_one("#modal-feedback", Input).focus()

    def _feedback(self) -> Optional[str]:
        return self.query_one("#modal-feedback", Input).value.strip() or None

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(self._feedback())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None if event.button.id == "accept" else self._feedback())

    def action_accept(self) -> None:
        self.dismiss(None)


class ReportScreen(Screen):
    """Full-screen rendered markdown of the finished report."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back to dashboard"),
        Binding("q", "app.quit", "Quit"),
    ]

    def __init__(self, markdown: str, path: str) -> None:
        super().__init__()
        self._markdown = markdown
        self._path = path

    def compose(self) -> ComposeResult:
        yield Static(
            Text(f" 📄 Final report · saved to {self._path}", style="bold green"),
            id="report-path",
        )
        with VerticalScroll(id="report-scroll"):
            yield Markdown(self._markdown)
        yield Footer()


class DashboardScreen(Screen):
    """The live research dashboard. Owns the engine thread."""

    BINDINGS = [
        Binding("l", "toggle_log", "Log"),
        Binding("r", "show_report", "Report"),
        Binding("q", "app.quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield StatusHeader()
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield PipelinePanel()
                yield MetricsPanel()
            with Vertical(id="main"):
                yield QuestionPanel()
                yield ResponsePanel()
                yield TranscriptPanel()
        yield ActivityLog()
        yield Footer()

    # -- lifecycle -----------------------------------------------------------

    def on_mount(self) -> None:
        app = self.app
        assert isinstance(app, AresApp)
        self._registry: Dict[str, Tuple[str, str]] = {}   # iid -> (name, color)
        self._analysts: List[Dict[str, str]] = []
        self._report_shown = False
        self.query_one(StatusHeader).set_run_info("", "", app.cfg.topic)
        # The engine emits from its own daemon thread; call_from_thread hops
        # every event onto the UI thread before any widget is touched.
        self.engine = ResearchEngine(
            app.cfg,
            emit=lambda event: app.call_from_thread(self._handle, event),
        )
        self.engine.start()

    def on_unmount(self) -> None:
        if hasattr(self, "engine"):
            self.engine.stop()

    def on_resize(self, event) -> None:
        """Responsive layout: drop the sidebar on narrow terminals."""
        try:
            self.query_one("#sidebar").display = (
                event.size.width >= _SIDEBAR_MIN_WIDTH)
        except Exception:
            pass

    # -- actions ----------------------------------------------------------------

    def action_toggle_log(self) -> None:
        log = self.query_one(ActivityLog)
        log.display = not log.display

    def action_show_report(self) -> None:
        app = self.app
        assert isinstance(app, AresApp)
        if app.report_markdown:
            app.push_screen(ReportScreen(app.report_markdown, app.report_path))
        else:
            self.notify("The report isn't ready yet.", severity="warning")

    # -- engine event dispatch (single funnel, UI thread) --------------------------

    def _handle(self, event: ev.EngineEvent) -> None:  # noqa: C901 — dispatch table
        app = self.app
        assert isinstance(app, AresApp)
        header = self.query_one(StatusHeader)
        pipeline = self.query_one(PipelinePanel)

        if isinstance(event, ev.RunInfo):
            header.set_run_info(event.provider, event.model, event.topic)

        elif isinstance(event, ev.StatusChanged):
            header.set_status(event.status)

        elif isinstance(event, ev.PhaseChanged):
            header.set_phase(event.phase)
            pipeline.set_phase(event.phase)

        elif isinstance(event, ev.AnalystsGenerated):
            self._analysts = event.analysts
            log = self.query_one(ActivityLog)
            for analyst in event.analysts:
                log.write_event(ev.LogLine(
                    "ai", "success",
                    f"persona: {analyst['name']} — {analyst['role']}"))

        elif isinstance(event, ev.AwaitingFeedback):
            app.push_screen(AnalystModal(self._analysts), self._on_feedback)

        elif isinstance(event, ev.InterviewRegistered):
            self._registry[event.interview_id] = (event.analyst_name, event.color)
            pipeline.register_interview(
                event.interview_id, event.analyst_name, event.color)
            self.query_one(TranscriptPanel).add_system(
                f"interview started · {event.analyst_name}")

        elif isinstance(event, ev.StageChanged):
            pipeline.set_stage(event.interview_id, event.stage)

        elif isinstance(event, ev.TokenBurst):
            name, color = self._registry.get(event.interview_id, ("…", "cyan"))
            if event.node == "ask_question":
                self.query_one(QuestionPanel).stream(
                    event.interview_id, name, color, event.text)
            else:
                self.query_one(ResponsePanel).stream(
                    event.interview_id, name, color, event.text)

        elif isinstance(event, ev.QuestionComplete):
            name, color = self._registry.get(event.interview_id, ("…", "cyan"))
            self.query_one(QuestionPanel).complete(
                event.interview_id, name, color, event.text)
            self.query_one(TranscriptPanel).add_question(name, color, event.text)

        elif isinstance(event, ev.AnswerComplete):
            name, color = self._registry.get(event.interview_id, ("…", "green"))
            self.query_one(ResponsePanel).complete(
                event.interview_id, name, color, event.text)
            self.query_one(TranscriptPanel).add_answer(name, color, event.text)

        elif isinstance(event, ev.LogLine):
            self.query_one(ActivityLog).write_event(event)

        elif isinstance(event, ev.MetricsSnapshot):
            self.query_one(MetricsPanel).update_snapshot(event)
            pipeline.set_progress(event.interviews_done, event.interviews_total)

        elif isinstance(event, ev.ReportReady):
            app.report_markdown = event.markdown
            app.report_path = event.path
            if event.error:
                self.notify(f"Report ready, but saving failed: {event.error}",
                            severity="error", timeout=10)
            else:
                self.notify("Report finalized ✔ — press R any time to view it.",
                            timeout=8)
            if not self._report_shown:
                self._report_shown = True
                app.push_screen(ReportScreen(event.markdown, event.path))

        elif isinstance(event, ev.RunError):
            message = event.message if len(event.message) < 300 else (
                event.message[:299] + "…")
            self.notify(f"Engine error: {message}", severity="error", timeout=15)
            log = self.query_one(ActivityLog)
            log.write_event(ev.LogLine("system", "error", message))
            if event.hint:
                log.write_event(ev.LogLine("system", "warning", event.hint))

    def _on_feedback(self, feedback: Optional[str]) -> None:
        self.query_one(ActivityLog).write_event(ev.LogLine(
            "system", "info",
            f"feedback: “{feedback}”" if feedback else "personas accepted"))
        self.engine.submit_feedback(feedback)


class AresApp(App[None]):
    """ARES — dashboard edition."""

    TITLE = "ARES"
    SUB_TITLE = "Autonomous Research & Multi-Agent Evaluation Engine"
    CSS_PATH = "app.tcss"

    def __init__(self, cfg: EngineConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.report_markdown: str = ""
        self.report_path: str = ""

    def on_mount(self) -> None:
        self.push_screen(
            DashboardScreen() if self.cfg.topic.strip() else SetupScreen())
