"""
ares_tui — a modern Textual dashboard for the ARES research engine.
====================================================================

This package is a *presentation layer only*: it imports the compiled
LangGraph pipeline from ``ARES.py`` (which stays fully functional as a
classic CLI) and drives it from a polished, non-blocking terminal UI.

Layers
------
    events.py    Typed event dataclasses — the one-way contract between
                 the engine thread and the UI thread.
    engine.py    ResearchEngine: runs the master graph on a daemon thread
                 and translates LangGraph stream output into events.
    widgets.py   Reusable, dumb UI widgets (they only render state they
                 are given — no business logic).
    app.py       The Textual App: screens, layout, event dispatch.

Run it:
    python -m ares_tui                      # interactive setup screen
    python -m ares_tui --topic "..."        # jump straight to the dashboard
"""

__version__ = "1.0.0"
