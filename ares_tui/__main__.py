"""Entry point: ``python -m ares_tui``.

Accepts the same flags (and ARES_* environment fallbacks) as the classic
CLI, so the two front-ends are interchangeable:

    python -m ares_tui                                   # setup screen
    python -m ares_tui --topic "..." --no-feedback       # straight to work
"""

from __future__ import annotations

import argparse
import os

from .app import AresApp
from .engine import EngineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ares_tui",
        description="ARES — dashboard edition (Textual TUI).",
    )
    parser.add_argument("--topic", default=os.getenv("ARES_TOPIC", ""),
                        help="Research topic (setup screen shown if omitted).")
    parser.add_argument("--max-analysts", type=int,
                        default=int(os.getenv("ARES_MAX_ANALYSTS", "3")),
                        help="Number of analyst personas (default: 3).")
    parser.add_argument("--max-turns", type=int,
                        default=int(os.getenv("ARES_MAX_TURNS", "3")),
                        help="Q&A turns per interview (default: 3).")
    parser.add_argument("--thread-id", default=os.getenv("ARES_THREAD_ID"),
                        help="Checkpoint thread id (default: random per run).")
    parser.add_argument("--output",
                        default=os.getenv("ARES_OUTPUT", "research_report.md"),
                        help="Report path (default: research_report.md).")
    parser.add_argument("--no-feedback", action="store_true",
                        help="Skip the persona review checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EngineConfig(
        topic=(args.topic or "").strip(),
        max_analysts=max(1, args.max_analysts),
        max_turns=max(1, args.max_turns),
        thread_id=args.thread_id,
        output=args.output,
        no_feedback=args.no_feedback,
    )
    AresApp(cfg).run()


if __name__ == "__main__":
    main()
