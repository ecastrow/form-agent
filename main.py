"""CLI entry point for the Intelligent Form Agent.

Usage:
    python main.py [initial_form_path ...]

Any file paths given as startup arguments are ingested before the
first prompt, so you can start straight in with a question about
them. Then interact in a loop:

    <question>          Ask about the ingested form(s)
    /upload <path>       Ingest another form (image or PDF)
    /help                Show available commands
    /quit, /exit         End the session

Ties together build_turn_update() (interface layer -- detects what
happened this turn) and build_graph() (the compiled agent) exactly
as designed: this script's only job is collecting input and
displaying output, never agent reasoning.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Project root on sys.path regardless of invocation method/cwd --
# same fix as scripts/try_extraction.py, see that file's comment
# for the full explanation.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.graph import build_graph  # noqa: E402
from src.turn_input import build_turn_update  # noqa: E402

load_dotenv()

HELP_TEXT = """
Commands:
  <question>            Ask about the ingested form(s)
  /upload <path>         Ingest another form (image or PDF)
  /help                  Show this message
  /quit, /exit           End the session
""".strip()


def _run_turn(graph, config: dict, **kwargs) -> None:
    """Builds one turn's update, invokes the graph, prints the result.

    Broad exception handling is deliberate here specifically -- this
    is the CLI boundary. A single failed API call (rate limit,
    network blip, etc.) should not crash the whole interactive
    session; the user can just try again.
    """
    update = build_turn_update(**kwargs)
    try:
        result = graph.invoke(update, config=config)
    except Exception as exc:  # noqa: BLE001 -- intentional CLI-level catch
        print(f"\n[error] That turn failed: {exc}\n")
        return

    print(f"\nAssistant: {result.get('response', '(no response)')}")
    if result.get("needs_escalation"):
        reason = result.get("escalation_reason")
        print(f"[flagged for review: {reason}]")
    print()


def main() -> None:
    graph = build_graph()
    # Fresh thread_id per process run -- each CLI session is its own
    # conversation. MemorySaver wouldn't persist across runs anyway,
    # but generating one explicitly documents that intent rather
    # than relying on that as an incidental side effect.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("Intelligent Form Agent -- type /help for commands.\n")

    startup_files = sys.argv[1:]
    if startup_files:
        print(f"Ingesting {len(startup_files)} form(s) from startup...")
        _run_turn(graph, config, uploaded_files=startup_files)

    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not raw:
            continue
        if raw in ("/quit", "/exit"):
            print("Goodbye.")
            break
        if raw == "/help":
            print(HELP_TEXT)
            continue
        if raw.startswith("/upload "):
            path = raw[len("/upload "):].strip()
            _run_turn(graph, config, uploaded_files=[path])
            continue

        _run_turn(graph, config, user_text=raw)


if __name__ == "__main__":
    main()
