"""Interactive and one-shot CLI entrypoint for the private AI agent."""

from __future__ import annotations

import argparse

from private_agent.config import load_settings
from private_agent.graph import AgentState, build_graph
from private_agent.logging_utils import configure_logging


def _initial_state(question: str) -> AgentState:
    """Create a clean state payload for graph invocation."""
    return {
        "question": question,
        "context": [],
        "answer": "",
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for interactive or one-shot execution."""
    parser = argparse.ArgumentParser(description="Run the private enterprise AI agent.")
    parser.add_argument(
        "--question",
        type=str,
        default="",
        help="Ask one question in non-interactive mode.",
    )
    return parser


def main() -> None:
    """Program entrypoint."""
    args = _build_arg_parser().parse_args()

    settings = load_settings()
    configure_logging(settings.log_level)

    app = build_graph(settings)

    if args.question.strip():
        result = app.invoke(_initial_state(args.question.strip()))
        print(result["answer"])
        return

    print("Private Enterprise AI Agent")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("User: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = app.invoke(_initial_state(question))
        print("\nAssistant:")
        print(result["answer"])
        print()


if __name__ == "__main__":
    main()
