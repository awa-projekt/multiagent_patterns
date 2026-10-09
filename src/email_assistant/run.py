"""Command line demo: run the e-mail workflows and compare the patterns.

uv run email-demo                         # comparison table, all patterns (native)
uv run email-demo --impl library          # same with the agentpatterns-based workflows
uv run email-demo -p router -e E-1004 -v  # one pattern, one e-mail, show the reply
uv run email-demo --mermaid swarm         # print the graph as a Mermaid diagram
uv run email-demo --digest                # map-reduce digest of the whole inbox
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from agentpatterns.testing import UsageTracker
from email_assistant import library_based, native
from email_assistant.data import BACKEND, EXPECTED, INBOX, reset_backend
from email_assistant.mock_llm import create_mock_llm


def _registry(impl: str) -> dict[str, Any]:
    patterns = dict(native.PATTERNS if impl == "native" else library_based.PATTERNS)
    if impl == "library":
        from email_assistant.library_based import composite

        patterns["composite"] = composite.build_graph
    return patterns


def run_pattern(name: str, build, model, emails, verbose: bool = False) -> dict[str, Any]:
    reset_backend()
    graph = build(model)
    rows = []
    for email in emails:
        tracker = UsageTracker()
        start = time.perf_counter()
        out = graph.invoke({"email": email}, config={"callbacks": [tracker]})
        elapsed = time.perf_counter() - start
        resolution, usage = out["resolution"], tracker.summary()
        expected = EXPECTED[email.id]
        ok = resolution.category == expected["category"] and resolution.action == expected["action"]
        rows.append({"email": email.id, "ok": ok, "usage": usage, "seconds": elapsed, "resolution": resolution})
        if verbose:
            print(
                f"\n--- {name} / {email.id}: {resolution.category} -> {resolution.action} "
                f"({usage['model_calls']} model calls, {usage['tool_calls']} tool calls)"
            )
            print(f"actions: {resolution.actions_taken}")
            if resolution.reply_body:
                print(resolution.reply_body)
            print(f"model calls by agent: {usage['model_calls_by_agent']}")
    return {
        "pattern": name,
        "correct": sum(r["ok"] for r in rows),
        "emails": len(rows),
        "model_calls": sum(r["usage"]["model_calls"] for r in rows),
        "tool_calls": sum(r["usage"]["tool_calls"] for r in rows),
        "input_tokens": sum(r["usage"]["input_tokens"] for r in rows),
        "multi_intent_calls": next((r["usage"]["model_calls"] for r in rows if r["email"] == "E-1004"), None),
        "rows": rows,
    }


def print_table(results: list[dict[str, Any]]) -> None:
    header = (
        "| Pattern | Correct | Model calls (total) | Calls E-1004 (multi-intent) | Tool calls | Est. input tokens |"
    )
    print(header)
    print("|" + "---|" * 6)
    for r in results:
        print(
            f"| {r['pattern']} | {r['correct']}/{r['emails']} | {r['model_calls']} | "
            f"{r['multi_intent_calls'] if r['multi_intent_calls'] is not None else '-'} | "
            f"{r['tool_calls']} | {r['input_tokens']:,} |"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--impl", choices=["native", "library"], default="native")
    parser.add_argument("-p", "--pattern", action="append", help="Pattern(s) to run (default: all)")
    parser.add_argument("-e", "--email", action="append", help="E-mail id(s) to process (default: all)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print every resolution")
    parser.add_argument("--mermaid", metavar="PATTERN", help="Print the pattern's graph as Mermaid and exit")
    parser.add_argument("--digest", action="store_true", help="Run the map-reduce inbox digest and exit")
    args = parser.parse_args(argv)

    model = create_mock_llm()
    registry = _registry(args.impl)

    if args.mermaid:
        print(registry[args.mermaid](model).get_graph(xray=1).draw_mermaid())
        return
    if args.digest:
        from email_assistant.library_based.composite import build_inbox_digest

        reset_backend()
        out = build_inbox_digest(model).invoke({"items": INBOX})
        print(json.dumps(out["output"], indent=2))
        return

    emails = [e for e in INBOX if not args.email or e.id in args.email]
    names = args.pattern or list(registry)
    results = [run_pattern(n, registry[n], model, emails, args.verbose) for n in names]
    print()
    print_table(results)
    if args.verbose:
        snapshot = BACKEND.snapshot()
        print("\nBackend after the last pattern:", {k: sorted(v) for k, v in snapshot.items()})


if __name__ == "__main__":
    main()
