"""End-to-end tests of the e-mail use case for every pattern (native and library-based)."""

from __future__ import annotations

import pytest
from langgraph.types import Command

from email_assistant import library_based, native
from email_assistant.data import BACKEND, EXPECTED, INBOX, reset_backend
from email_assistant.library_based import composite, supervisor

WORKFLOWS = (
    [("native", n, b) for n, b in native.PATTERNS.items()]
    + [("library", n, b) for n, b in library_based.PATTERNS.items()]
    + [("library", "composite", composite.build_graph)]
)


def _signature(resolution):
    return resolution.action, resolution.reply_body, tuple(resolution.actions_taken)


@pytest.fixture(scope="module")
def reference(email_model):
    """Resolutions of the single-agent baseline - every pattern must reach the same outcome."""
    reset_backend()
    graph = native.PATTERNS["single_agent"](email_model)
    return {e.id: _signature(graph.invoke({"email": e})["resolution"]) for e in INBOX}


@pytest.mark.parametrize(("impl", "name", "build"), WORKFLOWS, ids=[f"{i}-{n}" for i, n, _ in WORKFLOWS])
def test_workflow_handles_whole_inbox(impl, name, build, email_model, reference):
    graph = build(email_model)
    for email in INBOX:
        out = graph.invoke({"email": email})
        resolution, delivery = out["resolution"], out["delivery"]
        assert resolution.category == EXPECTED[email.id]["category"], email.id
        assert resolution.action == EXPECTED[email.id]["action"], email.id
        assert _signature(resolution) == reference[email.id], email.id
        assert delivery.status == {"reply": "sent", "escalate": "escalated", "ignore": "ignored"}[resolution.action]

    snapshot = BACKEND.snapshot()
    assert len(snapshot["refunds"]) == 1  # E-1001 duplicate charge; never refund the open invoice
    assert {t["queue"] for t in snapshot["tickets"].values()} == {"billing", "technical"}
    assert len(snapshot["leads"]) == 1
    assert [e["team"] for e in snapshot["escalations"].values()] == ["dpo"]
    assert len(snapshot["outbox"]) == 5  # every e-mail except spam gets a reply / acknowledgment


@pytest.mark.parametrize(("impl", "name", "build"), WORKFLOWS, ids=[f"{i}-{n}" for i, n, _ in WORKFLOWS])
def test_workflow_with_native_structured_output(impl, name, build, native_email_model, reference):
    """Same outcomes when structured output uses the provider's native JSON mode (ProviderStrategy / json_schema)."""
    graph = build(native_email_model)
    for email in INBOX:
        assert _signature(graph.invoke({"email": email})["resolution"]) == reference[email.id], email.id


def test_multi_intent_email_covers_both_requests(email_model):
    out = native.PATTERNS["router"](email_model).invoke({"email": INBOX[3]})
    body = out["resolution"].reply_body
    assert "INV-2026-0901" in body and "450.00 EUR" in body
    assert "10,000 rows" in body


def test_supervisor_with_single_task_tool(email_model, reference):
    graph = supervisor.build_graph(email_model, delegation="task_tool")
    for email in INBOX:
        assert _signature(graph.invoke({"email": email})["resolution"]) == reference[email.id]


def test_human_approval_interrupt_and_resume(email_model):
    graph = composite.build_graph(email_model, require_approval=True)
    config = {"configurable": {"thread_id": "refund"}}
    paused = graph.invoke({"email": INBOX[0]}, config)
    assert paused["__interrupt__"][0].value["actions_taken"][0].startswith("Refund")
    assert not BACKEND.outbox  # nothing sent before approval
    done = graph.invoke(Command(resume="approve"), config)
    assert done["delivery"].status == "sent"

    config = {"configurable": {"thread_id": "gdpr"}}
    graph.invoke({"email": INBOX[5]}, config)
    rejected = graph.invoke(Command(resume="reject"), config)
    assert rejected["resolution"].escalation_reason == "Rejected by human reviewer"


def test_inbox_digest_map_reduce(email_model):
    out = composite.build_inbox_digest(email_model).invoke({"items": INBOX})
    assert [r["email_id"] for r in out["results"]] == [e.id for e in INBOX]
    assert out["output"]["by_status"] == {"sent": 4, "ignored": 1, "escalated": 1}
    assert out["output"]["urgent"] == ["E-1002", "E-1006"]


def test_native_parallel_digest(email_model):
    graph = native.parallel.build_inbox_digest_graph(native.PATTERNS["skills"](email_model))
    digest = graph.invoke({"emails": INBOX})["digest"]
    assert digest["processed"] == 6 and digest["escalated"] == ["E-1006"]
