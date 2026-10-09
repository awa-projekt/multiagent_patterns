"""Opt-in smoke tests against a real chat model.

    uv sync --extra anthropic
    AGENTPATTERNS_TEST_MODEL=anthropic:claude-opus-5 uv run pytest -m real_model

Any `init_chat_model` identifier works (install the provider package). The tests
check that each pattern runs end to end with the provider's structured output
and tool calling and returns the declared structure - not the quality of the
answers. They cost real tokens (roughly 40-60 model calls per run).
"""

from __future__ import annotations

import os

import pytest
from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel, Field

import agentpatterns as ap
from agentpatterns.testing import UsageTracker

MODEL = os.environ.get("AGENTPATTERNS_TEST_MODEL")

pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(not MODEL, reason="set AGENTPATTERNS_TEST_MODEL (e.g. anthropic:claude-opus-5) to run"),
]


class Answer(BaseModel):
    """Final answer."""

    text: str = Field(description="The answer in one or two sentences.")


class Report(BaseModel):
    """Research report."""

    summary: str = Field(description="Three sentences at most.")
    sources: list[str] = Field(description="Source URLs the summary is based on.")


FACTS = {
    "market": "source://market-2026: The EU e-bike market grew 12% in 2026 to 6.1M units.",
    "rivals": "source://rivals-2026: The top three brands hold 41% market share.",
    "pricing": "source://pricing-2026: The median e-bike price is 2,400 EUR.",
}


@tool
def search(query: str) -> str:
    """Search market research. Topics: market size, competitors (rivals), pricing."""
    hits = [fact for key, fact in FACTS.items() if key in query.lower() or (key == "rivals" and "compet" in query)]
    return "\n".join(hits) or "No results. Try: market, rivals, pricing."


@tool
def lookup_invoice(invoice_id: str) -> str:
    """Look up an invoice by id."""
    return f"{invoice_id}: 348 EUR, paid twice on 2026-09-01 (duplicate charge)."


@pytest.fixture(scope="module")
def model():
    from langchain.chat_models import init_chat_model

    try:
        return init_chat_model(MODEL)
    except ImportError as exc:
        pytest.skip(f"provider package missing: {exc}")


def agent(model, name: str, prompt: str, tools=(), **kwargs):
    return create_agent(model, tools=list(tools), system_prompt=prompt, name=name, **kwargs)


def test_structured_llm(model):
    assert isinstance(ap.structured_llm(model, Answer).invoke("What is 2 + 3?"), Answer)


def test_router(model):
    router = ap.create_router(
        model,
        [
            ap.AgentSpec("billing", "Invoices, charges, refunds", agent(model, "billing", "Answer billing questions.")),
            ap.AgentSpec("tech", "Errors, outages, bugs", agent(model, "tech", "Answer technical questions.")),
        ],
        response_format=Answer,
    )
    out = router.invoke({"messages": [("user", "I was charged twice for my last invoice.")]})
    assert isinstance(out["structured_response"], Answer)
    assert out["routes"] and {r["agent"] for r in out["routes"]} <= {"billing", "tech"}


def test_pipeline_llm_step(model):
    pipeline = ap.create_pipeline(
        [ap.llm_step("classify", model, system_prompt="Answer the question briefly.", output_schema=Answer)]
    )
    out = pipeline.invoke({"messages": [("user", "Is Berlin in Germany?")]})
    assert isinstance(out["outputs"]["classify"], Answer)


def test_parallel_with_model_synthesizer(model):
    parallel = ap.create_parallel(
        [
            ap.AgentSpec(
                "optimist", "Optimist", agent(model, "optimist", "Give the most optimistic view in one line.")
            ),
            ap.AgentSpec("skeptic", "Skeptic", agent(model, "skeptic", "Give the most skeptical view in one line.")),
        ],
        model=model,
        response_format=Answer,
    )
    out = parallel.invoke({"messages": [("user", "Will e-bikes replace cars in cities?")]})
    assert isinstance(out["structured_response"], Answer)


def test_research_review_recipe(model):
    researcher = agent(model, "researcher", "Research the question with the search tool; cite sources.", [search])
    orchestrator = ap.create_orchestrator(
        model,
        [ap.AgentSpec("researcher", "Researches one question with a search tool", researcher)],
        response_format=Report,
    )
    reviewer = agent(
        model,
        "reviewer",
        "You review research reports. Pass a report only if it covers market size, competitors AND pricing, "
        "and every claim is backed by the research results.",
        response_format=ap.Evaluation,
    )
    reviewed = ap.create_evaluator_optimizer(
        orchestrator,
        reviewer,
        carry_over=["results"],
        evaluator_context=lambda s: ap.results_block(s.get("results", []), "Research results"),
        max_iterations=2,
    )
    tracker = UsageTracker()
    out = reviewed.invoke(
        {"messages": [("user", "Brief me on the EU e-bike market size and competitors.")]},
        {"callbacks": [tracker]},
    )
    assert isinstance(out["structured_response"], Report)
    assert isinstance(out["evaluation"], ap.Evaluation)
    assert out["results"] and 1 <= out["iterations"] <= 2
    assert tracker.tool_calls["search"] >= 1


def test_evaluator_with_model_judge(model):
    loop = ap.create_evaluator_optimizer(
        agent(model, "writer", "Write one short sentence."),
        model,
        evaluator_prompt="Pass the sentence if it is under 20 words.",
        max_iterations=2,
    )
    out = loop.invoke({"messages": [("user", "Describe the sea.")]})
    assert isinstance(out["evaluation"], ap.Evaluation)


def test_supervisor(model):
    billing = agent(model, "billing", "Resolve billing questions with the invoice tool.", [lookup_invoice])
    supervisor = ap.create_supervisor(
        model,
        [ap.AgentSpec("billing", "Invoices and charges; has invoice lookup", billing)],
        system_prompt="You coordinate customer service. Delegate billing questions to the billing agent.",
        response_format=Answer,
    )
    tracker = UsageTracker()
    out = supervisor.invoke({"messages": [("user", "Why was invoice INV-7 charged twice?")]}, {"callbacks": [tracker]})
    assert isinstance(out["structured_response"], Answer)
    assert tracker.tool_calls["billing"] == 1


def test_swarm_handoff(model):
    swarm = ap.create_swarm(
        model,
        [
            ap.SwarmAgent("front_desk", "Greets and routes", "Hand billing questions to billing. Never answer them."),
            ap.SwarmAgent("billing", "Billing expert", "Resolve billing questions.", tools=[lookup_invoice]),
        ],
        default_agent="front_desk",
    )
    out = swarm.invoke({"messages": [("user", "Invoice INV-7 was charged twice, please check.")]})
    assert out["active_agent"] == "billing"


def test_state_machine(model):
    agent_graph = ap.create_state_machine_agent(
        model,
        [
            ap.Step("triage", "Classify the request, then move to the answer step.", transitions=["answer"]),
            ap.Step("answer", "Answer the request briefly.", final=True),
        ],
        response_format=Answer,
    )
    out = agent_graph.invoke({"messages": [("user", "How do I reset my password?")]})
    assert out["current_step"] == "answer"
    assert isinstance(out["structured_response"], Answer)


def test_skills(model):
    skills_agent = ap.create_skills_agent(
        model,
        [ap.Skill("billing", "Invoices and charges", "Look up the invoice before answering.", tools=[lookup_invoice])],
        system_prompt="You are a support agent. Load the matching skill before answering.",
        response_format=Answer,
    )
    out = skills_agent.invoke({"messages": [("user", "Invoice INV-7 was charged twice, why?")]})
    assert "billing" in out.get("loaded_skills", [])
    assert isinstance(out["structured_response"], Answer)


def test_messaging(model):
    researcher = agent(
        model,
        "researcher",
        "Look up the EU e-bike market size with the search tool, send the numbers to the writer with send_message, "
        "then answer 'done'.",
        [search],
        middleware=[ap.MessagingMiddleware()],
    )
    writer = agent(
        model,
        "writer",
        "Write one sentence about the EU e-bike market, using only the facts your colleagues sent you.",
        middleware=[ap.MessagingMiddleware()],
    )
    pipeline = ap.create_pipeline([ap.agent_step("researcher", researcher), ap.agent_step("writer", writer)])
    mailbox = ap.Mailbox()
    pipeline.invoke(
        {"messages": [("user", "Brief me on the EU e-bike market.")]}, {"configurable": {"mailbox": mailbox}}
    )
    assert ("researcher", "writer") in [(m.sender, m.to) for m in mailbox.log]
    assert mailbox.undelivered == []
