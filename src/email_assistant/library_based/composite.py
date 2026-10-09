"""Composition example: patterns nested inside patterns, embedded in a custom workflow.

    e-mail workflow (StateGraph)
    └── pipeline                                   create_pipeline
        ├── triage   = parallel analysts           create_parallel      (gate: spam -> ignore)
        └── resolve  = reflection loop             create_evaluator_optimizer
                       └── generator = router      create_router
                                       └── routes = specialist agents (create_agent)
    ├── human_approval (optional interrupt())
    └── dispatch

`build_inbox_digest` maps the whole workflow over an inbox with `create_map_reduce`.
This works because every factory returns a graph with the same agent contract.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agentpatterns import (
    agent_as_node,
    agent_step,
    create_evaluator_optimizer,
    create_map_reduce,
    create_parallel,
    create_pipeline,
    create_router,
)
from email_assistant import prompts
from email_assistant.common import EmailInput, EmailOutput, EmailState, create_checkpointer, dispatch, email_message
from email_assistant.library_based.evaluator_optimizer import escalate
from email_assistant.library_based.parallel import aggregate
from email_assistant.library_based.specialists import specialist_specs
from email_assistant.schemas import (
    ComplianceScreen,
    EmailClassification,
    EmailResolution,
    ExtractedEntities,
    QualityReview,
    SentimentAnalysis,
)


def build_resolution_pipeline(model: BaseChatModel):
    """The nested pattern stack (agent contract in, `EmailResolution` out)."""
    from langchain.agents import create_agent

    from agentpatterns import AgentSpec

    def analyst(name: str, schema: type, system_prompt: str) -> AgentSpec:
        return AgentSpec(
            name, name, create_agent(model, tools=[], system_prompt=system_prompt, response_format=schema, name=name)
        )

    triage = create_parallel(
        [
            analyst("intent", EmailClassification, prompts.CLASSIFIER),
            analyst("sentiment", SentimentAnalysis, prompts.SENTIMENT_ANALYST),
            analyst("entities", ExtractedEntities, prompts.ENTITY_EXTRACTOR),
            analyst("compliance", ComplianceScreen, prompts.COMPLIANCE_SCREENER),
        ],
        aggregator=aggregate,
        name="triage",
    )
    router = create_router(
        model,
        specialist_specs(model),
        system_prompt=prompts.ROUTER,
        synthesizer_prompt=prompts.REPLY_WRITER,
        response_format=EmailResolution,
        name="inbox_router",
    )
    quality_checked = create_evaluator_optimizer(
        router,
        model,
        evaluator_prompt=prompts.REVIEWER,
        evaluation_schema=QualityReview,
        max_iterations=2,
        on_max_iterations=escalate,
        name="quality_loop",
    )

    def ignore_spam(state: dict[str, Any]) -> EmailResolution:
        email = state["context"]["email"]
        return EmailResolution(
            email_id=email.id,
            category="spam",
            priority="low",
            action="ignore",
            internal_note="Spam / phishing - no reply sent.",
        )

    return create_pipeline(
        [
            agent_step(
                "triage", triage, gate=lambda s: s["outputs"]["triage"]["category"] != "spam", on_gate_fail=ignore_spam
            ),
            agent_step("quality_loop", quality_checked),
        ],
        name="resolution_pipeline",
    )


def build_graph(model: BaseChatModel, *, require_approval: bool = False, checkpointer=None):
    """E-mail workflow around the nested patterns, optionally with human approval.

    With `require_approval=True`, escalations and replies with refunds pause
    with `interrupt()` until a human resumes with `Command(resume="approve" | "reject")`.
    """
    pipeline = build_resolution_pipeline(model)
    resolve = agent_as_node(
        pipeline,
        input=lambda s: {"messages": [email_message(s["email"])], "context": {"email": s["email"]}},
        output=lambda result, s: {"resolution": result["structured_response"]},
        name="resolve",
    )

    def human_approval(state: EmailState) -> dict:
        resolution = state["resolution"]
        risky = resolution.action == "escalate" or any(a.startswith("Refund") for a in resolution.actions_taken)
        if not risky:
            return {}
        decision = interrupt(
            {
                "email_id": resolution.email_id,
                "action": resolution.action,
                "actions_taken": resolution.actions_taken,
                "reply": resolution.reply_body,
            }
        )
        if decision == "reject":
            return {
                "resolution": resolution.model_copy(
                    update={"action": "escalate", "escalation_reason": "Rejected by human reviewer"}
                )
            }
        return {}

    builder = StateGraph(EmailState, input_schema=EmailInput, output_schema=EmailOutput)
    builder.add_node("resolve", resolve)
    builder.add_node("dispatch", dispatch)
    builder.add_edge(START, "resolve")
    if require_approval:
        builder.add_node("human_approval", human_approval)
        builder.add_edge("resolve", "human_approval")
        builder.add_edge("human_approval", "dispatch")
    else:
        builder.add_edge("resolve", "dispatch")
    builder.add_edge("dispatch", END)
    if require_approval and checkpointer is None:
        checkpointer = create_checkpointer()  # interrupts need a checkpointer
    return builder.compile(name="composite_workflow", checkpointer=checkpointer)


def build_inbox_digest(model: BaseChatModel, workflow=None):
    """Map any per-e-mail workflow over an inbox (`{"items": [Email, ...]}`) and summarize."""
    workflow = workflow or build_graph(model)

    def summarize(out: dict[str, Any]) -> dict[str, Any]:
        r = out["resolution"]
        return {
            "email_id": r.email_id,
            "category": r.category,
            "priority": r.priority,
            "status": out["delivery"].status,
            "actions_taken": r.actions_taken,
        }

    def digest(results: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "processed": len(results),
            "by_category": dict(Counter(r["category"] for r in results)),
            "by_status": dict(Counter(r["status"] for r in results)),
            "urgent": [r["email_id"] for r in results if r["priority"] == "urgent"],
        }

    return create_map_reduce(
        workflow, prepare=lambda email: {"email": email}, extract=summarize, reduce=digest, name="inbox_digest"
    )
