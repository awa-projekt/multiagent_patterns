"""Pattern 4 - Parallelization (fan-out / fan-in), native implementation.

Two flavours:

1. `build_graph` - *sectioning*: four independent analysts look at the same
   e-mail at the same time (static fan-out from START). Code aggregates their
   structured outputs into a triage report; a responder agent then resolves
   the e-mail using that report.

       START -> classify_intent    --\\
             -> analyze_sentiment  ---+-> aggregate(code) -[spam]-> finalize_spam -> dispatch
             -> extract_entities   ---+                   -[else]-> respond(agent) -> dispatch -> END
             -> screen_compliance  --/

2. `build_inbox_digest_graph` - *map-reduce*: one `Send` per e-mail runs a
   complete per-e-mail workflow in parallel; a reducer builds the digest.
"""

from __future__ import annotations

import operator
from collections import Counter
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from email_assistant import prompts
from email_assistant.common import EmailInput, EmailOutput, EmailState, dispatch, email_message
from email_assistant.schemas import (
    ComplianceScreen,
    Email,
    EmailClassification,
    EmailResolution,
    ExtractedEntities,
    SentimentAnalysis,
)
from email_assistant.tools import ALL_TOOLS

PRIORITY_ORDER = ["low", "normal", "high", "urgent"]


class ParallelState(EmailState):
    intent: NotRequired[EmailClassification]
    sentiment: NotRequired[SentimentAnalysis]
    entities: NotRequired[ExtractedEntities]
    compliance: NotRequired[ComplianceScreen]
    triage: NotRequired[dict[str, Any]]


def build_graph(model: BaseChatModel):
    def analyst(key: str, schema: type, system_prompt: str):
        llm = model.with_structured_output(schema)

        def run(state: ParallelState) -> dict:
            # Each branch writes its own key, so no reducer is needed.
            return {key: llm.invoke([SystemMessage(system_prompt), HumanMessage(state["email"].as_prompt())])}

        return run

    def aggregate(state: ParallelState) -> dict:
        """Fan-in: merge the independent analyses with plain code."""
        intent, sentiment, compliance = state["intent"], state["sentiment"], state["compliance"]
        priority = max(intent.priority, sentiment.urgency, key=PRIORITY_ORDER.index)
        needs_human = intent.needs_human or compliance.legal_threat or compliance.data_protection_request
        category = "spam" if compliance.phishing else intent.category
        triage = {
            "category": category,
            "additional_categories": intent.additional_categories,
            "priority": priority,
            "sentiment": sentiment.sentiment,
            "needs_human": needs_human,
            "summary": intent.summary,
            "signals": sentiment.signals,
            "entities": state["entities"].model_dump(),
            "compliance": compliance.model_dump(),
        }
        return {"triage": triage}

    def route_after_aggregate(state: ParallelState) -> Literal["finalize_spam", "respond"]:
        return "finalize_spam" if state["triage"]["category"] == "spam" else "respond"

    def finalize_spam(state: ParallelState) -> dict:
        return {
            "resolution": EmailResolution(
                email_id=state["email"].id,
                category="spam",
                priority="low",
                action="ignore",
                internal_note=f"Spam / phishing ({state['compliance'].notes}) - no reply sent.",
            )
        }

    responder = create_agent(
        model, tools=ALL_TOOLS, system_prompt=prompts.RESPONDER, response_format=EmailResolution, name="responder"
    )

    def respond(state: ParallelState) -> dict:
        message = email_message(state["email"])
        message.content += f"\n\nTriage report from the analysts:\n{state['triage']}"
        result = responder.invoke({"messages": [message]})
        return {"resolution": result["structured_response"]}

    builder = StateGraph(ParallelState, input_schema=EmailInput, output_schema=EmailOutput)
    analysts = {
        "classify_intent": analyst("intent", EmailClassification, prompts.CLASSIFIER),
        "analyze_sentiment": analyst("sentiment", SentimentAnalysis, prompts.SENTIMENT_ANALYST),
        "extract_entities": analyst("entities", ExtractedEntities, prompts.ENTITY_EXTRACTOR),
        "screen_compliance": analyst("compliance", ComplianceScreen, prompts.COMPLIANCE_SCREENER),
    }
    for name, node in analysts.items():
        builder.add_node(name, node)
        builder.add_edge(START, name)  # static fan-out
    builder.add_node("aggregate", aggregate)
    builder.add_edge(list(analysts), "aggregate")  # fan-in: waits for all branches
    builder.add_node("finalize_spam", finalize_spam)
    builder.add_node("respond", respond)
    builder.add_node("dispatch", dispatch)
    builder.add_conditional_edges("aggregate", route_after_aggregate)
    builder.add_edge("finalize_spam", "dispatch")
    builder.add_edge("respond", "dispatch")
    builder.add_edge("dispatch", END)
    return builder.compile(name="parallel_triage")


# --------------------------------------------------------------- map-reduce
class DigestState(TypedDict):
    emails: list[Email]
    results: Annotated[list[dict[str, Any]], operator.add]
    digest: NotRequired[dict[str, Any]]


class OneEmail(TypedDict):
    email: Email


def build_inbox_digest_graph(per_email_workflow):
    """Map a per-e-mail workflow (any pattern!) over the whole inbox, then reduce.

    START --Send x N--> process_email -> reduce -> END
    """

    def fan_out(state: DigestState) -> list[Send]:
        return [Send("process_email", {"email": e}) for e in state["emails"]]

    def process_email(state: OneEmail) -> dict:
        out = per_email_workflow.invoke({"email": state["email"]})
        r = out["resolution"]
        return {
            "results": [
                {
                    "email_id": r.email_id,
                    "category": r.category,
                    "priority": r.priority,
                    "action": r.action,
                    "status": out["delivery"].status,
                    "actions_taken": r.actions_taken,
                }
            ]
        }

    def reduce(state: DigestState) -> dict:
        results = sorted(state["results"], key=lambda r: r["email_id"])
        return {
            "digest": {
                "processed": len(results),
                "by_category": dict(Counter(r["category"] for r in results)),
                "by_status": dict(Counter(r["status"] for r in results)),
                "urgent": [r["email_id"] for r in results if r["priority"] == "urgent"],
                "escalated": [r["email_id"] for r in results if r["status"] == "escalated"],
                "results": results,
            }
        }

    builder = StateGraph(DigestState)
    builder.add_node("process_email", process_email)
    builder.add_node("reduce", reduce)
    builder.add_conditional_edges(START, fan_out, ["process_email"])
    builder.add_edge("process_email", "reduce")
    builder.add_edge("reduce", END)
    return builder.compile(name="inbox_digest")
