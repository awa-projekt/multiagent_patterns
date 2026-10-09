"""Pattern 2 - Sequential pipeline (prompt chaining), native implementation.

A fixed chain of single LLM calls with structured output, interleaved with
deterministic code steps and programmatic gates. No agent loop: the model
never decides *what happens next* - the graph does. The model proposes
actions; code executes them.

    START -> classify(LLM) -[spam]-> finalize_spam -> dispatch
                           -[else]-> enrich(code) -> plan_actions(LLM) -> execute_actions(code)
                                     -> write_reply(LLM) -> review(LLM)
                                     -[passed]-> dispatch
                                     -[failed]-> escalate_for_review(code) -> dispatch -> END
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, NotRequired

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from email_assistant import prompts
from email_assistant.common import EmailInput, EmailOutput, EmailState, dispatch
from email_assistant.schemas import (
    ActionPlan,
    EmailClassification,
    EmailResolution,
    QualityReview,
)
from email_assistant.tools import ALL_TOOLS

TOOLS = {t.name: t for t in ALL_TOOLS}
ACTION_TOOL = {
    "refund": "issue_refund",
    "ticket": "create_ticket",
    "lead": "create_sales_lead",
    "escalation": "escalate_to_human",
}


class PipelineState(EmailState):
    classification: NotRequired[EmailClassification]
    facts: NotRequired[list[dict[str, Any]]]
    plan: NotRequired[ActionPlan]
    executed: NotRequired[list[dict[str, Any]]]
    review: NotRequired[QualityReview]


def _facts_block(*fact_lists: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(f, default=str) for facts in fact_lists for f in facts)


def build_graph(model: BaseChatModel):
    classifier = model.with_structured_output(EmailClassification)
    planner = model.with_structured_output(ActionPlan)
    writer = model.with_structured_output(EmailResolution)
    reviewer = model.with_structured_output(QualityReview)

    def classify(state: PipelineState) -> dict:
        email = state["email"]
        result = classifier.invoke([SystemMessage(prompts.CLASSIFIER), HumanMessage(email.as_prompt())])
        return {"classification": result}

    def route_after_classify(state: PipelineState) -> Literal["finalize_spam", "enrich"]:
        # Gate 1 (programmatic): spam never reaches the expensive steps.
        return "finalize_spam" if state["classification"].category == "spam" else "enrich"

    def finalize_spam(state: PipelineState) -> dict:
        return {
            "resolution": EmailResolution(
                email_id=state["email"].id,
                category="spam",
                priority="low",
                action="ignore",
                internal_note="Spam / phishing - no reply sent.",
            )
        }

    def enrich(state: PipelineState) -> dict:
        """Deterministic retrieval: no LLM decides which systems to query."""
        email, classification = state["email"], state["classification"]
        text = f"{email.subject} {email.body}".lower()
        calls: list[tuple[str, dict]] = [("lookup_customer", {"email": email.sender})]
        calls += [("get_invoice", {"invoice_id": i}) for i in dict.fromkeys(re.findall(r"INV-\d{4}-\d{4}", email.body))]
        calls += [("check_service_status", {"service": s}) for s in ("dashboard", "export", "api") if s in text]
        if "sales" in classification.categories:
            seats = re.search(r"(\d+)\s*(?:seats|users)", text)
            plan = "enterprise" if "enterprise" in text else "business"
            calls.append(("get_plan_pricing", {"plan": plan, "seats": int(seats.group(1)) if seats else 10}))
        calls.append(("search_knowledge_base", {"query": f"{email.subject} {classification.summary}"}))
        return {"facts": [{"tool": t, "args": a, "result": TOOLS[t].invoke(a)} for t, a in calls]}

    def plan_actions(state: PipelineState) -> dict:
        prompt = (
            f"{state['email'].as_prompt()}\n\nClassification: {state['classification'].model_dump_json()}\n\n"
            f"Facts from our systems (one JSON per line):\n{_facts_block(state['facts'])}"
        )
        return {"plan": planner.invoke([SystemMessage(prompts.ACTION_PLANNER), HumanMessage(prompt)])}

    def execute_actions(state: PipelineState) -> dict:
        """The model proposed, code disposes: tools enforce policy on every action."""
        executed = []
        for action in state["plan"].actions:
            tool = ACTION_TOOL[action.kind]
            args = action.model_dump(exclude={"kind"})
            executed.append({"tool": tool, "args": args, "result": TOOLS[tool].invoke(args)})
        return {"executed": executed}

    def write_reply(state: PipelineState) -> dict:
        prompt = (
            f"{state['email'].as_prompt()}\n\nClassification: {state['classification'].model_dump_json()}\n\n"
            f"Research and executed actions (one JSON per line):\n{_facts_block(state['facts'], state['executed'])}"
        )
        return {"resolution": writer.invoke([SystemMessage(prompts.REPLY_WRITER), HumanMessage(prompt)])}

    def review(state: PipelineState) -> dict:
        prompt = f"{state['email'].as_prompt()}\n\nProposed resolution:\n{state['resolution'].model_dump_json()}"
        return {"review": reviewer.invoke([SystemMessage(prompts.REVIEWER), HumanMessage(prompt)])}

    def route_after_review(state: PipelineState) -> Literal["dispatch", "escalate_for_review"]:
        # Gate 2: a failed quality check never auto-sends.
        return "dispatch" if state["review"].passed else "escalate_for_review"

    def escalate_for_review(state: PipelineState) -> dict:
        resolution = state["resolution"].model_copy(
            update={
                "action": "escalate",
                "escalation_reason": "Quality review failed: " + "; ".join(state["review"].issues),
            }
        )
        return {"resolution": resolution}

    builder = StateGraph(PipelineState, input_schema=EmailInput, output_schema=EmailOutput)
    builder.add_node("classify", classify)
    builder.add_node("finalize_spam", finalize_spam)
    builder.add_node("enrich", enrich)
    builder.add_node("plan_actions", plan_actions)
    builder.add_node("execute_actions", execute_actions)
    builder.add_node("write_reply", write_reply)
    builder.add_node("review", review)
    builder.add_node("escalate_for_review", escalate_for_review)
    builder.add_node("dispatch", dispatch)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_after_classify)
    builder.add_edge("finalize_spam", "dispatch")
    builder.add_edge("enrich", "plan_actions")
    builder.add_edge("plan_actions", "execute_actions")
    builder.add_edge("execute_actions", "write_reply")
    builder.add_edge("write_reply", "review")
    builder.add_conditional_edges("review", route_after_review)
    builder.add_edge("escalate_for_review", "dispatch")
    builder.add_edge("dispatch", END)
    return builder.compile(name="sequential_pipeline")
