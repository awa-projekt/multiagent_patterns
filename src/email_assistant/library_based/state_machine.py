"""Pattern 9 - Handoffs as a state machine, built with `create_state_machine_agent` + `Step`."""

from typing import Any, NotRequired

from langchain.tools import ToolRuntime, tool

from agentpatterns import StateMachineState, Step, create_state_machine_agent, transition
from email_assistant import prompts
from email_assistant.common import build_email_workflow
from email_assistant.schemas import CATEGORY_TO_DOMAIN, Category, EmailResolution, Priority, Sentiment
from email_assistant.tools import DOMAIN_TOOLS, lookup_customer


class CaseState(StateMachineState):
    triage: NotRequired[dict[str, Any]]


@tool
def record_triage(
    category: Category,
    additional_categories: list[Category],
    priority: Priority,
    sentiment: Sentiment,
    needs_human: bool,
    summary: str,
    runtime: ToolRuntime,
):
    """Record the triage of the e-mail. Moves the case to the next step."""
    triage = {
        "category": category,
        "additional_categories": additional_categories,
        "priority": priority,
        "sentiment": sentiment,
        "needs_human": needs_human,
        "summary": summary,
    }
    target = "respond" if category == "spam" else "resolve"
    return transition(target, runtime.tool_call_id, f"Triage recorded. Next step: {target}.", triage=triage)


def only_triaged_domains(state, tools):
    """Narrow the resolve step to the tools of the triaged domains (+ transition tools)."""
    triage = state.get("triage") or {}
    domains = [
        CATEGORY_TO_DOMAIN[c]
        for c in [triage.get("category"), *triage.get("additional_categories", [])]
        if c in CATEGORY_TO_DOMAIN
    ]
    if triage.get("needs_human"):
        domains.append("relations")
    allowed = {t.name for d in domains for t in DOMAIN_TOOLS[d]}
    return [t for t in tools if t.name in allowed or t.name.startswith("go_to_")]


def build_graph(model):
    all_domain_tools = list({t.name: t for tools in DOMAIN_TOOLS.values() for t in tools}.values())
    agent = create_state_machine_agent(
        model,
        steps=[
            Step("triage", prompts.CASE_HANDLER_STEPS["triage"], tools=[lookup_customer, record_triage]),
            Step(
                "resolve",
                prompts.CASE_HANDLER_STEPS["resolve"],
                tools=all_domain_tools,
                transitions=["respond"],
                tool_filter=only_triaged_domains,
            ),
            Step("respond", prompts.CASE_HANDLER_STEPS["respond"], final=True),
        ],
        response_format=EmailResolution,
        state_schema=CaseState,
        name="case_handler",
    )
    return build_email_workflow(agent, "state_machine")
