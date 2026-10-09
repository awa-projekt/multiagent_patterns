"""Pattern 9 - Handoffs as a state machine (single agent + middleware), native implementation.

The LangChain docs recommend this variant of handoffs for most cases: ONE
agent whose configuration (system prompt, tools, structured output) changes
with a `current_step` state variable. Tools move the agent between steps by
returning `Command(update={"current_step": ...})`; a `wrap_model_call`
middleware applies the step's configuration before every model call.

    triage   (lookup_customer, record_triage)          --record_triage-->     resolve
    resolve  (tools of the triaged domains, finish)    --finish_resolution--> respond
    respond  (no tools, structured EmailResolution)    -> END

This enforces order: the agent cannot refund before it has triaged, and it
only ever sees the tools that make sense in its current step.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.messages import SystemMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from email_assistant import prompts
from email_assistant.common import EmailInput, EmailOutput, EmailState, dispatch, email_message
from email_assistant.schemas import CATEGORY_TO_DOMAIN, Category, EmailResolution, Priority, Sentiment
from email_assistant.tools import ALL_TOOLS, DOMAIN_TOOLS, lookup_customer


class CaseState(AgentState):
    current_step: NotRequired[str]
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
) -> Command:
    """Record the triage of the e-mail. Moves the case to the next step."""
    next_step = "respond" if category == "spam" else "resolve"
    triage = {
        "category": category,
        "additional_categories": additional_categories,
        "priority": priority,
        "sentiment": sentiment,
        "needs_human": needs_human,
        "summary": summary,
    }
    return Command(
        update={
            "messages": [ToolMessage(f"Triage recorded. Next step: {next_step}.", tool_call_id=runtime.tool_call_id)],
            "triage": triage,
            "current_step": next_step,
        }
    )


@tool
def finish_resolution(summary: str, runtime: ToolRuntime) -> Command:
    """Call when every request of the e-mail has been handled. Moves the case to 'respond'."""
    return Command(
        update={
            "messages": [ToolMessage(f"Resolution finished: {summary}", tool_call_id=runtime.tool_call_id)],
            "current_step": "respond",
        }
    )


def _resolve_tools(triage: dict[str, Any]) -> list:
    """Tools unlocked in the resolve step: only those of the triaged domains."""
    domains = [
        CATEGORY_TO_DOMAIN[c] for c in [triage["category"], *triage["additional_categories"]] if c in CATEGORY_TO_DOMAIN
    ]
    if triage.get("needs_human"):
        domains.append("relations")
    tools = {t.name: t for d in domains for t in DOMAIN_TOOLS[d]}
    return [*tools.values(), finish_resolution]


@wrap_model_call
def apply_step_config(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    step = request.state.get("current_step", "triage")
    triage = request.state.get("triage") or {}
    if step == "triage":
        tools = [lookup_customer, record_triage]
    elif step == "resolve":
        tools = _resolve_tools(triage)
    else:
        tools = []
    overrides: dict[str, Any] = {
        "system_message": SystemMessage(prompts.CASE_HANDLER_STEPS[step].format(triage=json.dumps(triage))),
        "tools": tools,
    }
    if step != "respond":
        overrides["response_format"] = None  # structured output only in the final step
    return handler(request.override(**overrides))


def build_agent(model: BaseChatModel, *, checkpointer=None):
    return create_agent(
        model,
        # All tools must be registered up front; the middleware narrows them per step.
        tools=[*ALL_TOOLS, record_triage, finish_resolution],
        state_schema=CaseState,
        middleware=[apply_step_config],
        response_format=EmailResolution,
        checkpointer=checkpointer,
        name="case_handler",
    )


def build_graph(model: BaseChatModel):
    agent = build_agent(model)

    def handle_email(state: EmailState) -> dict:
        result = agent.invoke({"messages": [email_message(state["email"])], "current_step": "triage"})
        return {"resolution": result["structured_response"]}

    return (
        StateGraph(EmailState, input_schema=EmailInput, output_schema=EmailOutput)
        .add_node("case_handler", handle_email)
        .add_node("dispatch", dispatch)
        .add_edge(START, "case_handler")
        .add_edge("case_handler", "dispatch")
        .add_edge("dispatch", END)
        .compile(name="state_machine")
    )
