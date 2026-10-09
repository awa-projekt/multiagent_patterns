"""Pattern 3 - Router, native implementation.

A single routing step (LLM with structured output) decides which specialist
agents must work on the e-mail - zero, one or several. Several routes run in
parallel via the `Send` API; a synthesizer merges their reports.

    START -> route(LLM) --Send--> billing_specialist  --\\
                        --Send--> technical_specialist --+--> synthesize(LLM) -> dispatch -> END
                        --(none)--------------------------/
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, NotRequired, TypedDict

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from email_assistant import prompts
from email_assistant.common import EmailInput, EmailOutput, EmailState, dispatch
from email_assistant.schemas import EmailResolution, SpecialistReport
from email_assistant.tools import DOMAIN_TOOLS

AgentName = Literal["billing_specialist", "technical_specialist", "sales_specialist", "relations_specialist"]


class Route(BaseModel):
    agent: AgentName = Field(description="Specialist that should handle this part.")
    task: str = Field(description="Self-contained task for the specialist, quoting the e-mail.")


class RoutingDecision(BaseModel):
    """Which specialists must work on the e-mail (empty for spam)."""

    routes: list[Route] = Field(default_factory=list)


class RouterState(EmailState):
    routes: NotRequired[list[Route]]
    # Parallel branches append to the same key -> needs a reducer.
    reports: Annotated[list[SpecialistReport], operator.add]


class SpecialistInput(TypedDict):
    """Private input of a specialist branch (sent via `Send`)."""

    task: str


def build_graph(model: BaseChatModel):
    router_llm = model.with_structured_output(RoutingDecision)
    writer = model.with_structured_output(EmailResolution)

    def route(state: RouterState) -> dict:
        decision = router_llm.invoke([SystemMessage(prompts.ROUTER), HumanMessage(state["email"].as_prompt())])
        return {"routes": decision.routes}

    def fan_out(state: RouterState) -> list[Send] | str:
        if not state["routes"]:
            return "synthesize"
        return [Send(r.agent, {"task": r.task}) for r in state["routes"]]

    def make_specialist(domain: str):
        agent = create_agent(
            model,
            tools=DOMAIN_TOOLS[domain],
            system_prompt=prompts.SPECIALISTS[domain],
            response_format=SpecialistReport,
            name=f"{domain}_specialist",
        )

        def run(state: SpecialistInput) -> dict:
            result = agent.invoke({"messages": [HumanMessage(state["task"])]})
            return {"reports": [result["structured_response"]]}

        return run

    def synthesize(state: RouterState) -> dict:
        reports = "\n".join(r.model_dump_json() for r in state.get("reports", []))
        prompt = f"{state['email'].as_prompt()}\n\nSpecialist reports (one JSON per line):\n{reports or '(none)'}"
        return {"resolution": writer.invoke([SystemMessage(prompts.REPLY_WRITER), HumanMessage(prompt)])}

    builder = StateGraph(RouterState, input_schema=EmailInput, output_schema=EmailOutput)
    builder.add_node("route", route)
    specialists = [f"{d}_specialist" for d in DOMAIN_TOOLS]
    for domain in DOMAIN_TOOLS:
        builder.add_node(f"{domain}_specialist", make_specialist(domain))
        builder.add_edge(f"{domain}_specialist", "synthesize")
    builder.add_node("synthesize", synthesize)
    builder.add_node("dispatch", dispatch)
    builder.add_edge(START, "route")
    builder.add_conditional_edges("route", fan_out, [*specialists, "synthesize"])
    builder.add_edge("synthesize", "dispatch")
    builder.add_edge("dispatch", END)
    return builder.compile(name="router")
