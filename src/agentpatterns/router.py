"""Router - classify once, dispatch to one or more specialised agents, synthesize.

A routing step (an LLM with structured output, or your own function) decides
which agents should handle the input and writes a self-contained task for
each. Selected agents run **in parallel** (`Send`), then a synthesizer merges
their results into one answer (optionally structured).

    router = create_router(
        model,
        routes=[AgentSpec("billing", "Invoices and refunds", billing_agent),
                AgentSpec("tech", "Errors and outages", tech_agent)],
        response_format=Answer,
    )

The router itself is stateless; for multi-turn chats wrap it as a tool of a
conversational agent (`agent_as_tool(AgentSpec("router", ..., router))`).
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer, Send
from pydantic import BaseModel, Field, create_model

from agentpatterns.core import (
    AgentResult,
    AgentSpec,
    PatternInput,
    PatternState,
    RetryPolicies,
    StepTimeout,
    StructuredOutputMethod,
    catalog,
    dual_node,
    literal_of,
    llm_output,
    results_block,
    spec_node,
    step_policies,
    structured_llm,
)

__all__ = ["RouteTask", "create_router"]

DEFAULT_ROUTER_PROMPT = (
    "You are a router. Decide which of the available agents should work on the user's request. "
    "Select every agent that is needed (possibly none) and give each a self-contained task that "
    "includes all information it needs."
)
DEFAULT_SYNTHESIZER_PROMPT = (
    "Combine the results of the agents below into one coherent, complete answer to the user's request."
)


class RouteTask(TypedDict):
    agent: str
    task: str


class RouterState(PatternState):
    routes: NotRequired[list[RouteTask]]
    route_results: Annotated[list[AgentResult], operator.add]


class RouterOutput(PatternInput):
    structured_response: NotRequired[Any]
    routes: NotRequired[list[RouteTask]]


def _routing_schema(names: Sequence[str], allow_multiple: bool) -> type[BaseModel]:
    route = create_model(
        "Route",
        agent=(literal_of(names), Field(description="Name of the agent that should handle this part.")),
        task=(str, Field(description="Self-contained task for the agent.")),
    )
    return create_model(
        "RoutingDecision",
        __doc__="Agents that must work on the request (empty list if none).",
        routes=(
            list[route],
            Field(
                default_factory=list,
                max_length=None if allow_multiple else 1,
                description="One entry per selected agent.",
            ),
        ),
    )


def create_router(
    model: BaseChatModel,
    routes: Sequence[AgentSpec],
    *,
    system_prompt: str = DEFAULT_ROUTER_PROMPT,
    allow_multiple: bool = True,
    route_fn: Callable[[dict[str, Any]], list[RouteTask]] | None = None,
    synthesizer_prompt: str = DEFAULT_SYNTHESIZER_PROMPT,
    response_format: type | None = None,
    synthesize: bool | None = None,
    structured_output_method: StructuredOutputMethod = "auto",
    retry_policy: RetryPolicies = None,
    timeout: StepTimeout = None,
    checkpointer: Checkpointer = None,
    name: str = "router",
) -> CompiledStateGraph:
    """Build a router following the agent contract.

    Args:
        model: Model for routing and synthesis.
        routes: Candidate agents (name, description, agent-contract runnable).
        system_prompt: Routing instructions; the agent catalog is appended automatically.
        allow_multiple: Allow fan-out to several agents (else at most one).
        route_fn: Deterministic routing instead of an LLM: `fn(state) -> [{"agent", "task"}]`.
        synthesizer_prompt: Instructions for merging results.
        response_format: Schema of the final answer (`structured_response`).
        synthesize: Force (True) or skip (False) the synthesis LLM call. Default:
            skip only if exactly one agent answered and no `response_format` is set.
        structured_output_method: How routing and synthesis produce structured output
            (see `StructuredOutputMethod`; "auto" = native where supported).
        retry_policy: Retries of the steps that run an agent or a model when they raise
            (see `RetryPolicies`).
        timeout: Time limit of one attempt of such a step; async runs only (see
            `StepTimeout`).
        checkpointer: Checkpoints the graph; only the outermost graph needs one.
        name: Graph name.
    """
    names = [r.name for r in routes]
    specs = {r.name: r for r in routes}
    reserved = {"route", "synthesize"}
    if clash := reserved.intersection(names):
        raise ValueError(f"Route names {clash} are reserved.")
    router_llm = structured_llm(model, _routing_schema(names, allow_multiple), structured_output_method)
    synth_llm = structured_llm(model, response_format, structured_output_method)
    router_system = f"{system_prompt}\n\nAvailable agents:\n{catalog(routes)}"

    def decide(result: Any) -> dict[str, Any]:
        tasks = [{"agent": r.agent, "task": r.task} for r in result.routes]
        return {"routes": tasks if allow_multiple else tasks[:1]}

    def route_sync(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        if route_fn is not None:
            return {"routes": route_fn(state)}
        return decide(router_llm.invoke([SystemMessage(router_system), *state["messages"]], config))

    async def route_async(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        if route_fn is not None:
            return {"routes": route_fn(state)}
        return decide(await router_llm.ainvoke([SystemMessage(router_system), *state["messages"]], config))

    def fan_out(state: dict[str, Any]) -> list[Send] | str:
        routes_ = [r for r in state.get("routes", []) if r["agent"] in specs]
        if not routes_:
            return "synthesize"
        return [Send(r["agent"], {"task": r["task"]}) for r in routes_]

    def synthesis_messages(state: dict[str, Any]) -> list:
        results = state.get("route_results", [])
        return [
            SystemMessage(synthesizer_prompt),
            *state["messages"],
            HumanMessage(results_block(results, "Results of the agents")),
        ]

    def needs_llm(state: dict[str, Any]) -> bool:
        if synthesize is not None:
            return synthesize
        return not (len(state.get("route_results", [])) == 1 and response_format is None)

    def finish(value: Any) -> dict[str, Any]:
        message, structured = llm_output(value, name)
        update: dict[str, Any] = {"messages": [message]}
        if response_format is not None:
            update["structured_response"] = structured
        return update

    def synth_sync(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        if not needs_llm(state):
            only = state["route_results"][0]
            return {"messages": only.messages[-1:]} if only.messages else finish(only.text)
        return finish(synth_llm.invoke(synthesis_messages(state), config))

    async def synth_async(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        if not needs_llm(state):
            return synth_sync(state, config)
        return finish(await synth_llm.ainvoke(synthesis_messages(state), config))

    policies = step_policies(retry_policy, timeout)
    builder = StateGraph(RouterState, input_schema=PatternInput, output_schema=RouterOutput)
    builder.add_node("route", dual_node(route_sync, route_async, "route"), **policies)
    for spec in routes:
        branch = spec_node(spec, lambda state: state["task"], lambda result, state: {"route_results": [result]})
        builder.add_node(spec.name, branch, **policies)
        builder.add_edge(spec.name, "synthesize")
    builder.add_node("synthesize", dual_node(synth_sync, synth_async, "synthesize"), **policies)
    builder.add_edge(START, "route")
    builder.add_conditional_edges("route", fan_out, [*names, "synthesize"])
    builder.add_edge("synthesize", END)
    return builder.compile(name=name, checkpointer=checkpointer)
