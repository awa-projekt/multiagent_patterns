"""Swarm / network - peer agents hand control to each other.

Each agent is a `create_agent` node in one graph and gets `transfer_to_<peer>`
tools. A handoff tool returns `Command(goto=peer, graph=Command.PARENT)`, so
control moves to the peer *inside the same run* - no central coordinator. The
active agent is kept in state (`active_agent`), so with a checkpointer the next
user turn continues with whoever held control last.

    swarm = create_swarm(
        model,
        [SwarmAgent("triage", "Front desk", "You greet and route...", handoffs=["billing", "tech"]),
         SwarmAgent("billing", "Billing expert", "You solve billing...", tools=[get_invoice]),
         SwarmAgent("tech", "Tech expert", "You solve tech issues...", tools=[check_status])],
        default_agent="triage",
        checkpointer=InMemorySaver(),
    )

Context engineering (`history`):
    "full"         - forward the whole conversation incl. the previous agent's tool
                     calls (what `langgraph-swarm` does; peers see each other's work).
    "handoff_only" - forward only the handoff tool call + its ToolMessage (with the
                     note). Leaner context, but the next agent only knows what the note says.

Limits (all per run): `max_handoffs` caps handoffs in total,
`SwarmAgent.max_activations` how often one agent may take control, and
`max_model_calls` / `max_tool_calls` each agent's loop *per activation*
(override per agent on `SwarmAgent`). Worst case: about
(max_handoffs + 1) x max_model_calls model calls; use `RunBudget` for one total.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer, Command

from agentpatterns.core import RetryPolicies, StepTimeout, step_policies
from agentpatterns.limits import OnLimit, loop_limits

__all__ = ["SwarmAgent", "SwarmState", "create_handoff_tool", "create_swarm"]

History = Literal["full", "handoff_only"]


class SwarmState(AgentState):
    active_agent: NotRequired[str]
    handoff_count: NotRequired[int]
    activations: NotRequired[dict[str, int]]
    """How often each agent took control in the current run."""


@dataclass
class SwarmAgent:
    """Definition of one peer in the swarm."""

    name: str
    description: str
    system_prompt: str
    tools: Sequence[BaseTool] = ()
    handoffs: Sequence[str] | None = None
    """Peers this agent may transfer to (`None` = all other agents)."""
    model: BaseChatModel | None = None
    max_activations: int | None = None
    """How often this agent may take control per run (starting counts); further handoffs to it are refused."""
    max_model_calls: int | None = None
    """This agent's model calls per activation (`None` = the swarm's `max_model_calls`)."""
    max_tool_calls: int | None = None
    """This agent's tool calls per activation (`None` = the swarm's `max_tool_calls`)."""


def create_handoff_tool(
    agent_name: str,
    *,
    description: str | None = None,
    history: History = "full",
    max_handoffs: int | None = None,
    max_activations: int | None = None,
    name: str | None = None,
) -> BaseTool:
    """Tool that transfers control to `agent_name` (a node of the parent graph).

    The tool takes a `note` for the receiving agent. When `max_handoffs` is
    reached, or `agent_name` already had control `max_activations` times in this
    run, the tool refuses and tells the model to go on without it (prevents
    ping-pong loops).
    """
    tool_name = name or f"transfer_to_{agent_name}"

    def handoff(note: str, runtime: ToolRuntime[Any, Any]) -> Command | str:
        """Transfer the conversation to a colleague.

        Args:
            note: What the colleague needs to know and do next.
        """
        count = runtime.state.get("handoff_count", 0) or 0
        if max_handoffs is not None and count >= max_handoffs:
            return f"Handoff limit ({max_handoffs}) reached - finish the task yourself."
        activations = dict(runtime.state.get("activations") or {})
        if max_activations is not None and activations.get(agent_name, 0) >= max_activations:
            return (
                f"{agent_name} already had control {max_activations} times in this run - "
                "finish the task yourself or transfer to someone else."
            )
        activations[agent_name] = activations.get(agent_name, 0) + 1
        tool_message = ToolMessage(
            content=f"Transferred to {agent_name}. Note: {note}", name=tool_name, tool_call_id=runtime.tool_call_id
        )
        messages = runtime.state["messages"]
        # The AIMessage that called this tool must stay paired with its ToolMessage.
        forwarded = [*messages, tool_message] if history == "full" else [messages[-1], tool_message]
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={
                "messages": forwarded,
                "active_agent": agent_name,
                "handoff_count": count + 1,
                "activations": activations,
            },
        )

    return StructuredTool.from_function(
        func=handoff,
        name=tool_name,
        description=description or f"Transfer the conversation to {agent_name}.",
        parse_docstring=True,
        metadata={"handoff_destination": agent_name},
    )


def create_swarm(
    model: BaseChatModel,
    agents: Sequence[SwarmAgent],
    *,
    default_agent: str | None = None,
    response_format: Any = None,
    history: History = "full",
    max_handoffs: int | None = 8,
    max_model_calls: int | None = 25,
    max_tool_calls: int | None = None,
    on_limit: OnLimit = "error",
    middleware: Sequence[AgentMiddleware] = (),
    retry_policy: RetryPolicies = None,
    timeout: StepTimeout = None,
    checkpointer: Checkpointer = None,
    name: str = "swarm",
) -> CompiledStateGraph:
    """Build a swarm of peer agents that hand off to each other.

    Args:
        model: Default model for all agents.
        agents: Peer definitions (names must be unique).
        default_agent: Agent that starts when no agent is active (default: first).
        response_format: Structured final answer; any agent may produce it.
        history: What a handoff forwards ("full" or "handoff_only", see module docs).
        max_handoffs: Maximum handoffs per run (loop protection).
        max_model_calls: Model calls per agent activation (override per `SwarmAgent`).
        max_tool_calls: Soft cap on tool calls per agent activation (override per `SwarmAgent`).
        on_limit: "error" (raise) or "end" (stop the swarm with a final message) at `max_model_calls`.
        middleware: Middleware applied to every agent.
        retry_policy: Retries of an agent's turn when it raises (see `RetryPolicies`).
        timeout: Time limit of one attempt of an agent's turn; async runs only (see
            `StepTimeout`).
        checkpointer: Enables multi-turn conversations that resume with the active agent.
        name: Graph name.
    """
    names = [a.name for a in agents]
    if len(set(names)) != len(names):
        raise ValueError("Agent names must be unique.")
    default = default_agent or names[0]
    by_name = {a.name: a for a in agents}
    handoff_tools = {
        a.name: create_handoff_tool(
            a.name,
            description=f"Transfer to {a.name}: {a.description}",
            history=history,
            max_handoffs=max_handoffs,
            max_activations=a.max_activations,
        )
        for a in agents
    }

    policies = step_policies(retry_policy, timeout)
    builder = StateGraph(SwarmState)
    for agent in agents:
        targets = [t for t in (agent.handoffs if agent.handoffs is not None else names) if t != agent.name]
        roster = "\n".join(f"- {t}: {by_name[t].description}" for t in targets)
        prompt = agent.system_prompt + (f"\n\nColleagues you can transfer to:\n{roster}" if targets else "")
        guards = loop_limits(
            max_model_calls=agent.max_model_calls if agent.max_model_calls is not None else max_model_calls,
            max_tool_calls=agent.max_tool_calls if agent.max_tool_calls is not None else max_tool_calls,
            on_limit=on_limit,
        )
        graph = create_agent(
            agent.model or model,
            tools=[*agent.tools, *(handoff_tools[t] for t in targets)],
            system_prompt=prompt,
            response_format=response_format,
            state_schema=SwarmState,
            middleware=[*guards, *middleware],
            name=agent.name,
        )
        # Agents are subgraph nodes sharing the `messages` channel with the swarm.
        builder.add_node(agent.name, graph, destinations=tuple(targets), **policies)

    def first_agent(state: SwarmState) -> str:
        return state.get("active_agent") or default

    def start(state: SwarmState) -> dict[str, Any]:
        # Handoff and activation budgets are per run; the starting agent's turn counts.
        return {"handoff_count": 0, "activations": {first_agent(state): 1}}

    builder.add_node("swarm_entry", start)
    builder.add_edge(START, "swarm_entry")
    builder.add_conditional_edges("swarm_entry", first_agent, names)
    # No outgoing edges: an agent that answers without handing off ends the run.
    return builder.compile(name=name, checkpointer=checkpointer)
