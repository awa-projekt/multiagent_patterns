"""Supervisor with subagents as tools - the multi-agent default recommended by LangChain v1.

The supervisor is a regular `create_agent`. Every subagent is exposed as a tool;
the supervisor decides whom to call with which task (several in parallel if
useful) and composes the final answer. Subagents are stateless and start
with a clean context (isolation), unless `input_mode="fork"`.

    supervisor = create_supervisor(
        model,
        [AgentSpec("researcher", "Finds facts", research_agent),
         AgentSpec("writer", "Writes texts", writer_agent)],
        system_prompt="You coordinate a research team...",
        response_format=Answer,
    )

`delegation="task_tool"` exposes ONE `task(agent_name, description)` tool
instead of one tool per subagent (convenient for many / team-owned agents).
`max_calls_per_agent` caps how often each subagent may be called per run.
Replaces the unmaintained `langgraph-supervisor` package.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import PrivateStateAttr
from langchain.tools import ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from agentpatterns.core import (
    AgentSpec,
    InputMode,
    _conversation_only,
    agent_as_tool,
    ainvoke_agent,
    catalog,
    invoke_agent,
    literal_of,
)
from agentpatterns.limits import OnLimit, loop_limits

__all__ = ["DelegationLimitMiddleware", "create_supervisor", "create_task_tool"]

Delegation = Literal["tool_per_agent", "task_tool"]


def create_task_tool(subagents: Sequence[AgentSpec], *, input_mode: InputMode = "task") -> BaseTool:
    """Single dispatch tool `task(agent_name, description)` over a registry of subagents."""
    registry = {s.name: s for s in subagents}

    def _messages(description: str, runtime: ToolRuntime[Any, Any] | None) -> list:
        history = _conversation_only(runtime.state.get("messages", [])) if (input_mode == "fork" and runtime) else []
        return [*history, HumanMessage(description)]

    def run(agent_name: str, description: str, runtime: ToolRuntime[Any, Any]) -> str:
        """Launch a subagent.

        Args:
            agent_name: Subagent that should do the task.
            description: Self-contained task description with all needed context.
        """
        return invoke_agent(registry[agent_name], _messages(description, runtime)).text

    async def arun(agent_name: str, description: str, runtime: ToolRuntime[Any, Any]) -> str:
        """Launch a subagent.

        Args:
            agent_name: Subagent that should do the task.
            description: Self-contained task description with all needed context.
        """
        return (await ainvoke_agent(registry[agent_name], _messages(description, runtime))).text

    # Enum constraint on agent_name: annotate with a Literal of the registered names.
    for fn in (run, arun):
        fn.__annotations__ = {
            "agent_name": literal_of(list(registry)),
            "description": str,
            "runtime": ToolRuntime[Any, Any],
            "return": str,
        }
    return StructuredTool.from_function(
        func=run,
        coroutine=arun,
        name="task",
        description="Launch a subagent for a self-contained task. Available agents:\n" + catalog(subagents),
        parse_docstring=True,
    )


class DelegationLimitState(AgentState):
    delegation_counts: NotRequired[Annotated[dict[str, int], UntrackedValue, PrivateStateAttr]]
    """Calls per subagent in the current run (untracked = reset per run, like the native limits)."""


class DelegationLimitMiddleware(AgentMiddleware):
    """Caps how often each subagent may be called per run.

    Soft limit: a call over the limit is not executed; it gets an error
    `ToolMessage` and the supervisor continues with the other subagents. Like
    `ToolCallLimitMiddleware`, it checks the calls of each model turn one by one,
    so parallel calls are counted correctly. Works for both delegation styles: the
    tool name is the subagent ("tool_per_agent") or its `agent_name` argument is
    ("task_tool").
    """

    state_schema = DelegationLimitState

    def __init__(self, limits: Mapping[str, int], *, delegation: Delegation = "tool_per_agent") -> None:
        super().__init__()
        self.limits = dict(limits)
        self.delegation = delegation

    def _target(self, call: ToolCall) -> str | None:
        if self.delegation == "task_tool":
            return call["args"].get("agent_name") if call["name"] == "task" else None
        return call["name"]

    def after_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        messages = state["messages"]
        index = next((i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], AIMessage)), None)
        if index is None or not messages[index].tool_calls:
            return None
        answered = {m.tool_call_id for m in messages[index + 1 :] if isinstance(m, ToolMessage)}
        counts = dict(state.get("delegation_counts") or {})
        refused = []
        for call in messages[index].tool_calls:
            agent = self._target(call)
            if agent not in self.limits or call["id"] in answered:
                continue
            if counts.get(agent, 0) >= self.limits[agent]:
                refused.append(
                    ToolMessage(
                        f"Delegation limit reached: '{agent}' may be called at most {self.limits[agent]} "
                        "times per run. Do not call it again; continue with the results you have.",
                        tool_call_id=call["id"],
                        name=call["name"],
                        status="error",
                    )
                )
            else:
                counts[agent] = counts.get(agent, 0) + 1
        # Calls answered here are skipped by the agent's tool node.
        return {"delegation_counts": counts, **({"messages": refused} if refused else {})}

    async def aafter_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        return self.after_model(state, runtime)


def per_agent_limits(limit: int | Mapping[str, int] | None, names: Sequence[str]) -> dict[str, int]:
    """Normalize `max_calls_per_agent`: one int for all subagents, or a mapping name -> limit."""
    if limit is None:
        return {}
    if not isinstance(limit, Mapping):
        return dict.fromkeys(names, limit)
    if unknown := set(limit) - set(names):
        raise ValueError(f"max_calls_per_agent names unknown subagents: {sorted(unknown)}")
    return dict(limit)


def create_supervisor(
    model: BaseChatModel | str,
    subagents: Sequence[AgentSpec],
    *,
    system_prompt: str,
    tools: Sequence[BaseTool] = (),
    response_format: Any = None,
    delegation: Delegation = "tool_per_agent",
    input_mode: InputMode = "task",
    max_model_calls: int | None = 25,
    max_tool_calls: int | None = None,
    on_limit: OnLimit = "error",
    max_calls_per_agent: int | Mapping[str, int] | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    checkpointer: Checkpointer = None,
    name: str = "supervisor",
    **agent_kwargs: Any,
) -> CompiledStateGraph:
    """Create a supervisor agent that delegates to subagents via tool calls.

    Args:
        model: Supervisor model.
        subagents: Subagents (name + description are the routing signal - write them carefully).
        system_prompt: Supervisor instructions (catalog of subagents is appended).
        tools: Extra tools the supervisor may use itself.
        response_format: Schema of the final answer (`structured_response`).
        delegation: "tool_per_agent" (one tool per subagent) or "task_tool" (single dispatch tool).
        input_mode: "task" (isolated subagent context) or "fork" (subagent sees the conversation).
        max_model_calls: Cap on supervisor model calls per run (subagents have their own limits).
        max_tool_calls: Soft cap on the supervisor's tool calls (delegations included) per run.
        on_limit: "error" (raise) or "end" (stop with a final message) at `max_model_calls`.
        max_calls_per_agent: How often each subagent may be called per run: one
            int for all, or `{"billing": 2, ...}`. Calls over the limit are refused
            (see `DelegationLimitMiddleware`).
        middleware: Extra middleware for the supervisor (e.g. HumanInTheLoopMiddleware).
        checkpointer: Checkpoints the supervisor; only the outermost graph needs one.
        name: Graph name.
        **agent_kwargs: Passed to `create_agent` (store, ...).
    """
    if delegation == "tool_per_agent":
        delegation_tools = [agent_as_tool(s, input_mode=input_mode) for s in subagents]
    else:
        delegation_tools = [create_task_tool(subagents, input_mode=input_mode)]
    guards = loop_limits(max_model_calls=max_model_calls, max_tool_calls=max_tool_calls, on_limit=on_limit)
    if delegation_limits := per_agent_limits(max_calls_per_agent, [s.name for s in subagents]):
        # after_model hooks run last-to-first: listed first, it skips calls the tool cap already refused.
        guards.insert(0, DelegationLimitMiddleware(delegation_limits, delegation=delegation))
    return create_agent(
        model,
        tools=[*delegation_tools, *tools],
        system_prompt=f"{system_prompt}\n\nSubagents you can delegate to:\n{catalog(subagents)}",
        response_format=response_format,
        middleware=[*guards, *middleware],
        checkpointer=checkpointer,
        name=name,
        **agent_kwargs,
    )
