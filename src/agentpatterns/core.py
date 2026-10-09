"""Shared building blocks of all patterns.

The central idea of this library is the **agent contract**. Every pattern
factory returns a compiled LangGraph graph that behaves like a
`langchain.agents.create_agent` graph:

    input:  {"messages": [...]}
    output: {"messages": [..., AIMessage(final answer)], "structured_response": <parsed> | absent}

Because of that, patterns compose freely: a router can route to a
supervisor, a supervisor can delegate to a swarm, an orchestrator's worker can
be a pipeline - and every pattern can be embedded in your own `StateGraph`
(`agent_as_node`) or exposed as a tool (`agent_as_tool`).
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain.tools import ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph.message import add_messages
from langgraph.types import RetryPolicy, TimeoutPolicy
from pydantic import BaseModel

from agentpatterns.messaging import is_agent_message

__all__ = [
    "AgentResult",
    "AgentSpec",
    "PatternInput",
    "PatternOutput",
    "PatternState",
    "StructuredOutputMethod",
    "agent_as_node",
    "agent_as_tool",
    "final_text",
    "invoke_agent",
    "make_serializer",
    "merge_dicts",
    "results_block",
    "structured_llm",
    "to_text",
]


# ------------------------------------------------------------------- state
class PatternInput(TypedDict):
    messages: list[AnyMessage]


class PatternOutput(TypedDict):
    messages: list[AnyMessage]
    structured_response: NotRequired[Any]


class PatternState(TypedDict):
    """Base state of every pattern graph (compatible with `create_agent`'s `AgentState`)."""

    messages: Annotated[list[AnyMessage], add_messages]
    structured_response: NotRequired[Any]


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """Reducer that merges dict updates from parallel branches."""
    return {**(left or {}), **(right or {})}


# ------------------------------------------------------------ agent specs
@dataclass
class AgentSpec:
    """A named, described participant of a pattern.

    `agent` is anything that follows the agent contract: a `create_agent`
    graph, any pattern graph of this library, or your own compiled graph with
    a `messages` key.
    """

    name: str
    description: str
    agent: Runnable

    def __post_init__(self) -> None:
        if not self.name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Agent name {self.name!r} must be alphanumeric (plus '_' and '-').")


@dataclass
class AgentResult:
    """Normalized result of invoking an agent-contract runnable."""

    name: str
    text: str
    structured: Any = None
    messages: list[BaseMessage] = field(default_factory=list)


def to_text(value: Any) -> str:
    """Serialize structured values for prompts and tool results."""
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=lambda o: o.model_dump() if isinstance(o, BaseModel) else str(o))
    return "" if value is None else str(value)


def final_text(result: dict[str, Any]) -> str:
    """The answer of an agent run: structured response (as JSON) or the last message's text."""
    if result.get("structured_response") is not None:
        return to_text(result["structured_response"])
    messages = result.get("messages") or []
    return messages[-1].text if messages else ""


def _to_messages(task: str | Sequence[BaseMessage]) -> list[BaseMessage]:
    return [HumanMessage(task)] if isinstance(task, str) else list(task)


def _result(name: str, output: dict[str, Any]) -> AgentResult:
    return AgentResult(
        name=name,
        text=final_text(output),
        structured=output.get("structured_response"),
        messages=list(output.get("messages") or []),
    )


def invoke_agent(
    spec: AgentSpec, task: str | Sequence[BaseMessage], config: RunnableConfig | None = None
) -> AgentResult:
    """Invoke an agent with a task (string -> single human message)."""
    return _result(spec.name, spec.agent.invoke({"messages": _to_messages(task)}, config))


async def ainvoke_agent(
    spec: AgentSpec, task: str | Sequence[BaseMessage], config: RunnableConfig | None = None
) -> AgentResult:
    return _result(spec.name, await spec.agent.ainvoke({"messages": _to_messages(task)}, config))


# ----------------------------------------------------------- composition
InputMode = Literal["task", "fork"]


def _conversation_only(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Parent history without tool traffic and messages from other agents (safe to forward to another agent)."""
    return [
        m
        for m in messages
        if (isinstance(m, HumanMessage) and not is_agent_message(m))
        or (isinstance(m, AIMessage) and not m.tool_calls and m.text)
    ]


def agent_as_tool(
    spec: AgentSpec,
    *,
    name: str | None = None,
    description: str | None = None,
    input_mode: InputMode = "task",
) -> BaseTool:
    """Expose an agent (or any pattern graph) as a tool - the core of the subagents pattern.

    The tool takes one argument, `task`. `input_mode` controls the subagent's context:
        "task" (default, isolated) - the subagent only sees the task text.
        "fork" - it also receives the caller's conversation (without tool traffic).
    The tool returns the subagent's structured response as JSON, or its final text.
    """

    def _messages(task: str, runtime: ToolRuntime[Any, Any] | None) -> list[BaseMessage]:
        forked = input_mode == "fork" and runtime is not None and runtime.state
        history = _conversation_only(runtime.state.get("messages", [])) if forked else []
        return [*history, HumanMessage(task)]

    def run(task: str, runtime: ToolRuntime[Any, Any]) -> str:
        """Delegate a task to the agent.

        Args:
            task: Self-contained description of the task, including all context the agent needs.
        """
        return invoke_agent(spec, _messages(task, runtime)).text

    async def arun(task: str, runtime: ToolRuntime[Any, Any]) -> str:
        """Delegate a task to the agent.

        Args:
            task: Self-contained description of the task, including all context the agent needs.
        """
        return (await ainvoke_agent(spec, _messages(task, runtime))).text

    return StructuredTool.from_function(
        func=run,
        coroutine=arun,
        name=name or spec.name,
        description=description or spec.description,
        parse_docstring=True,
    )


def agent_as_node(
    agent: Runnable,
    *,
    input: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    output: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
    name: str | None = None,
) -> Runnable:
    """Adapt an agent-contract graph to a parent graph with a *different* state schema.

    `input(parent_state) -> agent input`, `output(agent_result, parent_state) -> parent update`.
    If parent and agent share the `messages` key you can instead pass the
    compiled graph directly to `builder.add_node(...)`.

    The node calls `agent` directly, so LangGraph still detects it as a subgraph:
    `get_graph(xray=True)` draws its inner nodes and `get_state(subgraphs=True)`
    shows its state. The patterns run their participants through this function.
    """
    to_input = input or (lambda state: {"messages": state["messages"]})
    to_output = output or (lambda result, state: {"messages": result["messages"][-1:]})

    def sync(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        return to_output(agent.invoke(to_input(state), config), state)

    async def asynchronous(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        return to_output(await agent.ainvoke(to_input(state), config), state)

    return RunnableLambda(sync, afunc=asynchronous, name=name or getattr(agent, "name", None) or "agent")


# ---------------------------------------------------------- persistence
def make_serializer(*types: type) -> Any:
    """Checkpoint serializer that allow-lists this library's and your own state types.

    LangGraph >= 1.2 warns (and will refuse in strict mode) when a checkpoint
    contains unregistered classes. Use it with any checkpointer:
    `InMemorySaver(serde=make_serializer(MySchema))`, `PostgresSaver(conn, serde=...)`.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(allowed_msgpack_modules=[AgentResult, *types])


# -------------------------------------------------------------- internals
RetryPolicies = RetryPolicy | Sequence[RetryPolicy] | None
"""`retry_policy` of the graph factories: how a node that runs an agent or a model is
retried when it raises (LangGraph's `RetryPolicy`; with several, the first that matches
the error applies). `None`: no retries."""

StepTimeout = float | timedelta | TimeoutPolicy | None
"""`timeout` of the graph factories: how long one attempt of a node that runs an agent or
a model may take, in seconds, as a `timedelta`, or as LangGraph's `TimeoutPolicy` (which
can also cap idle time). An attempt over it is cancelled with `NodeTimeoutError`, which
`retry_policy` retries by default. Async runs only: LangGraph cannot cancel sync code, so
`invoke` refuses a step that has a timeout (`ValueError`). `None`: no timeout."""


def step_policies(retry_policy: RetryPolicies, timeout: StepTimeout) -> dict[str, Any]:
    """`add_node` keyword arguments for a node that runs an agent or a model."""
    return {"retry_policy": retry_policy, "timeout": timeout}


def dual_node(
    sync: Callable[[Any, RunnableConfig], Any],
    asynchronous: Callable[[Any, RunnableConfig], Awaitable[Any]],
    name: str,
) -> RunnableLambda:
    """A graph node with native sync and async implementations."""
    return RunnableLambda(sync, afunc=asynchronous, name=name)


def spec_node(
    spec: AgentSpec,
    task: Callable[[Any], str | Sequence[BaseMessage]],
    update: Callable[[AgentResult, Any], dict[str, Any]],
    *,
    name: str | None = None,
) -> Runnable:
    """Graph node that runs `spec` on `task(state)` and returns `update(result, state)`.

    Built on `agent_as_node`, so the participant shows up as a subgraph of the pattern.
    """
    return agent_as_node(
        spec.agent,
        input=lambda state: {"messages": _to_messages(task(state))},
        output=lambda output, state: update(_result(spec.name, output), state),
        name=name or spec.name,
    )


StructuredOutputMethod = Literal["auto", "json_schema", "function_calling"] | str
"""How pattern-internal LLM calls produce structured output (routing, plans, verdicts, answers).

"auto" (default) uses the provider's native structured output (`method="json_schema"`)
when the model's profile declares support - the same rule `create_agent` applies to
`response_format` - and the provider default (usually forced tool calling) otherwise.
Any other value is passed to `model.with_structured_output(method=...)` as is.
"""


def structured_llm(model: BaseChatModel, schema: type | None, method: StructuredOutputMethod = "auto") -> Runnable:
    """`model.with_structured_output(schema, method=...)` (see `StructuredOutputMethod`) or the plain model."""
    if schema is None:
        return model
    if method == "auto":
        profile = getattr(model, "profile", None) or {}
        method = "json_schema" if profile.get("structured_output") else None
    return model.with_structured_output(schema, method=method) if method else model.with_structured_output(schema)


def llm_output(value: Any, name: str) -> tuple[AIMessage, Any]:
    """Turn a (structured) model output into the final AIMessage + structured value."""
    if isinstance(value, BaseMessage):
        return AIMessage(content=value.text, name=name), None
    return AIMessage(content=to_text(value), name=name), value


def catalog(specs: Sequence[AgentSpec]) -> str:
    return "\n".join(f"- {s.name}: {s.description}" for s in specs)


def literal_of(names: Sequence[str]) -> Any:
    """`Literal[...]` over runtime values (enum constraint in structured output schemas)."""
    return Literal[tuple(names)]  # type: ignore[valid-type]


def results_block(results: Sequence[AgentResult | dict[str, Any]], header: str = "Results") -> str:
    """Render agent results or orchestrator `results` entries as a prompt block."""
    lines = []
    for r in results:
        if isinstance(r, AgentResult):
            lines.append(f"[{r.name}]\n{r.text}")
        else:
            label = r.get("agent") or r.get("worker") or "result"
            task = f" (task: {r['task']})" if r.get("task") else ""
            lines.append(f"[{label}]{task}\n{r.get('output', '')}")
    return f"{header}:\n\n" + ("\n\n".join(lines) if lines else "(none)")
