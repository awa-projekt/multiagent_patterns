"""State machine - handoffs *within one agent* via middleware.

The variant of handoffs recommended by the LangChain docs for most use cases.
One agent moves through named steps; each step has its own system prompt,
tools and allowed transitions. Tools switch steps by updating
`current_step` (auto-generated `go_to_<step>` tools, or your own tools using
`transition(...)`). Only steps marked `final=True` may produce the structured
response.

    agent = create_state_machine_agent(
        model,
        steps=[
            Step("collect", "Ask for the order number...", tools=[lookup_order], transitions=["resolve"]),
            Step("resolve", "Order: {order}. Solve the issue...", tools=[refund], transitions=["respond"]),
            Step("respond", "Write the final answer.", final=True),
        ],
        response_format=Answer,
        checkpointer=InMemorySaver(),   # keeps the current step across turns
    )

`StateMachineMiddleware` can also be added to any existing `create_agent`.
Prompts are templates: `{key}` is replaced by the state value `key`.

`Step(max_visits=N)` caps how often a run may enter a step (the step it starts
in counts). Transitions beyond that are refused and the agent stays where it is;
`max_model_calls` caps the whole loop.
"""

from __future__ import annotations

import dataclasses
import string
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Checkpointer, Command
from pydantic import BaseModel

from agentpatterns.core import to_text
from agentpatterns.limits import OnLimit, loop_limits

__all__ = [
    "StateMachineMiddleware",
    "StateMachineState",
    "Step",
    "create_state_machine_agent",
    "create_transition_tool",
    "transition",
]


class StateMachineState(AgentState):
    current_step: NotRequired[str]
    step_visits: NotRequired[dict[str, int]]
    """How often each step was entered in the current run."""


@dataclass
class Step:
    """One state of the machine."""

    name: str
    system_prompt: str | Callable[[dict[str, Any]], str]
    tools: Sequence[BaseTool] = ()
    transitions: Sequence[str] = ()
    """Steps reachable via auto-generated `go_to_<step>` tools."""
    final: bool = False
    """Structured output (`response_format`) is only offered in final steps."""
    tool_filter: Callable[[dict[str, Any], list[BaseTool]], list[BaseTool]] | None = None
    """Optional per-state narrowing of this step's tools."""
    max_visits: int | None = None
    """How often a run may enter this step (the step it starts in counts); further transitions are refused."""


def transition(target: str, tool_call_id: str, message: str | None = None, **updates: Any) -> Command:
    """Command for custom tools: move to `target` (and update other state keys)."""
    return Command(
        update={
            "messages": [ToolMessage(message or f"Moved to step '{target}'.", tool_call_id=tool_call_id)],
            "current_step": target,
            **updates,
        }
    )


def create_transition_tool(target: str, description: str | None = None) -> BaseTool:
    """Auto-generated `go_to_<target>` tool with a `reason` argument."""

    def go(reason: str, runtime: ToolRuntime[Any, Any]) -> Command:
        """Move the conversation to another step.

        Args:
            reason: Why the transition happens (kept in the conversation).
        """
        return transition(target, runtime.tool_call_id, f"Moved to step '{target}': {reason}")

    return StructuredTool.from_function(
        func=go,
        name=f"go_to_{target}",
        description=description or f"Move to the '{target}' step.",
        parse_docstring=True,
    )


def _render(template: str | Callable[[dict[str, Any]], str], state: dict[str, Any]) -> str:
    if callable(template):
        return template(state)
    values: defaultdict[str, str] = defaultdict(str)
    for key, value in state.items():
        if key != "messages":
            values[key] = to_text(value) if isinstance(value, (dict, list, BaseModel)) else str(value)
    return string.Formatter().vformat(template, (), values)


class StateMachineMiddleware(AgentMiddleware):
    """Applies the configuration of the current step before every model call."""

    state_schema = StateMachineState

    def __init__(self, steps: Sequence[Step], *, initial_step: str | None = None) -> None:
        super().__init__()
        self.steps = {s.name: s for s in steps}
        self.initial_step = initial_step or steps[0].name
        unknown = {t for s in steps for t in s.transitions if t not in self.steps}
        if unknown:
            raise ValueError(f"Unknown transition targets: {unknown}")
        self.transition_tools = {t: create_transition_tool(t) for s in steps for t in s.transitions}
        registered: dict[str, BaseTool] = {}
        for tool in [*(t for s in steps for t in s.tools), *self.transition_tools.values()]:
            registered.setdefault(tool.name, tool)
        # All tools are registered with the agent up front; `_configure` narrows them.
        self.tools = list(registered.values())
        self.go_to_targets = {tool.name: target for target, tool in self.transition_tools.items()}

    def _current(self, state: dict[str, Any]) -> str:
        return state.get("current_step") or self.initial_step

    def _configure(self, request: ModelRequest) -> ModelRequest:
        state = request.state
        step = self.steps[self._current(state)]
        tools = [*step.tools, *(self.transition_tools[t] for t in step.transitions)]
        if step.tool_filter is not None:
            tools = step.tool_filter(state, tools)
        overrides: dict[str, Any] = {
            "system_message": SystemMessage(_render(step.system_prompt, state)),
            "tools": tools,
        }
        if not step.final:
            overrides["response_format"] = None
        return request.override(**overrides)

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        return handler(self._configure(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        return await handler(self._configure(request))

    # ------------------------------------------------------------ visit limits
    def before_agent(self, state: dict[str, Any], runtime: Any) -> dict[str, Any]:
        return {"step_visits": {self._current(state): 1}}  # visits are counted per run

    async def abefore_agent(self, state: dict[str, Any], runtime: Any) -> dict[str, Any]:
        return self.before_agent(state, runtime)

    def _refusal(self, state: dict[str, Any], target: str) -> str | None:
        limit = self.steps[target].max_visits if target in self.steps else None
        if limit is None or (state.get("step_visits") or {}).get(target, 0) < limit:
            return None
        return (
            f"Transition to step '{target}' refused: it may be entered at most {limit} times per run. "
            f"You stay in step '{self._current(state)}' - finish from here."
        )

    def _refuse_go_to(self, request: ToolCallRequest) -> ToolMessage | None:
        """`go_to_<step>` has no side effects, so a refused transition is not executed at all."""
        target = self.go_to_targets.get(request.tool_call["name"])
        refusal = target and self._refusal(request.state, target)
        if not refusal:
            return None
        return ToolMessage(
            refusal, tool_call_id=request.tool_call["id"], name=request.tool_call["name"], status="error"
        )

    def _count_visit(self, request: ToolCallRequest, result: ToolMessage | Command) -> ToolMessage | Command:
        """Count entered steps; a custom tool's transition to a used-up step is undone (its work stays)."""
        if not isinstance(result, Command) or not isinstance(result.update, dict):
            return result
        state, target = request.state, result.update.get("current_step")
        if target is None or target == self._current(state):
            return result
        if refusal := self._refusal(state, target):
            update = {k: v for k, v in result.update.items() if k != "current_step"}
            update["messages"] = [
                m.model_copy(update={"content": f"{m.text}\n\n{refusal}"})
                if isinstance(m, ToolMessage) and m.tool_call_id == request.tool_call["id"]
                else m
                for m in update.get("messages", [])
            ]
            return dataclasses.replace(result, update=update)
        visits = dict(state.get("step_visits") or {})
        visits[target] = visits.get(target, 0) + 1
        return dataclasses.replace(result, update={**result.update, "step_visits": visits})

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]
    ) -> ToolMessage | Command:
        return self._refuse_go_to(request) or self._count_visit(request, handler(request))

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]
    ) -> ToolMessage | Command:
        return self._refuse_go_to(request) or self._count_visit(request, await handler(request))


def create_state_machine_agent(
    model: BaseChatModel | str,
    steps: Sequence[Step],
    *,
    initial_step: str | None = None,
    response_format: Any = None,
    state_schema: type | None = None,
    tools: Sequence[BaseTool] = (),
    max_model_calls: int | None = 25,
    max_tool_calls: int | None = None,
    on_limit: OnLimit = "error",
    middleware: Sequence[AgentMiddleware] = (),
    checkpointer: Checkpointer = None,
    name: str = "state_machine",
    **agent_kwargs: Any,
) -> CompiledStateGraph:
    """Create an agent driven by `StateMachineMiddleware`.

    `state_schema` may add keys your prompts/tools use (extend `StateMachineState`).
    A `checkpointer` keeps the current step across turns.
    `max_model_calls` / `max_tool_calls` / `on_limit` cap the loop across all
    steps (see `agentpatterns.limits`); `Step.max_visits` caps single steps.
    """
    guards = loop_limits(max_model_calls=max_model_calls, max_tool_calls=max_tool_calls, on_limit=on_limit)
    return create_agent(
        model,
        tools=list(tools),
        middleware=[*guards, StateMachineMiddleware(steps, initial_step=initial_step), *middleware],
        response_format=response_format,
        state_schema=state_schema,
        checkpointer=checkpointer,
        name=name,
        **agent_kwargs,
    )
