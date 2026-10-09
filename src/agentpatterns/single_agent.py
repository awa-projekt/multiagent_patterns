"""Single agent - the baseline every multi-agent design should be compared against.

A thin, opinionated wrapper around `langchain.agents.create_agent`: one model,
all tools, a tool-calling loop, optional structured output, plus guardrails
(model/tool call limits) that you want in production anyway.

    agent = create_single_agent(model, tools, system_prompt="...", response_format=Answer)
    agent.invoke({"messages": [("user", "...")]})["structured_response"]
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from agentpatterns.limits import OnLimit, loop_limits

__all__ = ["create_single_agent"]


def create_single_agent(
    model: BaseChatModel | str,
    tools: Sequence[BaseTool | Any],
    *,
    system_prompt: str | None = None,
    response_format: Any = None,
    max_model_calls: int | None = 25,
    max_tool_calls: int | None = None,
    on_limit: OnLimit = "error",
    middleware: Sequence[AgentMiddleware] = (),
    checkpointer: Checkpointer = None,
    name: str = "single_agent",
    **agent_kwargs: Any,
) -> CompiledStateGraph:
    """Create a tool-calling agent that follows the agent contract.

    Args:
        model: Chat model instance or `init_chat_model` string.
        tools: All tools the agent may use.
        system_prompt: Instructions for the agent.
        response_format: Optional schema (Pydantic/dataclass/TypedDict) or
            `ToolStrategy`/`ProviderStrategy`; the result lands in `structured_response`.
        max_model_calls: Per-run cap on model calls. `None` = unlimited.
        max_tool_calls: Per-run cap on tool calls (further calls are refused). `None` = unlimited.
        on_limit: "error" (raise) or "end" (stop with a final message) when
            `max_model_calls` is reached (see `agentpatterns.limits`).
        middleware: Additional middleware (HITL, summarization, retries, ...).
        checkpointer: Checkpoints the agent; only the outermost graph needs one.
        name: Graph name (used when embedded as a subgraph and in traces).
        **agent_kwargs: Passed through to `create_agent` (store, context_schema, ...).
    """
    guards = loop_limits(max_model_calls=max_model_calls, max_tool_calls=max_tool_calls, on_limit=on_limit)
    return create_agent(
        model,
        tools=list(tools),
        system_prompt=system_prompt,
        response_format=response_format,
        middleware=[*guards, *middleware],
        checkpointer=checkpointer,
        name=name,
        **agent_kwargs,
    )
