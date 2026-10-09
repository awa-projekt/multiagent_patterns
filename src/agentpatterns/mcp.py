"""MCP tools from a server the run names.

`McpTools` gives an agent the tools of an MCP server it only learns about when it runs:
the server is a function of the run's context (`runtime.context`), e.g. a tenant's
server or, in an eval, one server per case. The agent itself is built once.

    agent = create_agent(
        model,
        tools=[],
        middleware=[
            ToolRetryMiddleware(retry_on=unreachable, on_failure=lambda e: "The records server is unreachable."),
            McpTools(lambda ctx: ctx.mcp_url, tools=["search", "get_record"], server="records"),
        ],
        context_schema=Context,
    )
    await agent.ainvoke({"messages": [...]}, context=Context(mcp_url=...))

Built on `langchain.mcp` (beta, on FastMCP). The target can be anything `MCPAdapter`
accepts: an http(s) URL, or a `fastmcp.Client` for auth, timeouts or a tool-list cache
shared between runs.

When the agent starts, the middleware lists the server's tools once and keeps their
definitions in the agent's state; every model call is offered them, and every call of
one connects to the run's server. An error the server reports reaches the model as the
tool's answer. A server that asks for input during a call interrupts the run
(`langchain.mcp.elicitation`). A server that cannot be reached raises (see
`unreachable`): around a tool call, retry it with `ToolRetryMiddleware` placed before
`McpTools`; while listing, with the enclosing step's `retry_policy`.

Async only, like the MCP client. Needs the `mcp` extra: `multiagent-patterns[mcp]`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any, NotRequired

import anyio
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

try:
    import httpx2
    from langchain.mcp import MCPAdapter, as_langchain_tool
    from langchain.mcp.adapter import MCPAdapterTarget
    from mcp.types import Tool as McpTool
except ImportError as error:  # pragma: no cover - depends on the installed extras
    raise ImportError("agentpatterns.mcp needs the mcp extra: pip install 'multiagent-patterns[mcp]'") from error

__all__ = ["McpTools", "McpToolsState", "unreachable"]

TargetFor = Callable[[Any], MCPAdapterTarget]
"""The MCP server for a run, from the run's context (`None` without one)."""


def _by_server(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    return {**(left or {}), **(right or {})}


class McpToolsState(AgentState):
    mcp_tools: NotRequired[Annotated[dict[str, list[dict[str, Any]]], PrivateStateAttr, _by_server]]
    """Per server, the definitions of the tools listed when the agent started."""


_TRANSPORT_FAILURES = (httpx2.TransportError, anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream)
_UNAVAILABLE = frozenset({502, 503, 504})


def unreachable(error: BaseException) -> bool:
    """Whether an MCP call failed because the server could not be reached.

    FastMCP reports that as `RuntimeError("Client failed to connect: ...")` caused by
    the transport error, or as the transport error itself, and a gateway in front of a
    server that is down answers 502, 503 or 504. An error the server reports for a call
    (an unknown id, say) is no such failure: the agent gets it as the tool's answer.
    Use it as `retry_on=` of `ToolRetryMiddleware` or of a `RetryPolicy`.
    """
    if isinstance(error, BaseExceptionGroup):
        return any(unreachable(inner) for inner in error.exceptions)
    if isinstance(error, _TRANSPORT_FAILURES):
        return True
    if isinstance(error, httpx2.HTTPStatusError):
        return error.response.status_code in _UNAVAILABLE
    return error.__cause__ is not None and unreachable(error.__cause__)


class McpTools(AgentMiddleware):
    """Offers an agent the tools of the MCP server `target(runtime.context)` names.

    Args:
        target: The run's server, from the run's context: anything `MCPAdapter`
            accepts, e.g. `lambda ctx: ctx.mcp_url`, or `lambda ctx:
            fastmcp.Client(ctx.mcp_url, auth=ctx.token, cache=...)` for auth or a
            tool-list cache shared between runs. Called when the agent starts and for
            every model and tool call, so it should not do I/O.
        tools: The tools the agent gets, by name; all the server lists if `None`. A
            name the server does not list fails the agent's start.
        server: Names the server; an agent with several `McpTools` needs a different
            one for each.
    """

    state_schema = McpToolsState

    def __init__(self, target: TargetFor, *, tools: Sequence[str] | None = None, server: str = "mcp") -> None:
        super().__init__()
        self.target = target
        self.tool_names = tuple(tools) if tools is not None else None
        self.server = server

    @property
    def name(self) -> str:
        return f"McpTools_{self.server}"

    def _adapter(self, context: Any) -> MCPAdapter:
        return MCPAdapter(self.target(context))

    def _definitions(self, state: Any) -> list[dict[str, Any]]:
        return ((state or {}).get("mcp_tools") or {}).get(self.server, [])

    async def abefore_agent(self, state: McpToolsState, runtime: Runtime[Any]) -> dict[str, Any]:
        adapter = self._adapter(runtime.context)
        async with adapter:
            listed = await adapter.client.list_tools()
        by_name = {tool.name: tool for tool in listed}
        if self.tool_names is not None:
            if missing := [name for name in self.tool_names if name not in by_name]:
                raise ValueError(f"The MCP server {self.server!r} lists no tool {', '.join(missing)}.")
            listed = [by_name[name] for name in self.tool_names]
        definitions = [tool.model_dump(mode="json", by_alias=True, exclude_none=True) for tool in listed]
        return {"mcp_tools": {self.server: definitions}}

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        offered = {tool.name if isinstance(tool, BaseTool) else tool.get("name") for tool in request.tools}
        definitions = [d for d in self._definitions(request.state) if d["name"] not in offered]
        if not definitions:
            return await handler(request)
        client = self._adapter(request.runtime.context if request.runtime is not None else None).client
        mine = [await as_langchain_tool(McpTool.model_validate(d), client) for d in definitions]
        return await handler(request.override(tools=[*request.tools, *mine]))

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]
    ) -> ToolMessage | Command:
        name = request.tool_call["name"]
        definition = next((d for d in self._definitions(request.state) if d["name"] == name), None)
        if definition is None:
            return await handler(request)
        # Connected while the tool is built and called: the tool then knows the
        # server's protocol version, which decides whether it can interrupt for input.
        adapter = self._adapter(request.runtime.context if request.runtime is not None else None)
        async with adapter:
            if "outputSchema" in definition:
                # The client reads output schemas from a listing; without one it lists
                # past its cache. Listing here goes through the cache.
                await adapter.client.list_tools()
            tool = await as_langchain_tool(McpTool.model_validate(definition), adapter.client)
            return await handler(request.override(tool=tool))
