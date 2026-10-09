"""Test utilities: a scripted, tool-calling chat model and a usage tracker.

`ScriptedChatModel` is a drop-in replacement for a real chat model. It supports
everything the patterns rely on:

* `bind_tools()` -> the model sees the tool schemas of the current call
  (`create_agent`, `ToolStrategy` structured output and
  `with_structured_output()` all go through this),
* parallel tool calls (several tool calls in one `AIMessage`),
* structured output via tool calling (`ToolStrategy`, `with_structured_output()`)
  and via native structured output (`ProviderStrategy`,
  `with_structured_output(method="json_schema")`). Pass
  `profile={"structured_output": True}` to simulate a model with native support;
  `create_agent` and this library then pick the native path automatically, like
  they do for current Claude or OpenAI models. `turn.structured(...)` answers
  correctly on both paths.

Instead of replaying a fixed list of answers (like `GenericFakeChatModel`) it
calls a *policy* function for every model call. The policy receives a
`ModelTurn` (messages + bound tools) and returns what a real LLM would return.
Because the policy is a pure function of the conversation, parallel branches
and subagents stay deterministic regardless of execution order.

    def policy(turn: ModelTurn):
        if not turn.called("get_weather"):
            return turn.call("get_weather", city="Berlin")
        return turn.say(f"It is {turn.result('get_weather')}")

    model = ScriptedChatModel(policy=policy)

`serve_mcp(server)` runs an MCP server over HTTP for a test and yields its URL,
so agents with `McpTools` talk to a real MCP server (needs the `mcp` extra).
"""

from __future__ import annotations

import json
import threading
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_json_schema, convert_to_openai_tool
from pydantic import BaseModel

from agentpatterns.messaging import AGENT_MESSAGE_KEY, AgentMessage, is_agent_message

__all__ = ["ModelTurn", "ScriptedChatModel", "ScriptError", "UsageTracker", "serve_mcp"]


class ScriptError(RuntimeError):
    """Raised when a policy does something a real model could not do."""


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


def _tool_name(tool: dict[str, Any]) -> str:
    return tool.get("function", {}).get("name") or tool.get("name", "")


@dataclass
class ModelTurn:
    """Everything the model 'sees' in one call, plus helpers to build a reply."""

    messages: list[BaseMessage]
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: Any = None
    response_format: dict[str, Any] | None = None
    """Native structured output request (`{"type": "json_schema", "json_schema": {"name", "schema"}}`)."""

    # ------------------------------------------------------------------ reading
    @property
    def system(self) -> str:
        """All system prompt text of this call (joined)."""
        return "\n".join(m.text for m in self.messages if isinstance(m, SystemMessage))

    @property
    def human_messages(self) -> list[str]:
        return [m.text for m in self.messages if isinstance(m, HumanMessage)]

    @property
    def first_human(self) -> str:
        humans = self.human_messages
        return humans[0] if humans else ""

    @property
    def last_human(self) -> str:
        humans = self.human_messages
        return humans[-1] if humans else ""

    @property
    def agent_messages(self) -> list[AgentMessage]:
        """Messages from other agents delivered into this conversation (`MessagingMiddleware`)."""
        return [AgentMessage(**m.additional_kwargs[AGENT_MESSAGE_KEY]) for m in self.messages if is_agent_message(m)]

    @property
    def conversation_text(self) -> str:
        """All human + tool message text; handy for keyword heuristics."""
        return "\n".join(m.text for m in self.messages if isinstance(m, (HumanMessage, ToolMessage)))

    @property
    def tool_names(self) -> list[str]:
        return [_tool_name(t) for t in self.tools]

    def has_tool(self, name: str) -> bool:
        return name in self.tool_names

    def tools_with_prefix(self, prefix: str) -> list[str]:
        return [n for n in self.tool_names if n.startswith(prefix)]

    def tool_parameters(self, name: str) -> dict[str, Any]:
        """JSON schema of a bound tool's parameters."""
        for tool in self.tools:
            if _tool_name(tool) == name:
                return tool.get("function", {}).get("parameters", {})
        return {}

    def tool_calls_made(self, name: str | None = None) -> list[dict[str, Any]]:
        """Tool calls the model already issued in this conversation."""
        calls = [tc for m in self.messages if isinstance(m, AIMessage) for tc in m.tool_calls]
        return [c for c in calls if name is None or c["name"] == name]

    def called(self, name: str) -> bool:
        return bool(self.tool_calls_made(name)) or bool(self.tool_results(name))

    def tool_results(self, name: str | None = None) -> list[ToolMessage]:
        """Tool results visible in this conversation (optionally filtered by tool name)."""
        return [m for m in self.messages if isinstance(m, ToolMessage) and (name is None or m.name == name)]

    def result(self, name: str) -> str | None:
        """Latest raw result of a tool, or None."""
        results = self.tool_results(name)
        return results[-1].text if results else None

    def json_results(self, name: str) -> list[Any]:
        """All results of a tool, JSON-decoded where possible."""
        out = []
        for m in self.tool_results(name):
            try:
                out.append(json.loads(m.text))
            except (json.JSONDecodeError, TypeError):
                out.append(m.text)
        return out

    def json_result(self, name: str) -> Any:
        results = self.json_results(name)
        return results[-1] if results else None

    def call_results(self) -> list[dict[str, Any]]:
        """Completed tool calls as `{"tool", "args", "result"}` dicts (result JSON-decoded)."""
        by_id = {m.tool_call_id: m for m in self.messages if isinstance(m, ToolMessage)}
        pairs = []
        for call in self.tool_calls_made():
            message = by_id.get(call["id"])
            if message is None:
                continue
            try:
                result = json.loads(message.text)
            except (json.JSONDecodeError, TypeError):
                result = message.text
            pairs.append({"tool": call["name"], "args": call["args"], "result": result})
        return pairs

    # ----------------------------------------------------------------- replying
    def call(self, name: str, **args: Any) -> AIMessage:
        """Reply with a single tool call."""
        return self.call_many((name, args))

    def call_many(self, *calls: tuple[str, dict[str, Any]], content: str = "") -> AIMessage:
        """Reply with several (parallel) tool calls."""
        for name, _ in calls:
            if not self.has_tool(name):
                raise ScriptError(f"Policy tried to call tool {name!r}, but only {self.tool_names} are bound.")
        return AIMessage(
            content=content,
            tool_calls=[
                {"name": name, "args": args, "id": _new_call_id(), "type": "tool_call"} for name, args in calls
            ],
        )

    def say(self, text: str) -> AIMessage:
        """Reply with plain text (ends an agent loop)."""
        return AIMessage(content=text)

    def structured(self, data: BaseModel | dict[str, Any], tool_name: str | None = None) -> AIMessage:
        """Reply with structured output - as JSON text if native structured output
        was requested, else as a call of the structured output tool.

        The target tool is resolved from `tool_name`, the class name of `data`,
        or - if exactly one tool is bound (`with_structured_output`) - that tool.
        """
        args = data.model_dump(mode="json") if isinstance(data, BaseModel) else dict(data)
        name = tool_name or (type(data).__name__ if isinstance(data, BaseModel) else None)
        if name is not None and self.has_tool(name):
            return self.call(name, **args)
        if self.response_format is not None:
            return AIMessage(content=json.dumps(args))
        if len(self.tools) == 1:
            name = self.tool_names[0]
        else:
            name = self._tool_matching(args) or name
        if name is None:
            raise ScriptError("Cannot infer structured output tool; pass tool_name=.")
        return self.call(name, **args)

    def _tool_matching(self, args: dict[str, Any]) -> str | None:
        """The bound tool whose parameter schema fits `args` (structured output tools preferred)."""
        keys = set(args)
        fitting = [
            n
            for n in self.tool_names
            if set(self.tool_parameters(n).get("required", []))
            <= keys
            <= set(self.tool_parameters(n).get("properties", {}))
        ]
        fitting.sort(key=lambda n: not n[:1].isupper())  # schema classes are CamelCase
        return fitting[0] if fitting else None

    def structured_tool_schema(self) -> dict[str, Any] | None:
        """JSON schema of the requested structured output: the native response format,
        or the single bound tool (`with_structured_output` via tool calling)."""
        if self.response_format is not None:
            return self.response_format.get("json_schema", {}).get("schema")
        if len(self.tools) != 1:
            return None
        return self.tools[0].get("function", {}).get("parameters")


Policy = Callable[[ModelTurn], AIMessage | str]


class ScriptedChatModel(BaseChatModel):
    """Deterministic chat model driven by a policy function (see module docs)."""

    policy: Policy
    model_name: str = "scripted-mock"
    strict_tool_choice: bool = True
    """If True, raise when a tool call is forced (`tool_choice='any'`) but the policy returns text."""

    @property
    def _llm_type(self) -> str:
        return "scripted-chat-model"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | Any],
        *,
        tool_choice: str | dict | bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        formatted = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=formatted, tool_choice=tool_choice, **kwargs)

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        method: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Any]:
        """Tool calling (default, `method="function_calling"`) or native JSON output (`method="json_schema"`)."""
        if method in (None, "function_calling"):
            return super().with_structured_output(schema, include_raw=include_raw, **kwargs)
        if method != "json_schema":
            raise ValueError(f"Unsupported structured output method {method!r}.")
        if include_raw:
            raise NotImplementedError("include_raw is not supported with method='json_schema'.")
        json_schema = convert_to_json_schema(schema)
        llm = self.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {"name": json_schema.get("title", "response"), "schema": json_schema},
            }
        )
        is_model = isinstance(schema, type) and issubclass(schema, BaseModel)
        return llm | (PydanticOutputParser(pydantic_object=schema) if is_model else JsonOutputParser())

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        turn = ModelTurn(
            messages=list(messages),
            tools=list(kwargs.get("tools") or []),
            tool_choice=kwargs.get("tool_choice"),
            response_format=kwargs.get("response_format"),
        )
        output = self.policy(turn)
        message = AIMessage(content=output) if isinstance(output, str) else output
        forced = turn.tool_choice in ("any", "required", True) or isinstance(turn.tool_choice, dict)
        if self.strict_tool_choice and forced and turn.tools and not message.tool_calls:
            raise ScriptError(
                f"tool_choice forces a tool call, but the policy answered with text. Bound tools: {turn.tool_names}"
            )
        message.usage_metadata = _estimate_usage(turn, message)
        message.response_metadata = {"model_name": self.model_name}
        return ChatResult(generations=[ChatGeneration(message=message)])


def _estimate_usage(turn: ModelTurn, message: AIMessage) -> dict[str, int]:
    """Rough token estimate (~4 characters per token) so patterns can be compared."""
    prompt_chars = sum(len(m.text) for m in turn.messages) + len(json.dumps(turn.tools))
    if turn.response_format is not None:
        prompt_chars += len(json.dumps(turn.response_format))
    for m in turn.messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            prompt_chars += len(json.dumps([tc["args"] for tc in m.tool_calls]))
    output_chars = len(message.text) + len(json.dumps([tc["args"] for tc in message.tool_calls]))
    input_tokens, output_tokens = prompt_chars // 4 + 1, output_chars // 4 + 1
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


class UsageTracker(BaseCallbackHandler):
    """Callback handler counting model calls, tokens and tool calls of a run.

    Works with real models too (reads `usage_metadata`). Pass it via
    `config={"callbacks": [tracker]}`; callbacks propagate into subgraphs and
    subagents invoked from tools.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.model_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.tool_calls: Counter[str] = Counter()
        self.model_calls_by_agent: Counter[str] = Counter()

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[BaseMessage]], **kwargs: Any) -> None:
        metadata = kwargs.get("metadata") or {}
        agent = metadata.get("lc_agent_name") or metadata.get("langgraph_node") or "?"
        with self._lock:
            self.model_calls += 1
            self.model_calls_by_agent[str(agent)] += 1

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        with self._lock:
            for generations in response.generations:
                for gen in generations:
                    usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
                    self.input_tokens += usage.get("input_tokens", 0)
                    self.output_tokens += usage.get("output_tokens", 0)

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        with self._lock:
            self.tool_calls[serialized.get("name") or kwargs.get("name") or "?"] += 1

    def summary(self) -> dict[str, Any]:
        return {
            "model_calls": self.model_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tool_calls": sum(self.tool_calls.values()),
            "tools": dict(self.tool_calls),
            "model_calls_by_agent": dict(self.model_calls_by_agent),
        }


@contextmanager
def serve_mcp(server: Any) -> Iterator[str]:
    """Serve an MCP server over streamable HTTP on a free local port; yields its URL.

    `server` is a `fastmcp.FastMCP` or an `mcp.server.mcpserver.MCPServer`. It runs in
    a background thread for the duration of the `with` block.
    """
    import socket
    import time

    import uvicorn

    path = "/mcp"
    if hasattr(server, "http_app"):  # fastmcp
        app = server.http_app(path=path)
    else:  # the MCP SDK's MCPServer
        app = server.streamable_http_app(streamable_http_path=path)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    web = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=web.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not web.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("The MCP test server did not start.")
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}{path}"
    finally:
        web.should_exit = True
        thread.join(timeout=10)
