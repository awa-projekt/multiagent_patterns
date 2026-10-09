"""`McpTools`: an agent built once that works against the MCP server each run names."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx2
import pytest
from fastmcp import Client, Context, FastMCP
from fastmcp.client.caching import KeyValueResponseCacheStore
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from mcp.client.caching import CacheConfig
from mcp.types import ElicitRequest, ElicitRequestFormParams, InputRequiredResult

import agentpatterns as ap
from agentpatterns.mcp import McpTools, unreachable
from agentpatterns.testing import serve_mcp
from tests.conftest import role_model


@dataclass
class Ctx:
    url: str


CONFIRM = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


def server(label: str, **settings) -> FastMCP:
    mcp = FastMCP(label, **settings)

    @mcp.tool()
    def where() -> str:
        """Which server this is."""
        return label

    @mcp.tool()
    def lookup(record_id: str) -> str:
        """A record by id."""
        if record_id == "missing":
            raise ToolError(f"no record {record_id}")
        return f"{label}:{record_id}"

    @mcp.tool()
    async def delete(record_id: str, ctx: Context) -> str:
        """Deletes a record once the user confirms."""
        if not ctx.input_responses:  # first round: ask, the client calls again with the answer
            confirm = ElicitRequestFormParams(message=f"Delete record {record_id}?", requested_schema=CONFIRM)
            return InputRequiredResult(input_requests={"confirm": ElicitRequest(params=confirm)})
        answer = ctx.input_responses["confirm"]
        return f"deleted {record_id}" if answer.action == "accept" and answer.content["ok"] else "kept"

    return mcp


@pytest.fixture(scope="module")
def servers():
    with serve_mcp(server("A")) as a, serve_mcp(server("B")) as b:
        yield {"A": a, "B": b}


def target(ctx: Ctx) -> str:
    return ctx.url


def where_agent(mcp_target=target, **mcp_kwargs):
    """Calls `where`, then answers with what it said."""

    def policy(turn):
        if not turn.called("where"):
            return turn.call("where")
        return turn.say(f"at {turn.result('where')}")

    return create_agent(
        role_model({"w": policy}),
        tools=[],
        system_prompt="role=w",
        middleware=[McpTools(mcp_target, **mcp_kwargs)],
        name="w",
    )


def answer(result: dict) -> str:
    return result["messages"][-1].text


def test_each_run_uses_the_server_its_context_names(servers):
    async def run():
        agent = where_agent()
        a, b = await asyncio.gather(
            agent.ainvoke({"messages": [("user", "go")]}, context=Ctx(servers["A"])),
            agent.ainvoke({"messages": [("user", "go")]}, context=Ctx(servers["B"])),
        )
        assert (answer(a), answer(b)) == ("at A", "at B")
        assert "mcp_tools" not in a  # the tool definitions stay private to the agent

    asyncio.run(run())


def test_only_the_named_tools_are_offered(servers):
    async def run():
        seen = []

        def policy(turn):
            seen.append(sorted(turn.tool_names))
            return turn.say("done")

        agent = create_agent(
            role_model({"w": policy}),
            tools=[],
            system_prompt="role=w",
            middleware=[McpTools(target, tools=["where", "lookup"])],
        )
        await agent.ainvoke({"messages": [("user", "go")]}, context=Ctx(servers["A"]))
        assert seen == [["lookup", "where"]]

    asyncio.run(run())


def test_a_tool_the_server_does_not_list_fails_the_start(servers):
    async def run():
        agent = where_agent(tools=["where", "nonexistent"])
        with pytest.raises(ValueError, match="nonexistent"):
            await agent.ainvoke({"messages": [("user", "go")]}, context=Ctx(servers["A"]))

    asyncio.run(run())


def test_an_error_the_server_reports_reaches_the_model(servers):
    async def run():
        def policy(turn):
            if not turn.called("lookup"):
                return turn.call("lookup", record_id="missing")
            return turn.say(f"got: {turn.result('lookup')}")

        agent = create_agent(role_model({"w": policy}), tools=[], system_prompt="role=w", middleware=[McpTools(target)])
        result = await agent.ainvoke({"messages": [("user", "go")]}, context=Ctx(servers["B"]))
        assert "no record missing" in answer(result)

    asyncio.run(run())


def test_a_server_down_at_the_start_is_unreachable():
    async def run():
        with serve_mcp(server("gone")) as url:
            pass  # stopped again: nothing listens there any more
        with pytest.raises(BaseException) as raised:
            await where_agent().ainvoke({"messages": [("user", "go")]}, context=Ctx(url))
        assert unreachable(raised.value)

    asyncio.run(run())


def test_a_server_lost_mid_run_is_retried_then_answered(servers):
    async def run():
        with serve_mcp(server("short-lived")) as url:
            ctx = Ctx(url)

        def policy(turn):
            if not turn.called("where"):
                ctx.url = url  # the server is gone by the time the tool is called
                return turn.call("where")
            return turn.say(f"got: {turn.result('where')}")

        ctx.url = servers["A"]
        agent = create_agent(
            role_model({"w": policy}),
            tools=[],
            system_prompt="role=w",
            middleware=[
                ToolRetryMiddleware(
                    max_retries=1, initial_delay=0, retry_on=unreachable, on_failure=lambda e: "server down"
                ),
                McpTools(target),
            ],
        )
        result = await agent.ainvoke({"messages": [("user", "go")]}, context=ctx)
        assert answer(result) == "got: server down"

    asyncio.run(run())


def test_a_subagent_reaches_the_server_of_the_run(servers):
    """Patterns pass the run's context through to the agents inside them."""

    async def run():

        def lead(turn):
            if not turn.tool_results():
                return turn.call("w", task="where are you?")
            return turn.say(turn.tool_results()[-1].text)

        supervisor = ap.create_supervisor(
            role_model({"lead": lead}),
            [ap.AgentSpec("w", "Finds out where it is.", where_agent())],
            system_prompt="role=lead",
        )
        result = await supervisor.ainvoke({"messages": [("user", "go")]}, context=Ctx(servers["B"]))
        assert answer(result) == "at B"

    asyncio.run(run())


def test_two_servers_on_one_agent(servers):
    async def run():
        @dataclass
        class Both:
            a: str
            b: str

        def policy(turn):
            if not turn.called("where"):
                return turn.call_many(("where", {}), ("lookup", {"record_id": "7"}))
            return turn.say(f"{turn.result('where')} + {turn.result('lookup')}")

        agent = create_agent(
            role_model({"w": policy}),
            tools=[],
            system_prompt="role=w",
            middleware=[
                McpTools(lambda ctx: ctx.a, tools=["where"], server="a"),
                McpTools(lambda ctx: ctx.b, tools=["lookup"], server="b"),
            ],
        )
        result = await agent.ainvoke({"messages": [("user", "go")]}, context=Both(servers["A"], servers["B"]))
        assert answer(result) == "A + B:7"

    asyncio.run(run())


def test_a_server_that_asks_for_input_interrupts_the_run(servers):
    """The server's question becomes a LangGraph interrupt; the answer resumes the call."""

    async def run():
        def policy(turn):
            if not turn.called("delete"):
                return turn.call("delete", record_id="7")
            return turn.say(turn.result("delete"))

        agent = create_agent(
            role_model({"w": policy}),
            tools=[],
            system_prompt="role=w",
            middleware=[McpTools(target)],
            checkpointer=InMemorySaver(),
        )
        config = {"configurable": {"thread_id": "confirm"}}
        paused = await agent.ainvoke({"messages": [("user", "delete 7")]}, config, context=Ctx(servers["A"]))
        [question] = paused["__interrupt__"]
        assert question.value["type"] == "mcp_elicitation"
        [request] = question.value["requests"]
        assert request["message"] == "Delete record 7?"
        accept = {"responses": {request["key"]: {"action": "accept", "content": {"ok": True}}}}
        result = await agent.ainvoke(Command(resume=accept), config, context=Ctx(servers["A"]))
        assert answer(result) == "deleted 7"

    asyncio.run(run())


def test_runs_share_the_tool_list_per_tenant_while_the_server_allows():
    """A `fastmcp.Client` target with a shared cache store: one tools/list per tenant and TTL."""
    listings = []

    class CountListings(Middleware):
        async def on_list_tools(self, context, call_next):
            listings.append(1)
            return await call_next(context)

    @dataclass
    class TenantCtx:
        url: str
        tenant: str

    store = KeyValueResponseCacheStore()

    def tenant_client(ctx: TenantCtx) -> Client:
        return Client(ctx.url, cache=CacheConfig(store=store, partition=ctx.tenant, target_id="records"))

    async def run(url: str):
        agent = where_agent(tenant_client)
        for tenant in ["a", "a", "b"]:
            result = await agent.ainvoke({"messages": [("user", "go")]}, context=TenantCtx(url, tenant))
            assert answer(result) == "at cached"

    cached = server("cached", cache_ttl=60)  # the server allows caching its lists for 60 s
    cached.add_middleware(CountListings())
    with serve_mcp(cached) as url:
        asyncio.run(run(url))
    assert len(listings) == 2  # tenant "a" once, tenant "b" once


def status_error(code: int) -> httpx2.HTTPStatusError:
    request = httpx2.Request("POST", "http://mcp.test/mcp")
    return httpx2.HTTPStatusError("status", request=request, response=httpx2.Response(code, request=request))


def test_unreachable_tells_transport_failures_from_errors_the_server_reports():
    request = httpx2.Request("POST", "http://mcp.test/mcp")
    connect = httpx2.ConnectError("refused", request=request)
    wrapped = RuntimeError("Client failed to connect: refused")
    wrapped.__cause__ = connect

    assert unreachable(connect)
    assert unreachable(wrapped)
    assert unreachable(ExceptionGroup("task group", [wrapped]))
    assert unreachable(status_error(503))  # a gateway in front of a server that is down
    assert not unreachable(status_error(500))
    assert not unreachable(RuntimeError("Client failed to connect"))  # no transport cause
    assert not unreachable(ToolError("no record 7"))
