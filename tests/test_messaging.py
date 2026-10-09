"""Messaging between the agents of a run (`agentpatterns.messaging`)."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import RetryPolicy

import agentpatterns as ap
from agentpatterns.messaging import MessageRefusedError, is_agent_message
from tests.conftest import role_model


def wait_until(condition, timeout: float = 5.0) -> None:
    """Block a policy (it runs in a worker thread) until another agent did something."""
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for another agent")
        time.sleep(0.005)


def run(graph, inputs, mailbox: ap.Mailbox | None, mode: str = "sync", **config):
    if mailbox is not None:
        config.setdefault("configurable", {})["mailbox"] = mailbox
    if mode == "sync":
        return graph.invoke(inputs, config)
    return asyncio.run(graph.ainvoke(inputs, config))


def messaging_agent(model, name: str, **middleware_kwargs):
    return create_agent(
        model,
        [],
        system_prompt=f"role={name}",
        middleware=[ap.MessagingMiddleware(**middleware_kwargs)],
        name=name,
    )


def tool_messages(result, name: str) -> list[ToolMessage]:
    return [m for m in result["messages"] if isinstance(m, ToolMessage) and m.name == name]


# --------------------------------------------------------- agents in parallel
@pytest.mark.parametrize("mode", ["sync", "async"])
def test_parallel_agents_message_each_other_while_they_run(mode):
    mailbox = ap.Mailbox()

    def researcher(t):
        if not t.called("send_message"):
            wait_until(lambda: any(a.name == "analyst" for a in mailbox.agents()))  # the analyst has started
            return t.call("send_message", to="analyst", message="Revenue grew 12%.")
        return t.say(f"researched. {t.result('send_message')}")

    def analyst(t):
        assert "Your address is analyst." in t.system
        if not t.agent_messages:
            wait_until(lambda: mailbox.log)  # the researcher writes while the analyst thinks
            return t.say("analysis without data")  # would finish, but has unread mail
        message = t.agent_messages[-1]
        return t.say(f"analysis: {message.text} (from {message.sender})")

    model = role_model({"researcher": researcher, "analyst": analyst})
    team = ap.create_parallel(
        [
            ap.AgentSpec("researcher", "Finds facts.", messaging_agent(model, "researcher")),
            ap.AgentSpec("analyst", "Analyses.", messaging_agent(model, "analyst")),
        ]
    )
    out = run(team, {"messages": [("user", "Q3 report")]}, mailbox, mode)

    results = out["branch_results"]
    assert results["analyst"].text == "analysis: Revenue grew 12%. (from researcher)"
    assert results["researcher"].text == (
        "researched. Sent to analyst. It is running and gets the message before its next step."
    )
    [delivered] = [m for m in results["analyst"].messages if is_agent_message(m)]
    assert delivered.text == '<agent-message from="researcher" to="analyst">\nRevenue grew 12%.\n</agent-message>'
    assert [(m.sender, m.to, m.text) for m in mailbox.log] == [("researcher", "analyst", "Revenue grew 12%.")]
    assert mailbox.undelivered == []
    assert {(a.address, a.status) for a in mailbox.agents()} == {("researcher", "finished"), ("analyst", "finished")}


def test_copies_of_one_agent_get_their_own_addresses():
    both_running, both_listed = threading.Barrier(2, timeout=5), threading.Barrier(2, timeout=5)

    def mapper(t):
        if not t.called("list_agents"):
            both_running.wait()
            return t.call("list_agents")
        both_listed.wait()  # neither copy finishes before both have listed
        return t.say(t.result("list_agents"))

    agent = messaging_agent(role_model({"mapper": mapper}), "mapper", description="Maps one item.")
    graph = ap.create_map_reduce(
        agent,
        prepare=lambda item: {"messages": [("user", item)]},
        extract=lambda output: output["messages"][-1].text,
    )
    out = run(graph, {"items": ["a", "b"]}, ap.Mailbox())

    assert sorted(out["results"]) == [
        "Agents of this run:\n- mapper (running): Maps one item.\n- mapper#2 (running, you)",
        "Agents of this run:\n- mapper (running, you)\n- mapper#2 (running): Maps one item.",
    ]


def test_one_middleware_serves_every_agent_of_a_swarm_and_a_handoff_ends_a_run():
    def triage(t):
        if not t.called("send_message"):
            return t.call("send_message", to="billing", message="The customer is a VIP.")
        return t.call("transfer_to_billing", note="invoice question")

    def billing(t):
        assert "Your address is billing." in t.system
        return t.say(" | ".join(f"{m.sender}: {m.text}" for m in t.agent_messages if m.to == "billing"))

    swarm = ap.create_swarm(
        role_model({"triage": triage, "billing": billing}),
        [
            ap.SwarmAgent("triage", "Front desk.", "role=triage", handoffs=["billing"]),
            ap.SwarmAgent("billing", "Invoices.", "role=billing"),
        ],
        middleware=[ap.MessagingMiddleware()],  # each agent's address is its own name
    )
    mailbox = ap.Mailbox()
    out = run(swarm, {"messages": [("user", "invoice?")]}, mailbox)

    assert out["messages"][-1].text == "triage: The customer is a VIP."
    assert [(a.address, a.status) for a in mailbox.agents()] == [("triage", "finished"), ("billing", "finished")]


# ------------------------------------------------------- not running (yet)
def test_a_message_waits_for_an_agent_that_has_not_started():
    mailbox = ap.Mailbox()
    mailbox.send("writer", "Keep it under 50 words.", sender="editor")  # from code, before the run

    def drafter(t):
        if not t.called("send_message"):
            return t.call("send_message", to="writer", message="Use the Q3 numbers.")
        return t.say(t.result("send_message"))

    def writer(t):
        return t.say(" | ".join(f"{m.sender}: {m.text}" for m in t.agent_messages))

    model = role_model({"drafter": drafter, "writer": writer})
    pipeline = ap.create_pipeline(
        [
            ap.agent_step("drafter", messaging_agent(model, "drafter")),
            ap.agent_step("writer", messaging_agent(model, "writer")),
        ]
    )
    out = run(pipeline, {"messages": [("user", "report")]}, mailbox)

    assert out["outputs"]["drafter"] == (
        "No agent named writer has started in this run so far (agents so far: none). "
        "The message waits and is delivered when writer starts."
    )
    assert out["outputs"]["writer"] == "editor: Keep it under 50 words. | drafter: Use the Q3 numbers."
    assert mailbox.undelivered == []


def test_a_message_to_a_finished_agent_is_reported_as_undelivered():
    mailbox = ap.Mailbox()

    def early(t):
        return t.say("done early")

    def late(t):
        if not t.called("send_message"):
            wait_until(lambda: ("early", "finished") in {(a.name, a.status) for a in mailbox.agents()})
            return t.call("send_message", to="early", message="One more thing.")
        return t.say(t.result("send_message"))

    model = role_model({"early": early, "late": late})
    team = ap.create_parallel(
        [
            ap.AgentSpec("early", "Finishes first.", messaging_agent(model, "early")),
            ap.AgentSpec("late", "Writes later.", messaging_agent(model, "late")),
        ]
    )
    out = run(team, {"messages": [("user", "go")]}, mailbox)

    assert out["branch_results"]["late"].text == (
        "early has finished. The message waits and is delivered if an agent named early starts again in this run."
    )
    assert [m.text for m in mailbox.undelivered] == ["One more thing."]


def test_addresses_resolve_to_the_running_agent_with_that_name():
    mailbox = ap.Mailbox()
    first = mailbox._join(("run-1", "worker"), "")
    second = mailbox._join(("run-2", "worker"), "")
    assert (first, second) == ("worker", "worker#2")
    assert mailbox._join(("run-1", "worker"), "") == "worker"  # the same agent run (e.g. a retry) keeps its address

    assert mailbox.send("worker#1", "to the first", sender="ops").startswith("Sent to worker.")
    mailbox._finish(("run-1", "worker"))
    assert mailbox.send("worker", "redirected", sender="ops") == (
        "Sent to worker#2 (worker is not running). It is running and gets the message before its next step."
    )
    mailbox._join(("run-3", "worker"), "")
    with pytest.raises(MessageRefusedError, match="several agents named worker are: worker#2, worker#3"):
        mailbox.send("worker", "ambiguous", sender="ops")
    with pytest.raises(MessageRefusedError, match="no agent worker#9"):
        mailbox.send("worker#9", "unknown copy", sender="ops")
    with pytest.raises(MessageRefusedError, match="your own address"):
        mailbox._send_from(("run-2", "worker"), "worker#2", "me")
    assert [(m.to, m.text) for m in mailbox.log] == [("worker", "to the first"), ("worker#2", "redirected")]


# ---------------------------------------------------------------- guardrails
def test_can_message_and_the_message_limit_refuse_sends_and_the_agent_goes_on():
    script = [
        ("list_agents", {}),
        ("send_message", {"to": "boss", "message": "hi"}),
        ("send_message", {"to": "helper", "message": "note 0"}),
        ("send_message", {"to": "helper", "message": "note 1"}),
    ]

    def solo(t):
        step = len(t.tool_calls_made())
        return t.call(script[step][0], **script[step][1]) if step < len(script) else t.say("done")

    agent = messaging_agent(role_model({"solo": solo}), "solo", can_message=["helper"])
    mailbox = ap.Mailbox(max_messages_per_agent=1)
    out = run(agent, {"messages": [("user", "go")]}, mailbox)

    assert (
        tool_messages(out, "list_agents")[0].text
        == "Agents of this run:\n- solo (running, you)\n- helper (not started)"
    )
    refusals = [(m.status, m.text) for m in tool_messages(out, "send_message")]
    assert refusals[0] == ("error", "You may not message boss. You may message: helper.")
    assert refusals[1][0] == "success" and refusals[1][1].startswith("No agent named helper has started")
    assert refusals[2] == (
        "error",
        "Message limit reached: an agent may send at most 1 messages per run. Continue without messaging.",
    )
    assert out["messages"][-1].text == "done"
    assert [m.text for m in mailbox.log] == ["note 0"]


def test_without_a_mailbox_the_middleware_is_off():
    def solo(t):
        assert not t.has_tool("send_message") and not t.has_tool("list_agents")
        assert "agent-message" not in t.system
        return t.say("alone")

    agent = messaging_agent(role_model({"solo": solo}), "solo")
    assert run(agent, {"messages": [("user", "go")]}, None)["messages"][-1].text == "alone"
    with pytest.raises(TypeError, match="must be a Mailbox"):
        agent.invoke({"messages": [("user", "go")]}, {"configurable": {"mailbox": "nope"}})


def test_a_name_is_required():
    agent = create_agent(
        role_model({"x": lambda t: t.say("x")}), [], system_prompt="role=x", middleware=[ap.MessagingMiddleware()]
    )
    with pytest.raises(ValueError, match="needs a name"):
        run(agent, {"messages": [("user", "go")]}, ap.Mailbox())
    with pytest.raises(ValueError, match="alphanumeric"):
        ap.MessagingMiddleware("no spaces")


# ------------------------------------------------------- retries, threads, fork
def test_a_retried_agent_keeps_its_address_and_gets_its_messages_again():
    mailbox = ap.Mailbox()
    mailbox.send("analyst", "Use the EU numbers.", sender="lead")
    attempts = []

    def analyst(t):
        attempts.append([m.text for m in t.agent_messages])
        if len(attempts) == 1:
            raise RuntimeError("transient")
        return t.say(f"using: {t.agent_messages[0].text}")

    team = ap.create_parallel(
        [ap.AgentSpec("analyst", "Analyses.", messaging_agent(role_model({"analyst": analyst}), "analyst"))],
        retry_policy=RetryPolicy(max_attempts=2, initial_interval=0, retry_on=RuntimeError),
    )
    out = run(team, {"messages": [("user", "go")]}, mailbox)

    assert out["branch_results"]["analyst"].text == "using: Use the EU numbers."
    assert attempts == [["Use the EU numbers."], ["Use the EU numbers."]]
    assert [(a.address, a.status) for a in mailbox.agents()] == [("analyst", "finished")]


def test_each_run_reads_its_own_mailbox_on_a_checkpointed_thread():
    def solo(t):
        return t.say(" | ".join(m.text for m in t.agent_messages))

    agent = create_agent(
        role_model({"solo": solo}),
        [],
        system_prompt="role=solo",
        middleware=[ap.MessagingMiddleware("solo")],
        checkpointer=InMemorySaver(),
    )
    replies = []
    for text in ("first", "second"):
        mailbox = ap.Mailbox()
        mailbox.send("solo", text, sender="ops")
        out = run(agent, {"messages": [("user", "go")]}, mailbox, configurable={"thread_id": "t"})
        replies.append(out["messages"][-1].text)
    # Turn 2 sees both deliveries in its history, and its new mailbox's message is not skipped.
    assert replies == ["first", "first | second"]


def test_forked_subagents_do_not_see_the_supervisors_peer_messages():
    def lead(t):
        if not t.tool_results():
            return t.call("helper", task="help")
        return t.say(t.tool_results()[-1].text)

    def helper(t):
        return t.say(f"saw: {t.human_messages}")

    model = role_model({"lead": lead, "helper": helper})
    supervisor = ap.create_supervisor(
        model,
        [ap.AgentSpec("helper", "Helps.", create_agent(model, [], system_prompt="role=helper"))],
        system_prompt="role=lead",
        input_mode="fork",
        middleware=[ap.MessagingMiddleware("lead")],
    )
    mailbox = ap.Mailbox()
    mailbox.send("lead", "internal note", sender="ops")
    out = run(supervisor, {"messages": [("user", "question")]}, mailbox)

    assert out["messages"][-1].text == "saw: ['question', 'help']"
