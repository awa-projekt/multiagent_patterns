"""Unit tests of the agentpatterns factories, independent of the e-mail use case."""

from __future__ import annotations

import asyncio
import itertools
import json
import re
from dataclasses import dataclass
from typing import NotRequired

import pytest
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from pydantic import BaseModel

import agentpatterns as ap
from tests.conftest import role_model


class Answer(BaseModel):
    """Final answer."""

    text: str


def echo_agent(prefix: str, name: str | None = None):
    """Agent without tools that answers `<prefix>: <last human message>`."""
    model = role_model({"echo": lambda t: t.say(f"{prefix}: {t.last_human}")})
    return create_agent(model, tools=[], system_prompt="role=echo", name=name or prefix)


def spec(name: str, prefix: str | None = None) -> ap.AgentSpec:
    return ap.AgentSpec(name, f"The {name} agent.", echo_agent(prefix or name, name))


# ------------------------------------------------------------ single agent
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def test_single_agent_tool_loop_and_structured_output():
    def calc(t):
        if not t.called("add"):
            return t.call("add", a=2, b=3)
        return t.structured(Answer(text=t.result("add")))

    agent = ap.create_single_agent(role_model({"calc": calc}), [add], system_prompt="role=calc", response_format=Answer)
    assert agent.invoke({"messages": [("user", "2+3?")]})["structured_response"] == Answer(text="5")


def test_single_agent_model_call_limit_stops_runaway_loops():
    looping = role_model({"loop": lambda t: t.call("add", a=1, b=1)})
    agent = ap.create_single_agent(looping, [add], system_prompt="role=loop", max_model_calls=3)
    with pytest.raises(Exception, match="(?i)limit"):
        agent.invoke({"messages": [("user", "go")]})


# ---------------------------------------------------------------- pipeline
class Label(BaseModel):
    """Label."""

    label: str


def test_pipeline_runs_steps_and_passes_outputs():
    model = role_model(
        {
            "label": lambda t: t.structured({"label": "question"}),
            "answer": lambda t: t.say("answer using " + ("question" if "question" in t.last_human else "?")),
        }
    )
    pipeline = ap.create_pipeline(
        [
            ap.llm_step("label", model, system_prompt="role=label", output_schema=Label),
            ap.function_step("length", lambda s: len(s["context"]["text"])),
            ap.llm_step("answer", model, system_prompt="role=answer"),
        ]
    )
    out = pipeline.invoke({"messages": [("user", "Why?")], "context": {"text": "Why?"}})
    assert out["outputs"]["label"] == Label(label="question")
    assert out["outputs"]["length"] == 4
    assert out["structured_response"] == "answer using question"
    assert out["messages"][-1].text == "answer using question"


def test_pipeline_gate_short_circuits_with_fallback():
    calls = []
    model = role_model(
        {"label": lambda t: t.structured({"label": "spam"}), "never": lambda t: calls.append(1) or t.say("x")}
    )
    pipeline = ap.create_pipeline(
        [
            ap.llm_step(
                "label",
                model,
                system_prompt="role=label",
                output_schema=Label,
                gate=lambda s: s["outputs"]["label"].label != "spam",
                on_gate_fail=lambda s: {"ignored": True},
            ),
            ap.llm_step("never", model, system_prompt="role=never"),
        ]
    )
    out = pipeline.invoke({"messages": [("user", "win money")]})
    assert out["structured_response"] == {"ignored": True}
    assert out["stopped_at"] == "label"
    assert calls == []


def test_pipeline_async():
    pipeline = ap.create_pipeline([ap.agent_step("echo", echo_agent("E"))])
    out = asyncio.run(pipeline.ainvoke({"messages": [("user", "hi")]}))
    assert out["structured_response"] == "E: hi"


# ------------------------------------------------------------------ router
def router_policy(targets):
    def route(t):
        return t.structured({"routes": [{"agent": a, "task": f"task for {a}"} for a in targets]})

    return route


def test_router_fans_out_in_parallel_and_synthesizes():
    model = role_model(
        {
            "router": router_policy(["alpha", "beta"]),
            "synth": lambda t: t.structured(
                {
                    "text": " + ".join(
                        sorted(line for line in t.last_human.splitlines() if ":" in line and "task" in line)
                    )
                }
            ),
        }
    )
    router = ap.create_router(
        model,
        [spec("alpha"), spec("beta"), spec("gamma")],
        system_prompt="role=router",
        synthesizer_prompt="role=synth",
        response_format=Answer,
    )
    out = router.invoke({"messages": [("user", "do both")]})
    assert [r["agent"] for r in out["routes"]] == ["alpha", "beta"]
    assert out["structured_response"].text == "alpha: task for alpha + beta: task for beta"


def test_router_single_route_passes_through_without_synthesis():
    model = role_model({"router": router_policy(["alpha"])})
    router = ap.create_router(model, [spec("alpha"), spec("beta")], system_prompt="role=router")
    out = router.invoke({"messages": [("user", "x")]})
    assert out["messages"][-1].text == "alpha: task for alpha"
    assert "structured_response" not in out


def test_router_with_deterministic_route_fn():
    model = role_model({"synth": lambda t: t.say("merged")})
    router = ap.create_router(
        model,
        [spec("alpha"), spec("beta")],
        synthesizer_prompt="role=synth",
        route_fn=lambda s: [{"agent": "beta", "task": "rule-based"}],
        synthesize=True,
    )
    assert router.invoke({"messages": [("user", "x")]})["messages"][-1].text == "merged"


def test_router_routing_schema_constrains_agent_names():
    seen = {}

    def route(t):
        seen["schema"] = t.structured_tool_schema()
        return t.structured({"routes": []})

    model = role_model({"router": route, "synth": lambda t: t.say("nothing to do")})
    router = ap.create_router(
        model, [spec("alpha"), spec("beta")], system_prompt="role=router", synthesizer_prompt="role=synth"
    )
    assert router.invoke({"messages": [("user", "x")]})["messages"][-1].text == "nothing to do"
    assert '"enum": ["alpha", "beta"]' in json.dumps(seen["schema"])


# ---------------------------------------------------------------- parallel
def test_parallel_with_code_aggregator():
    graph = ap.create_parallel(
        [spec("a"), spec("b")], aggregator=lambda results, state: sorted(r.text for r in results.values())
    )
    out = graph.invoke({"messages": [("user", "q")]})
    assert out["structured_response"] == ["a: q", "b: q"]
    assert set(out["branch_results"]) == {"a", "b"}


def test_parallel_with_llm_synthesis():
    model = role_model({"synth": lambda t: t.structured({"text": "both"})})
    graph = ap.create_parallel(
        [spec("a"), spec("b")], model=model, synthesizer_prompt="role=synth", response_format=Answer
    )
    assert graph.invoke({"messages": [("user", "q")]})["structured_response"] == Answer(text="both")


def test_voting_returns_majority():
    answers = itertools.cycle(["yes", "no", "yes"])
    voter = create_agent(role_model({"vote": lambda t: t.say(next(answers))}), tools=[], system_prompt="role=vote")
    graph = ap.create_voting(ap.AgentSpec("voter", "votes", voter), n=3)
    out = graph.invoke({"messages": [("user", "?")]})
    assert out["messages"][-1].text == "yes"
    assert out["votes"] == {"yes": 2, "no": 1}


def test_map_reduce_keeps_order_and_handles_empty_input():
    from langchain_core.runnables import RunnableLambda

    graph = ap.create_map_reduce(RunnableLambda(lambda x: x * 10), reduce=sum)
    out = graph.invoke({"items": [3, 1, 2]})
    assert out["results"] == [30, 10, 20]
    assert out["output"] == 60
    assert graph.invoke({"items": []})["output"] == 0


# ------------------------------------------------------------ orchestrator
def test_orchestrator_replans_until_done():
    def plan(t):
        done = t.last_human.count("[")  # each worker result is rendered as "[worker]"
        if "Results so far" not in t.last_human:
            return t.structured({"tasks": [{"worker": "researcher", "instruction": "find"}]})
        if done == 1:
            return t.structured(
                {"tasks": [{"worker": "writer", "instruction": "write"}, {"worker": "writer", "instruction": "polish"}]}
            )
        return t.structured({"tasks": []})

    model = role_model({"plan": plan, "synth": lambda t: t.structured({"text": t.last_human.count("[") * "x"})})
    graph = ap.create_orchestrator(
        model,
        [spec("researcher"), spec("writer")],
        planner_prompt="role=plan",
        synthesizer_prompt="role=synth",
        response_format=Answer,
        max_rounds=3,
    )
    out = graph.invoke({"messages": [("user", "report")]})
    assert [(r["round"], r["worker"], r["task"]) for r in out["results"]] == [
        (1, "researcher", "find"),
        (2, "writer", "write"),
        (2, "writer", "polish"),
    ]
    assert out["structured_response"] == Answer(text="xxx")


def test_orchestrator_single_round_skips_replanning():
    planner_calls = []

    def plan(t):
        planner_calls.append(1)
        return t.structured({"tasks": [{"worker": "w", "instruction": "go"}]})

    model = role_model({"plan": plan, "synth": lambda t: t.say("done")})
    graph = ap.create_orchestrator(model, [spec("w")], planner_prompt="role=plan", synthesizer_prompt="role=synth")
    assert graph.invoke({"messages": [("user", "x")]})["messages"][-1].text == "done"
    assert len(planner_calls) == 1


# -------------------------------------------------------------- supervisor
def supervisor_policy(tool_args):
    def supervise(t):
        if not t.tool_results():
            return t.call_many(*tool_args)
        return t.structured({"text": " | ".join(sorted(m.text for m in t.tool_results()))})

    return supervise


def test_supervisor_calls_subagents_as_parallel_tools():
    model = role_model({"sup": supervisor_policy([("alpha", {"task": "one"}), ("beta", {"task": "two"})])})
    sup = ap.create_supervisor(model, [spec("alpha"), spec("beta")], system_prompt="role=sup", response_format=Answer)
    assert sup.invoke({"messages": [("user", "x")]})["structured_response"].text == "alpha: one | beta: two"


def test_supervisor_single_task_tool():
    model = role_model({"sup": supervisor_policy([("task", {"agent_name": "beta", "description": "two"})])})
    sup = ap.create_supervisor(
        model, [spec("alpha"), spec("beta")], system_prompt="role=sup", response_format=Answer, delegation="task_tool"
    )
    assert sup.invoke({"messages": [("user", "x")]})["structured_response"].text == "beta: two"


def test_supervisor_fork_mode_forwards_conversation():
    seen = {}
    sub_model = role_model({"sub": lambda t: seen.setdefault("humans", t.human_messages) and t.say("ok")})
    sub = ap.AgentSpec("sub", "sub", create_agent(sub_model, tools=[], system_prompt="role=sub"))
    model = role_model({"sup": supervisor_policy([("sub", {"task": "continue"})])})
    sup = ap.create_supervisor(model, [sub], system_prompt="role=sup", response_format=Answer, input_mode="fork")
    sup.invoke({"messages": [("user", "original request")]})
    assert seen["humans"] == ["original request", "continue"]


def test_hierarchy_nests_supervisors():
    lead = role_model({"lead": supervisor_policy([("alpha", {"task": "t"})])})
    top = role_model({"top": supervisor_policy([("team", {"task": "delegate"})])})
    team = ap.Team(
        "team", "A team.", system_prompt="role=lead", members=[spec("alpha")], response_format=Answer, model=lead
    )
    graph = ap.create_hierarchy(top, [team], system_prompt="role=top", response_format=Answer)
    out = graph.invoke({"messages": [("user", "x")]})
    assert out["structured_response"].text == '{"text":"alpha: t"}'


# ------------------------------------------------------------------- swarm
def swarm_model():
    def alice(t):
        if "billing" in t.last_human and not t.tool_results():
            return t.call("transfer_to_bob", note="billing question")
        return t.say("alice answers")

    def bob(t):
        return t.say("bob answers")

    return role_model({"alice": alice, "bob": bob})


def test_swarm_handoff_and_active_agent_memory():
    swarm = ap.create_swarm(
        swarm_model(),
        [ap.SwarmAgent("alice", "Front desk", "role=alice"), ap.SwarmAgent("bob", "Billing", "role=bob")],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "1"}}
    first = swarm.invoke({"messages": [("user", "billing issue")]}, config)
    assert first["messages"][-1].text == "bob answers"
    assert first["active_agent"] == "bob"
    second = swarm.invoke({"messages": [("user", "thanks")]}, config)  # resumes with bob
    assert second["messages"][-1].text == "bob answers"


def test_swarm_handoff_only_history_forwards_just_the_pair():
    swarm = ap.create_swarm(
        swarm_model(),
        [ap.SwarmAgent("alice", "Front desk", "role=alice"), ap.SwarmAgent("bob", "Billing", "role=bob")],
        history="handoff_only",
    )
    out = swarm.invoke({"messages": [("user", "billing issue")]})
    kinds = [type(m).__name__ for m in out["messages"]]
    assert kinds == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]


def test_swarm_max_handoffs_prevents_ping_pong():
    def ping(t):
        target = "transfer_to_bob" if t.has_tool("transfer_to_bob") else "transfer_to_alice"
        last = t.tool_results()[-1].text if t.tool_results() else ""
        if "limit" in last:
            return t.say("finishing myself")
        return t.call(target, note="yours")

    swarm = ap.create_swarm(
        role_model({"alice": ping, "bob": ping}),
        [ap.SwarmAgent("alice", "A", "role=alice"), ap.SwarmAgent("bob", "B", "role=bob")],
        max_handoffs=3,
    )
    out = swarm.invoke({"messages": [("user", "go")]})
    assert out["messages"][-1].text == "finishing myself"
    assert out["handoff_count"] == 3


# ----------------------------------------------------------- state machine
def test_state_machine_moves_through_steps_with_step_specific_tools():
    seen_tools = []

    def agent(t):
        seen_tools.append(sorted(t.tool_names))
        if "step=collect" in t.system:
            return t.call("go_to_solve", reason="collected")
        if "step=solve" in t.system:
            if not t.called("add"):
                return t.call("add", a=1, b=2)
            return t.call("go_to_answer", reason="solved")
        return t.structured(Answer(text=f"sum={t.result('add')}"))

    graph = ap.create_state_machine_agent(
        role_model({"sm": agent}),
        [
            ap.Step("collect", "role=sm step=collect", transitions=["solve"]),
            ap.Step("solve", "role=sm step=solve", tools=[add], transitions=["answer"]),
            ap.Step("answer", "role=sm step=answer", final=True),
        ],
        response_format=Answer,
    )
    out = graph.invoke({"messages": [("user", "1+2")]})
    assert out["structured_response"] == Answer(text="sum=3")
    assert out["current_step"] == "answer"
    assert seen_tools == [["go_to_solve"], ["add", "go_to_answer"], ["add", "go_to_answer"], ["Answer"]]


def test_state_machine_custom_transition_tool_and_prompt_template():
    class State(ap.StateMachineState):
        topic: NotRequired[str]

    @tool
    def set_topic(topic: str, runtime: ToolRuntime):
        """Remember the topic and move on."""
        return ap.transition("done", runtime.tool_call_id, topic=topic)

    def agent(t):
        if "step=start" in t.system:
            return t.call("set_topic", topic="billing")
        return t.say(t.system)

    graph = ap.create_state_machine_agent(
        role_model({"sm": agent}),
        [ap.Step("start", "role=sm step=start", tools=[set_topic]), ap.Step("done", "role=sm topic={topic}")],
        state_schema=State,
    )
    assert graph.invoke({"messages": [("user", "x")]})["messages"][-1].text == "role=sm topic=billing"


# ------------------------------------------------------------------ skills
@tool
def query_db(sql: str) -> str:
    """Run SQL."""
    return "42 rows"


def test_skills_unlock_tools_progressively():
    seen = []

    def agent(t):
        seen.append((sorted(t.tool_names), "sql: Write SQL" in t.system))
        if not t.called("load_skill"):
            return t.call_many(("load_skill", {"skill_name": "sql"}), ("load_skill", {"skill_name": "legal"}))
        if not t.called("query_db"):
            return t.call("query_db", sql="select 1")
        return t.say(t.result("query_db"))

    graph = ap.create_skills_agent(
        role_model({"skills": agent}),
        [
            ap.Skill("sql", "Write SQL", "Use ANSI SQL.", tools=[query_db]),
            ap.Skill("legal", "Contracts", "Be careful."),
        ],
        system_prompt="role=skills",
    )
    out = graph.invoke({"messages": [("user", "count rows")]})
    assert out["messages"][-1].text == "42 rows"
    assert out["loaded_skills"] == ["sql", "legal"]  # parallel loads merged by the reducer
    assert seen[0] == (["load_skill"], True)  # query_db hidden before loading, catalog in prompt
    assert "query_db" in seen[1][0]


# ------------------------------------------------------ evaluator-optimizer
def reflective_generator():
    def write(t):
        feedback = [h for h in t.human_messages if "feedback" in h.lower()]
        return t.structured(Answer(text=f"draft v{len(feedback) + 1}"))

    return create_agent(role_model({"gen": write}), tools=[], system_prompt="role=gen", response_format=Answer)


def test_evaluator_optimizer_iterates_until_passed():
    def judge(t):
        ok = "v2" in t.last_human
        return t.structured(
            {
                "passed": ok,
                "score": 1.0 if ok else 0.3,
                "issues": [] if ok else ["too short"],
                "feedback": "fine" if ok else "make it longer",
            }
        )

    loop = ap.create_evaluator_optimizer(
        reflective_generator(), role_model({"judge": judge}), evaluator_prompt="role=judge"
    )
    out = loop.invoke({"messages": [("user", "write something")]})
    assert out["structured_response"] == Answer(text="draft v2")
    assert out["iterations"] == 2
    assert out["evaluation"].passed


def test_evaluator_optimizer_applies_fallback_at_max_iterations():
    judge = role_model(
        {"judge": lambda t: t.structured({"passed": False, "score": 0, "issues": ["bad"], "feedback": "no"})}
    )
    loop = ap.create_evaluator_optimizer(
        reflective_generator(),
        judge,
        evaluator_prompt="role=judge",
        max_iterations=2,
        on_max_iterations=lambda draft, ev: Answer(text=f"needs human ({draft.text})"),
    )
    out = loop.invoke({"messages": [("user", "x")]})
    assert out["structured_response"] == Answer(text="needs human (draft v2)")
    assert out["iterations"] == 2


# ------------------------------------------------- research -> review recipe
class Report(BaseModel):
    """Research report."""

    summary: str
    sources: list[str]


def research_review(searches: list[str], reviews: list[str]):
    """Orchestrator with a researcher subagent, wrapped in a review loop with an agent reviewer."""

    @tool
    def search(query: str) -> str:
        """Search the web."""
        searches.append(query)
        return f"source://{query.replace(' ', '-')} says: facts about {query}"

    def research(t):
        return t.call("search", query=t.first_human) if not t.called("search") else t.say(t.result("search"))

    def plan(t):
        if "Results so far" not in t.last_human:  # fresh research
            return t.structured({"tasks": [{"worker": "researcher", "instruction": q} for q in ("market", "rivals")]})
        wants_pricing = any("pricing" in h for h in t.human_messages[:-1])  # reviewer feedback / follow-up
        if wants_pricing and "(task: pricing)" not in t.last_human:
            return t.structured({"tasks": [{"worker": "researcher", "instruction": "pricing"}]})
        return t.structured({"tasks": []})  # nothing to research, just rewrite

    def synth(t):
        return t.structured(Report(summary="report", sources=sorted(set(re.findall(r"source://\S+", t.last_human)))))

    def review(t):
        reviews.append(t.last_human)
        ok = "source://pricing" in t.last_human.split("Candidate to evaluate")[1]
        return t.structured(
            ap.Evaluation(
                passed=ok,
                score=1.0 if ok else 0.4,
                issues=[] if ok else ["no pricing data"],
                feedback="fine" if ok else "Add pricing research.",
            )
        )

    model = role_model({"research": research, "plan": plan, "synth": synth, "review": review})
    researcher = create_agent(model, tools=[search], system_prompt="role=research", name="researcher")
    reviewer = create_agent(
        model, tools=[], system_prompt="role=review", response_format=ap.Evaluation, name="reviewer"
    )
    orchestrator = ap.create_orchestrator(
        model,
        [ap.AgentSpec("researcher", "Researches one question", researcher)],
        planner_prompt="role=plan",
        synthesizer_prompt="role=synth",
        response_format=Report,
    )
    return ap.create_evaluator_optimizer(
        orchestrator,
        reviewer,
        carry_over=["results"],
        evaluator_context=lambda s: ap.results_block(s.get("results", []), "Research results"),
        max_iterations=3,
    )


def test_review_loop_revises_research_without_redoing_it():
    searches, reviews = [], []
    out = research_review(searches, reviews).invoke({"messages": [("user", "Research the EV market")]})

    assert searches == ["market", "rivals", "pricing"]  # the revision only researched the gap
    assert [(r["round"], r["task"]) for r in out["results"]] == [(1, "market"), (1, "rivals"), (2, "pricing")]
    assert out["iterations"] == 2 and out["evaluation"].passed
    assert out["structured_response"].sources == ["source://market", "source://pricing", "source://rivals"]
    assert all("Research results" in r and "source://market" in r for r in reviews)  # reviewer saw the evidence


def test_review_loop_as_subgraph_builds_on_earlier_research():
    class Workflow(MessagesState):
        results: list  # no reducer: the loop returns the complete list
        structured_response: NotRequired[Report]
        evaluation: NotRequired[ap.Evaluation]

    searches, reviews = [], []
    graph = (
        StateGraph(Workflow)
        .add_node("research", research_review(searches, reviews))
        .add_edge(START, "research")
        .compile(checkpointer=InMemorySaver(serde=ap.make_serializer(Report, ap.Evaluation)))
    )
    config = {"configurable": {"thread_id": "t1"}}
    graph.invoke({"messages": [("user", "Research the EV market")]}, config)
    assert searches == ["market", "rivals", "pricing"]

    out = graph.invoke({"messages": [("user", "Anything new on pricing?")]}, config)
    assert searches == ["market", "rivals", "pricing"]  # follow-up answered from the stored research
    assert len(out["results"]) == 3 and out["evaluation"].passed


def test_orchestrator_continues_from_earlier_results():
    planner_inputs = []

    def plan(t):
        planner_inputs.append(t.last_human)
        return t.structured({"tasks": [{"worker": "w", "instruction": "more"}]})

    model = role_model({"plan": plan, "synth": lambda t: t.say("done")})
    graph = ap.create_orchestrator(
        model, [spec("w")], planner_prompt="role=plan", synthesizer_prompt="role=synth", max_rounds=2
    )
    earlier = [{"round": 1, "worker": "w", "task": "first", "output": "w: first"}]
    out = graph.invoke({"messages": [("user", "x")], "results": earlier})
    assert [r["round"] for r in out["results"]] == [1, 2, 3]  # numbering continues, two new rounds
    assert "(task: first)" in planner_inputs[0]


def test_evaluator_optimizer_input_validation():
    with pytest.raises(ValueError, match="used by the loop"):
        ap.create_evaluator_optimizer(spec("g").agent, echo_agent("r"), carry_over=["candidate"])
    judge = role_model({"judge": lambda t: t.say("ok")})
    with pytest.raises(ValueError, match="evaluator_prompt"):
        ap.create_evaluator_optimizer(spec("g").agent, judge)
    loop = ap.create_evaluator_optimizer(spec("g").agent, echo_agent("r"))  # agent without response_format
    with pytest.raises(ValueError, match="no structured_response"):
        loop.invoke({"messages": [("user", "x")]})


# ------------------------------------------------ structured output method
def test_structured_output_auto_uses_native_json_schema_when_supported():
    calls = []

    def route(t):
        calls.append(("native" if t.response_format else "tool", t.tool_names))
        return t.structured({"routes": [{"agent": "alpha", "task": "go"}]})

    def synth(t):
        calls.append(("native" if t.response_format else "tool", t.tool_names))
        return t.structured(Answer(text="done"))

    def build(**kwargs):
        model = role_model({"router": route, "synth": synth}, profile={"structured_output": True})
        return ap.create_router(
            model,
            [spec("alpha")],
            system_prompt="role=router",
            synthesizer_prompt="role=synth",
            response_format=Answer,
            **kwargs,
        )

    assert build().invoke({"messages": [("user", "x")]})["structured_response"] == Answer(text="done")
    assert calls == [("native", []), ("native", [])]

    calls.clear()
    build(structured_output_method="function_calling").invoke({"messages": [("user", "x")]})
    assert calls == [("tool", ["RoutingDecision"]), ("tool", ["Answer"])]


# ------------------------------------------------------------- composition
def test_agent_as_node_embeds_pattern_in_custom_graph():
    from typing import TypedDict

    class Ticket(TypedDict, total=False):
        question: str
        answer: str

    router = ap.create_router(
        role_model({"router": router_policy(["alpha"])}), [spec("alpha")], system_prompt="role=router"
    )
    node = ap.agent_as_node(
        router,
        input=lambda s: {"messages": [HumanMessage(s["question"])]},
        output=lambda r, s: {"answer": r["messages"][-1].text},
    )
    graph = StateGraph(Ticket).add_node("router", node).add_edge(START, "router").compile()
    assert graph.invoke({"question": "q"})["answer"] == "alpha: task for alpha"


def test_pattern_graph_as_subgraph_node_with_shared_messages():
    parallel = ap.create_parallel([spec("a"), spec("b")])
    graph = StateGraph(MessagesState).add_node("team", parallel).add_edge(START, "team").compile()
    out = graph.invoke({"messages": [("user", "hi")]})
    assert isinstance(out["messages"][-1], AIMessage)
    assert "a: hi" in out["messages"][-1].text and "b: hi" in out["messages"][-1].text


def test_patterns_nest_router_inside_supervisor():
    router = ap.create_router(
        role_model({"router": router_policy(["alpha"])}), [spec("alpha")], system_prompt="role=router"
    )
    model = role_model({"sup": supervisor_policy([("desk", {"task": "handle"})])})
    sup = ap.create_supervisor(
        model, [ap.AgentSpec("desk", "A routed help desk.", router)], system_prompt="role=sup", response_format=Answer
    )
    assert sup.invoke({"messages": [("user", "x")]})["structured_response"].text == "alpha: task for alpha"


def test_async_supervisor_and_router():
    model = role_model({"sup": supervisor_policy([("alpha", {"task": "one"})])})
    sup = ap.create_supervisor(model, [spec("alpha")], system_prompt="role=sup", response_format=Answer)
    out = asyncio.run(sup.ainvoke({"messages": [("user", "x")]}))
    assert out["structured_response"].text == "alpha: one"
    router = ap.create_router(
        role_model({"router": router_policy(["alpha"])}), [spec("alpha")], system_prompt="role=router"
    )
    assert asyncio.run(router.ainvoke({"messages": [("user", "x")]}))["messages"][-1].text == "alpha: task for alpha"


# -------------------------------------------------------------- graph view
@pytest.mark.parametrize(
    "make, expected",
    [
        (lambda: ap.create_router(role_model({}), [spec("a"), spec("b")], route_fn=lambda s: []), {"a", "b"}),
        (lambda: ap.create_parallel([spec("a"), spec("b")]), {"a", "b"}),
        (lambda: ap.create_voting(spec("a")), {"sample"}),
        (lambda: ap.create_map_reduce(echo_agent("m")), {"map"}),
        (lambda: ap.create_orchestrator(role_model({}), [spec("a"), spec("b")]), {"a", "b"}),
        (
            lambda: ap.create_pipeline([ap.agent_step("a", echo_agent("a")), ap.agent_step("b", echo_agent("b"))]),
            {"a", "b"},
        ),
        (lambda: ap.create_evaluator_optimizer(echo_agent("gen"), echo_agent("judge")), {"generate", "evaluate"}),
        (
            lambda: ap.create_swarm(
                role_model({}), [ap.SwarmAgent("a", "A.", "role=a"), ap.SwarmAgent("b", "B.", "role=b")]
            ),
            {"a", "b"},
        ),
    ],
    ids=["router", "parallel", "voting", "map_reduce", "orchestrator", "pipeline", "evaluator_optimizer", "swarm"],
)
def test_participants_are_subgraphs_in_the_graph_view(make, expected):
    graph = make()
    assert {name for name, _ in graph.get_subgraphs()} == expected
    assert {f"{name}:model" for name in expected} <= set(graph.get_graph(xray=True).nodes)


def test_nested_patterns_expand_level_by_level_in_the_graph_view():
    router = ap.create_router(role_model({}), [spec("a")], route_fn=lambda s: [])
    pipeline = ap.create_pipeline([ap.agent_step("desk", router)])
    assert "desk:a:model" in pipeline.get_graph(xray=True).nodes
    assert "desk:a:model" not in pipeline.get_graph(xray=1).nodes  # xray=N limits the depth


def test_interrupted_participant_state_is_visible_from_the_pattern():
    from langgraph.types import Command, interrupt

    @tool
    def refund(amount: int) -> str:
        """Refund an amount."""
        return "refunded" if interrupt(f"Refund {amount}?") else "declined"

    def billing(t):
        return t.call("refund", amount=5) if not t.called("refund") else t.say(f"done: {t.result('refund')}")

    agent = create_agent(role_model({"billing": billing}), tools=[refund], system_prompt="role=billing")
    router = ap.create_router(
        role_model({}),
        [ap.AgentSpec("billing", "Refunds.", agent)],
        route_fn=lambda s: [{"agent": "billing", "task": "refund 5"}],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t"}}
    router.invoke({"messages": [("user", "refund")]}, config)
    inner = router.get_state(config, subgraphs=True).tasks[0].state
    assert inner.next == ("tools",)
    assert inner.values["messages"][-1].tool_calls[0]["args"] == {"amount": 5}
    assert router.invoke(Command(resume=True), config)["messages"][-1].text == "done: refunded"


def test_pipeline_step_needs_either_run_or_agent():
    with pytest.raises(ValueError, match="either `run` or `agent`"):
        ap.PipelineStep("empty")
    with pytest.raises(ValueError, match="either `run` or `agent`"):
        ap.PipelineStep("both", run=lambda state, config: 1, agent=echo_agent("a"))


# ------------------------------------------------------------------ limits
def tool_messages(out, name=None):
    return [m for m in out["messages"] if isinstance(m, ToolMessage) and (name is None or m.name == name)]


def looping_agent(kind: str, **limits):
    """Each factory with an agent whose model calls `add` forever."""
    loop = role_model({"loop": lambda t: t.call("add", a=1, b=1)})
    builders = {
        "single_agent": lambda: ap.create_single_agent(loop, [add], system_prompt="role=loop", **limits),
        "supervisor": lambda: ap.create_supervisor(loop, [], tools=[add], system_prompt="role=loop", **limits),
        "swarm": lambda: ap.create_swarm(loop, [ap.SwarmAgent("a", "A", "role=loop", tools=[add])], **limits),
        "state_machine": lambda: ap.create_state_machine_agent(
            loop, [ap.Step("s", "role=loop", tools=[add])], **limits
        ),
        "skills": lambda: ap.create_skills_agent(
            loop, [ap.Skill("x", "X", "Do x.")], tools=[add], system_prompt="role=loop", **limits
        ),
        "create_agent": lambda: create_agent(
            loop, [add], system_prompt="role=loop", middleware=ap.loop_limits(**limits)
        ),
    }
    return builders[kind]()


@pytest.mark.parametrize("kind", ["single_agent", "supervisor", "swarm", "state_machine", "skills", "create_agent"])
def test_every_agent_loop_can_be_capped(kind):
    with pytest.raises(ap.ModelCallLimitExceededError):
        looping_agent(kind, max_model_calls=3).invoke({"messages": [("user", "go")]})
    out = looping_agent(kind, max_model_calls=3, on_limit="end").invoke({"messages": [("user", "go")]})
    assert len(tool_messages(out, "add")) == 3
    assert "limit" in out["messages"][-1].text.lower()


def test_tool_call_cap_refuses_further_calls_and_lets_the_model_finish():
    def calc(t):
        if any(m.status == "error" for m in t.tool_results()):
            return t.say(f"done after {len(t.tool_results())} results")
        return t.call("add", a=1, b=1)

    agent = ap.create_single_agent(role_model({"calc": calc}), [add], system_prompt="role=calc", max_tool_calls=2)
    out = agent.invoke({"messages": [("user", "go")]})
    assert [m.status for m in tool_messages(out)] == ["success", "success", "error"]
    assert out["messages"][-1].text == "done after 3 results"


def test_swarm_loop_limits_count_per_activation_with_per_agent_override():
    def alice(t):
        if len(t.tool_results("add")) < 3:
            return t.call("add", a=1, b=1)
        return t.call("transfer_to_bob", note="yours")

    swarm = ap.create_swarm(
        role_model({"alice": alice, "bob": lambda t: t.call("add", a=2, b=2)}),
        [
            ap.SwarmAgent("alice", "A", "role=alice", tools=[add]),
            ap.SwarmAgent("bob", "B", "role=bob", tools=[add], max_model_calls=2),
        ],
        max_model_calls=5,
        on_limit="end",
    )
    out = swarm.invoke({"messages": [("user", "go")]})
    assert len(tool_messages(out, "add")) == 3 + 2  # alice: 4 calls (< 5), bob: stopped after his 2
    assert "limit" in out["messages"][-1].text.lower()


def test_swarm_max_activations_caps_how_often_an_agent_takes_control():
    def ping(t):
        if t.tool_results() and "already had control" in t.tool_results()[-1].text:
            return t.say("finishing myself")
        return t.call("transfer_to_bob" if t.has_tool("transfer_to_bob") else "transfer_to_alice", note="yours")

    swarm = ap.create_swarm(
        role_model({"alice": ping, "bob": ping}),
        [ap.SwarmAgent("alice", "A", "role=alice", max_activations=2), ap.SwarmAgent("bob", "B", "role=bob")],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "1"}}
    out = swarm.invoke({"messages": [("user", "go")]}, config)
    assert out["activations"] == {"alice": 2, "bob": 2}  # the third handoff to alice was refused
    assert out["active_agent"] == "bob"
    assert out["messages"][-1].text == "finishing myself"
    again = swarm.invoke({"messages": [("user", "next")]}, config)
    assert again["activations"] == {"bob": 1}  # counted per run


def test_state_machine_max_visits_refuses_transitions_into_a_used_up_step():
    def agent(t):
        here = "a" if "step=a" in t.system else "b"
        if t.tool_results() and "refused" in t.tool_results()[-1].text:
            return t.say(f"done in {here}")
        return t.call("go_to_b" if here == "a" else "go_to_a", reason="ping")

    graph = ap.create_state_machine_agent(
        role_model({"sm": agent}),
        [
            ap.Step("a", "role=sm step=a", transitions=["b"], max_visits=2),
            ap.Step("b", "role=sm step=b", transitions=["a"]),
        ],
    )
    out = graph.invoke({"messages": [("user", "go")]})
    assert out["step_visits"] == {"a": 2, "b": 2}  # the start in `a` counts
    assert out["current_step"] == "b"
    assert out["messages"][-1].text == "done in b"


def test_state_machine_visit_limit_undoes_a_custom_tools_transition_but_keeps_its_work():
    class State(ap.StateMachineState):
        attempts: NotRequired[int]

    @tool
    def submit(runtime: ToolRuntime):
        """Submit the work for review."""
        attempt = (runtime.state.get("attempts") or 0) + 1
        return ap.transition("review", runtime.tool_call_id, f"Submitted attempt {attempt}.", attempts=attempt)

    def agent(t):
        if "step=review" in t.system:
            return t.call("go_to_work", reason="needs changes")
        if t.tool_results() and "refused" in t.tool_results()[-1].text:
            return t.say(t.tool_results()[-1].text)
        return t.call("submit")

    graph = ap.create_state_machine_agent(
        role_model({"sm": agent}),
        [
            ap.Step("work", "role=sm step=work", tools=[submit]),
            ap.Step("review", "role=sm step=review", transitions=["work"], max_visits=1),
        ],
        state_schema=State,
    )
    out = graph.invoke({"messages": [("user", "go")]})
    assert out["attempts"] == 2  # the tool's own update is kept
    assert out["current_step"] == "work"
    assert out["step_visits"] == {"work": 2, "review": 1}
    assert out["messages"][-1].text.startswith("Submitted attempt 2.\n\nTransition to step 'review' refused")


def test_supervisor_max_calls_per_agent_counts_parallel_calls():
    calls = [("alpha", {"task": "1"}), ("alpha", {"task": "2"}), ("alpha", {"task": "3"}), ("beta", {"task": "4"})]

    def supervise(t):
        return t.call_many(*calls) if not t.tool_results() else t.say("done")

    sup = ap.create_supervisor(
        role_model({"sup": supervise}), [spec("alpha"), spec("beta")], system_prompt="role=sup", max_calls_per_agent=2
    )
    out = sup.invoke({"messages": [("user", "x")]})
    by_id = {m.tool_call_id: m for m in tool_messages(out)}
    results = [by_id[c["id"]] for c in out["messages"][1].tool_calls]
    assert [m.text for m in results[:2]] == ["alpha: 1", "alpha: 2"]
    assert results[2].status == "error" and results[2].text.startswith("Delegation limit reached: 'alpha'")
    assert results[3].text == "beta: 4"


def test_delegation_and_tool_caps_refuse_a_call_only_once():
    def supervise(t):
        return t.call_many(("alpha", {"task": "1"}), ("alpha", {"task": "2"})) if not t.tool_results() else t.say("ok")

    sup = ap.create_supervisor(
        role_model({"sup": supervise}),
        [spec("alpha")],
        system_prompt="role=sup",
        max_tool_calls=1,
        max_calls_per_agent=1,
    )
    out = sup.invoke({"messages": [("user", "x")]})
    ids = [m.tool_call_id for m in tool_messages(out)]
    assert len(ids) == len(set(ids)) == 2  # one answer per call, as providers require


def test_supervisor_max_calls_per_agent_with_task_tool_across_turns():
    def supervise(t):
        done = t.tool_results()
        if len(done) < 3:
            agent = "alpha" if len(done) < 2 else "beta"
            return t.call("task", agent_name=agent, description=f"job {len(done)}")
        return t.say(" | ".join(m.text for m in done))

    sup = ap.create_supervisor(
        role_model({"sup": supervise}),
        [spec("alpha"), spec("beta")],
        system_prompt="role=sup",
        delegation="task_tool",
        max_calls_per_agent={"alpha": 1},
    )
    answer = sup.invoke({"messages": [("user", "x")]})["messages"][-1].text
    assert answer.startswith("alpha: job 0 | Delegation limit reached: 'alpha'")
    assert answer.endswith("beta: job 2")


def test_max_calls_per_agent_validates_names_and_reaches_nested_teams():
    model = role_model({"sup": lambda t: t.say("-")})
    with pytest.raises(ValueError, match="unknown"):
        ap.create_supervisor(model, [spec("alpha")], system_prompt="role=sup", max_calls_per_agent={"alhpa": 1})

    def lead(t):
        if not t.tool_results():
            return t.call_many(("alpha", {"task": "1"}), ("alpha", {"task": "2"}))
        return t.say(" | ".join(sorted(m.text for m in t.tool_results())))

    team = ap.Team(
        "team", "A team.", system_prompt="role=lead", members=[spec("alpha")], model=role_model({"lead": lead})
    )
    top = role_model({"top": supervisor_policy([("team", {"task": "delegate"})])})
    hierarchy = ap.create_hierarchy(
        top, [team], system_prompt="role=top", response_format=Answer, max_calls_per_agent={"alpha": 1}
    )
    report = hierarchy.invoke({"messages": [("user", "x")]})["structured_response"].text
    assert "alpha: 1" in report and "alpha: 2" not in report and "Delegation limit reached" in report
    with pytest.raises(ValueError, match="unknown"):
        ap.create_hierarchy(top, [team], system_prompt="role=top", max_calls_per_agent={"nobody": 1})


def budgeted_supervisor():
    """4 model calls (plan, alpha, beta, answer) and 2 tool calls (the delegations)."""
    model = role_model({"sup": supervisor_policy([("alpha", {"task": "one"}), ("beta", {"task": "two"})])})
    return ap.create_supervisor(model, [spec("alpha"), spec("beta")], system_prompt="role=sup", response_format=Answer)


def test_run_budget_counts_calls_across_nested_agents_and_pattern_internals():
    budget = ap.RunBudget(max_model_calls=4, max_tool_calls=2)
    out = budgeted_supervisor().invoke({"messages": [("user", "x")]}, {"callbacks": [budget]})
    assert out["structured_response"].text == "alpha: one | beta: two"
    assert (budget.model_calls, budget.tool_calls) == (4, 2)

    budget.reset()
    router = ap.create_router(
        role_model({"router": router_policy(["alpha"])}), [spec("alpha")], system_prompt="role=router"
    )
    router.invoke({"messages": [("user", "x")]}, {"callbacks": [budget]})
    assert budget.model_calls == 2  # routing decision + alpha


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_run_budget_stops_the_whole_run(mode):
    budget, graph = ap.RunBudget(max_model_calls=2), budgeted_supervisor()
    inputs, config = {"messages": [("user", "x")]}, {"callbacks": [budget]}
    with pytest.raises(ap.BudgetExceededError, match="at most 2 model calls"):
        graph.invoke(inputs, config) if mode == "sync" else asyncio.run(graph.ainvoke(inputs, config))
    assert budget.model_calls == 2  # the call over budget was never made


def test_run_budget_tool_cap_and_validation():
    with pytest.raises(ap.BudgetExceededError) as error:
        budgeted_supervisor().invoke({"messages": [("user", "x")]}, {"callbacks": [ap.RunBudget(max_tool_calls=1)]})
    assert error.value.kind == "tool"
    with pytest.raises(ValueError):
        ap.RunBudget()


# ------------------------------------------------------ evaluator_input
class Verdict(BaseModel):
    """Review verdict."""

    passed: bool


def test_evaluator_input_replaces_what_the_evaluator_is_sent():
    sent = []

    def review(t):
        sent.append([type(m).__name__ for m in t.messages])
        sent.append(t.last_human)
        return t.structured(Verdict(passed=True))

    reviewer = create_agent(role_model({"rev": review}), tools=[], system_prompt="role=rev", response_format=Verdict)
    loop = ap.create_evaluator_optimizer(
        reflective_generator(),
        reviewer,
        passed=lambda v: v.passed,
        evaluator_input=lambda s: [*s["request"], HumanMessage(f"Check this draft: {s['candidate'].text}")],
    )
    out = loop.invoke({"messages": [("user", "write something")]})
    assert out["structured_response"] == Answer(text="draft v1")
    assert sent == [["SystemMessage", "HumanMessage", "HumanMessage"], "Check this draft: draft v1"]


def test_evaluator_input_for_a_model_evaluator_keeps_its_rubric():
    seen = []

    def judge(t):
        seen.append((t.system, t.human_messages))
        return t.structured({"passed": True, "score": 1, "issues": [], "feedback": ""})

    loop = ap.create_evaluator_optimizer(
        reflective_generator(),
        role_model({"judge": judge}),
        evaluator_prompt="role=judge",
        evaluator_input=lambda s: f"Only the draft: {s['candidate'].text}",
    )
    loop.invoke({"messages": [("user", "write something")]})
    assert seen == [("role=judge", ["Only the draft: draft v1"])]


def test_evaluator_input_excludes_evaluator_context():
    with pytest.raises(ValueError, match="evaluator_input"):
        ap.create_evaluator_optimizer(
            reflective_generator(), echo_agent("r"), evaluator_input=lambda s: "x", evaluator_context=lambda s: "y"
        )


# ----------------------------------------------------------- retry_policy
def flaky(agent, failures: int):
    """Wraps an agent so its first `failures` calls raise."""
    calls = itertools.count()

    def run(state, config):
        if next(calls) < failures:
            raise ConnectionError("flaky")
        return agent.invoke(state, config)

    from langchain_core.runnables import RunnableLambda

    return RunnableLambda(run, name="flaky")


def test_retry_policy_retries_only_the_step_that_failed():
    from langgraph.types import RetryPolicy

    generated = []

    def write(t):
        generated.append(1)
        return t.structured(Answer(text="draft"))

    generator = create_agent(role_model({"gen": write}), tools=[], system_prompt="role=gen", response_format=Answer)
    reviewer = create_agent(
        role_model({"rev": lambda t: t.structured(Verdict(passed=True))}),
        tools=[],
        system_prompt="role=rev",
        response_format=Verdict,
    )
    retry = RetryPolicy(retry_on=ConnectionError, initial_interval=0, jitter=False)
    loop = ap.create_evaluator_optimizer(
        generator, flaky(reviewer, failures=1), passed=lambda v: v.passed, retry_policy=retry
    )
    assert loop.invoke({"messages": [("user", "x")]})["evaluation"].passed
    assert len(generated) == 1  # the failed review ran again, the generation did not

    without = ap.create_evaluator_optimizer(generator, flaky(reviewer, failures=1), passed=lambda v: v.passed)
    with pytest.raises(ConnectionError):
        without.invoke({"messages": [("user", "x")]})


def hangs(agent, calls: int):
    """Wraps an agent so its first `calls` (async) calls hang until they are cancelled."""
    counter = itertools.count()

    async def run(state, config):
        if next(counter) < calls:
            await asyncio.Event().wait()  # never set: only a timeout ends this call
        return await agent.ainvoke(state, config)

    from langchain_core.runnables import RunnableLambda

    return RunnableLambda(run, name="hangs")


def one_worker_graph(factory: str, worker: ap.AgentSpec, **policies):
    """`factory` with `worker` as its only participant and `policies` (retry_policy, timeout)."""
    return {
        "parallel": lambda: ap.create_parallel([worker], **policies),
        "pipeline": lambda: ap.create_pipeline([ap.agent_step("w", worker.agent)], **policies),
        "router": lambda: ap.create_router(
            role_model({"router": router_policy(["w"]), "synth": lambda t: t.say("done")}),
            [worker],
            system_prompt="role=router",
            synthesizer_prompt="role=synth",
            **policies,
        ),
        "orchestrator": lambda: ap.create_orchestrator(
            role_model(
                {
                    "plan": lambda t: t.structured({"tasks": [{"worker": "w", "instruction": "go"}]}),
                    "synth": lambda t: t.say("done"),
                }
            ),
            [worker],
            planner_prompt="role=plan",
            synthesizer_prompt="role=synth",
            **policies,
        ),
        "voting": lambda: ap.create_voting(worker, n=1, **policies),
    }[factory]()


FACTORIES = ["parallel", "pipeline", "router", "orchestrator", "voting"]


@pytest.mark.parametrize("factory", FACTORIES)
def test_retry_policy_on_the_other_graph_factories(factory):
    from langgraph.types import RetryPolicy

    retry = RetryPolicy(retry_on=ConnectionError, initial_interval=0, jitter=False)
    worker = ap.AgentSpec("w", "The worker.", flaky(echo_agent("w"), failures=1))
    graph = one_worker_graph(factory, worker, retry_policy=retry)
    assert graph.invoke({"messages": [("user", "x")]})["messages"][-1].text


@pytest.mark.parametrize("factory", FACTORIES)
def test_timeout_cancels_a_hung_step_and_the_retry_runs_it_again(factory):
    from langgraph.types import RetryPolicy

    retry = RetryPolicy(initial_interval=0, jitter=False)  # NodeTimeoutError is retryable by default
    worker = ap.AgentSpec("w", "The worker.", hangs(echo_agent("w"), calls=1))
    graph = one_worker_graph(factory, worker, retry_policy=retry, timeout=0.2)
    out = asyncio.run(asyncio.wait_for(graph.ainvoke({"messages": [("user", "x")]}), 10))
    assert out["messages"][-1].text


def test_a_timeout_without_retries_fails_the_run_and_needs_an_async_run():
    from langgraph.errors import NodeTimeoutError

    graph = ap.create_parallel([ap.AgentSpec("w", "The worker.", hangs(echo_agent("w"), calls=1))], timeout=0.2)
    with pytest.raises(NodeTimeoutError):
        asyncio.run(asyncio.wait_for(graph.ainvoke({"messages": [("user", "x")]}), 10))
    with pytest.raises(ValueError, match="only supported for async nodes"):
        graph.invoke({"messages": [("user", "x")]})


def test_a_swarm_with_a_timeout_still_hands_off():
    swarm = ap.create_swarm(
        swarm_model(),
        [ap.SwarmAgent("alice", "Front desk", "role=alice"), ap.SwarmAgent("bob", "Billing", "role=bob")],
        timeout=5,
    )
    out = asyncio.run(swarm.ainvoke({"messages": [("user", "billing issue")]}))
    assert (out["messages"][-1].text, out["active_agent"]) == ("bob answers", "bob")


def test_run_budget_counts_every_attempt_of_a_retried_step():
    from langgraph.types import RetryPolicy

    attempts = itertools.count()

    def route(turn):
        if next(attempts) == 0:
            raise ConnectionError("provider hiccup")
        return router_policy(["alpha"])(turn)

    router = ap.create_router(
        role_model({"router": route}),
        [spec("alpha")],
        system_prompt="role=router",
        retry_policy=RetryPolicy(retry_on=ConnectionError, initial_interval=0, jitter=False),
    )
    budget = ap.RunBudget(max_model_calls=10)
    router.invoke({"messages": [("user", "x")]}, {"callbacks": [budget]})
    assert budget.model_calls == 3  # the failed routing call, its retry, alpha


# ------------------------------------------------------------ checkpointer
@pytest.mark.parametrize(
    "make",
    [
        lambda cp: ap.create_single_agent(
            role_model({"echo": lambda t: t.say("hi")}), [], system_prompt="role=echo", checkpointer=cp
        ),
        lambda cp: ap.create_parallel([spec("a")], checkpointer=cp),
        lambda cp: ap.create_pipeline([ap.agent_step("a", spec("a").agent)], checkpointer=cp),
        lambda cp: ap.create_evaluator_optimizer(
            reflective_generator(),
            role_model({"judge": lambda t: t.structured({"passed": True, "score": 1, "issues": [], "feedback": ""})}),
            evaluator_prompt="role=judge",
            checkpointer=cp,
        ),
    ],
    ids=["single_agent", "parallel", "pipeline", "evaluator_optimizer"],
)
def test_factories_take_a_checkpointer(make):
    graph = make(InMemorySaver())
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"messages": [("user", "x")]}, config)
    assert graph.get_state(config).values["messages"]


# ------------------------------------------------------------- run context
@dataclass
class Tenant:
    name: str


@tool
def whoami(runtime: ToolRuntime[Tenant]) -> str:
    """The tenant of the run."""
    return runtime.context.name


@pytest.mark.parametrize("pattern", ["supervisor", "parallel", "pipeline", "router", "evaluator_optimizer"])
@pytest.mark.parametrize("mode", ["sync", "async"])
def test_the_run_context_reaches_agents_inside_every_pattern(pattern, mode, recwarn):
    def ask(t):
        return t.call("whoami") if not t.called("whoami") else t.say(f"tenant {t.result('whoami')}")

    worker = create_agent(role_model({"w": ask}), tools=[whoami], system_prompt="role=w", name="w")
    judge = create_agent(
        role_model({"judge": lambda t: t.structured(Verdict(passed=True))}),
        tools=[],
        system_prompt="role=judge",
        response_format=Verdict,
    )

    def lead(t):
        return t.call("w", task="who?") if not t.tool_results() else t.say(t.tool_results()[-1].text)

    graph = {
        "supervisor": lambda: ap.create_supervisor(
            role_model({"lead": lead}), [ap.AgentSpec("w", "Asks.", worker)], system_prompt="role=lead"
        ),
        "parallel": lambda: ap.create_parallel([ap.AgentSpec("w", "Asks.", worker)]),
        "pipeline": lambda: ap.create_pipeline([ap.agent_step("w", worker)]),
        "router": lambda: ap.create_router(
            role_model({"r": lambda t: t.say("unused")}),
            [ap.AgentSpec("w", "Asks.", worker)],
            route_fn=lambda s: [{"agent": "w", "task": "who?"}],
            synthesizer_prompt="role=r",
        ),
        "evaluator_optimizer": lambda: ap.create_evaluator_optimizer(worker, judge, passed=lambda v: v.passed),
    }[pattern]()
    inputs, context = {"messages": [("user", "who?")]}, Tenant(f"{pattern}-{mode}")
    out = (
        graph.invoke(inputs, context=context) if mode == "sync" else asyncio.run(graph.ainvoke(inputs, context=context))
    )
    assert f"tenant {pattern}-{mode}" in json.dumps(
        [m.text for m in out["messages"]] + [str(out.get("structured_response"))]
    )
    assert not [w for w in recwarn if "serializ" in str(w.message).lower()]
