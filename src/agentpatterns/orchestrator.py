"""Orchestrator-workers (with optional re-planning, i.e. plan-and-execute).

An orchestrator LLM decomposes the request into tasks *at runtime* using
structured output (`Plan`). Each task runs on a worker in parallel (`Send`).
With `max_rounds > 1` the orchestrator sees all results and may plan further
rounds (e.g. research first, then act); an empty plan ends the loop. A
synthesizer finally turns all results into the answer.

Earlier work can be passed in as `results` (e.g. by a review loop with
`carry_over=["results"]`): the planner then sees it and plans only what is
still missing, and the synthesizer uses old and new results together.

    orchestrator = create_orchestrator(
        model,
        workers=[AgentSpec("researcher", "Looks things up", research_agent),
                 AgentSpec("operator", "Executes changes", ops_agent)],
        planner_prompt="Plan the work for the request...",
        max_rounds=3,
        response_format=Answer,
    )

Difference to the router: tasks are *derived* from the request (and from
earlier results), not a classification of it; the same worker may get several
tasks. Difference to the supervisor: planning is explicit (inspectable plan in
state) and execution is a graph, not an agent loop.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer, Send
from pydantic import BaseModel, Field, create_model

from agentpatterns.core import (
    AgentResult,
    AgentSpec,
    PatternInput,
    PatternState,
    RetryPolicies,
    StepTimeout,
    StructuredOutputMethod,
    catalog,
    dual_node,
    literal_of,
    llm_output,
    results_block,
    spec_node,
    step_policies,
    structured_llm,
)

__all__ = ["OrchestratorInput", "create_orchestrator"]

DEFAULT_PLANNER_PROMPT = (
    "You are an orchestrator. Break the user's request into concrete, self-contained tasks and assign "
    "each task to the most suitable worker."
)
REPLAN_SUFFIX = (
    "\n\nYou work in rounds. The results of earlier rounds are shown to you. Plan only the tasks that are "
    "still needed; return an empty task list when the request is fully handled."
)
DEFAULT_SYNTHESIZER_PROMPT = "Write the final answer to the user's request based on the work results below."


class OrchestratorState(PatternState):
    round: NotRequired[int]
    last_round: NotRequired[int]  # round limit of the current run
    plan: NotRequired[list[dict[str, str]]]
    results: Annotated[list[dict[str, Any]], operator.add]


class OrchestratorInput(PatternInput):
    results: NotRequired[list[dict[str, Any]]]  # earlier work to build on


class OrchestratorOutput(PatternInput):
    structured_response: NotRequired[Any]
    results: list[dict[str, Any]]


class _TaskInput(TypedDict):
    instruction: str
    round: int


def _plan_schema(names: Sequence[str]) -> type[BaseModel]:
    task = create_model(
        "PlannedTask",
        worker=(literal_of(names), Field(description="Worker that executes the task.")),
        instruction=(str, Field(description="Explicit, self-contained instructions.")),
    )
    return create_model(
        "Plan",
        __doc__="Tasks for the next round. An empty list means the request is fully handled.",
        tasks=(list[task], Field(default_factory=list)),
    )


def create_orchestrator(
    model: BaseChatModel,
    workers: Sequence[AgentSpec],
    *,
    planner_prompt: str = DEFAULT_PLANNER_PROMPT,
    synthesizer_prompt: str = DEFAULT_SYNTHESIZER_PROMPT,
    response_format: type | None = None,
    max_rounds: int = 1,
    max_tasks_per_round: int = 10,
    structured_output_method: StructuredOutputMethod = "auto",
    retry_policy: RetryPolicies = None,
    timeout: StepTimeout = None,
    checkpointer: Checkpointer = None,
    name: str = "orchestrator",
) -> CompiledStateGraph:
    """Build an orchestrator-workers graph following the agent contract.

    Args:
        model: Model for planning and synthesis.
        workers: Worker agents (name, description, agent-contract runnable).
        planner_prompt: Planning instructions; the worker catalog is appended.
        synthesizer_prompt: Instructions for the final answer.
        response_format: Schema of the final answer (`structured_response`).
        max_rounds: Planning rounds (1 = classic orchestrator-workers; >1 = re-planning).
        max_tasks_per_round: Safety cap on fan-out width.
        structured_output_method: How planning and synthesis produce structured
            output (see `StructuredOutputMethod`; "auto" = native where supported).
        retry_policy: Retries of the steps that run an agent or a model when they raise
            (see `RetryPolicies`).
        timeout: Time limit of one attempt of such a step; async runs only (see
            `StepTimeout`).
        checkpointer: Checkpoints the graph; only the outermost graph needs one.
        name: Graph name.

    Input keys: `messages`, optionally `results` (earlier work, same format as the output).
    Output keys: `messages`, `structured_response`, `results`
    (list of `{"round", "worker", "task", "output"}`, including the earlier work).
    `max_rounds` counts the planning rounds of one run; round numbers continue
    after the earlier results.
    """
    specs = {w.name: w for w in workers}
    if clash := {"prepare", "plan", "synthesize", "collect"}.intersection(specs):
        raise ValueError(f"Worker names {clash} are reserved.")
    planner = structured_llm(model, _plan_schema(list(specs)), structured_output_method)
    synth_llm = structured_llm(model, response_format, structured_output_method)
    system = f"{planner_prompt}\n\nWorkers:\n{catalog(workers)}"

    def prepare(state: dict[str, Any]) -> dict[str, Any]:
        """Per-run bookkeeping: continue numbering after earlier results, reset the round limit."""
        done = max((r.get("round", 0) for r in state.get("results", [])), default=0)
        return {"round": done, "last_round": done + max_rounds, "plan": []}

    def planning_messages(state: dict[str, Any]) -> list:
        replanning = max_rounds > 1 or bool(state.get("results"))
        messages = [SystemMessage(system + (REPLAN_SUFFIX if replanning else "")), *state["messages"]]
        if state.get("results"):
            messages.append(HumanMessage(results_block(state["results"], "Results so far")))
        return messages

    def to_update(state: dict[str, Any], plan: Any) -> dict[str, Any]:
        tasks = [{"worker": t.worker, "instruction": t.instruction} for t in plan.tasks][:max_tasks_per_round]
        return {"plan": tasks, "round": state["round"] + 1}

    def plan_sync(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        return to_update(state, planner.invoke(planning_messages(state), config))

    async def plan_async(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        return to_update(state, await planner.ainvoke(planning_messages(state), config))

    def assign(state: dict[str, Any]) -> list[Send] | str:
        tasks = [t for t in state.get("plan", []) if t["worker"] in specs]
        if not tasks or state["round"] > state["last_round"]:
            return "synthesize"
        return [Send(t["worker"], {"instruction": t["instruction"], "round": state["round"]}) for t in tasks]

    def make_worker(spec: AgentSpec):
        def record(result: AgentResult, state: _TaskInput) -> dict[str, Any]:
            return {
                "results": [
                    {"round": state["round"], "worker": spec.name, "task": state["instruction"], "output": result.text}
                ]
            }

        return spec_node(spec, lambda state: state["instruction"], record)

    def after_workers(state: dict[str, Any]) -> str:
        return "plan" if state["round"] < state["last_round"] else "synthesize"

    def synthesis_messages(state: dict[str, Any]) -> list:
        return [
            SystemMessage(synthesizer_prompt),
            *state["messages"],
            HumanMessage(results_block(state.get("results", []), "Work results")),
        ]

    def finish(value: Any) -> dict[str, Any]:
        message, structured = llm_output(value, name)
        update: dict[str, Any] = {"messages": [message]}
        if response_format is not None:
            update["structured_response"] = structured
        return update

    def synth_sync(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        return finish(synth_llm.invoke(synthesis_messages(state), config))

    async def synth_async(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        return finish(await synth_llm.ainvoke(synthesis_messages(state), config))

    policies = step_policies(retry_policy, timeout)
    builder = StateGraph(OrchestratorState, input_schema=OrchestratorInput, output_schema=OrchestratorOutput)
    builder.add_node("prepare", prepare)
    builder.add_node("plan", dual_node(plan_sync, plan_async, "plan"), **policies)
    builder.add_node("synthesize", dual_node(synth_sync, synth_async, "synthesize"), **policies)
    builder.add_node("collect", lambda state: {})  # barrier: all workers of a round finish first
    for spec in workers:
        builder.add_node(spec.name, make_worker(spec), **policies)
        builder.add_edge(spec.name, "collect")
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "plan")
    builder.add_conditional_edges("plan", assign, [*specs, "synthesize"])
    builder.add_conditional_edges("collect", after_workers, ["plan", "synthesize"])
    builder.add_edge("synthesize", END)
    return builder.compile(name=name, checkpointer=checkpointer)
