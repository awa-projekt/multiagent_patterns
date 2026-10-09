"""Parallelization - fan-out / fan-in in three flavours.

* `create_parallel` (sectioning): different agents work on the *same* input at
  the same time; an aggregator (code or LLM) merges their results.
* `create_voting`: the *same* agent runs N times; the majority answer wins
  (use a sampling temperature > 0 for real diversity).
* `create_map_reduce`: one runnable is mapped over a list of items with
  `Send`; a reducer combines the per-item results.

    triage = create_parallel(
        [AgentSpec("sentiment", "...", sentiment_agent), AgentSpec("entities", "...", entity_agent)],
        aggregator=lambda results, state: {n: r.structured for n, r in results.items()},
    )
"""

from __future__ import annotations

import operator
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer, Send

from agentpatterns.core import (
    AgentResult,
    AgentSpec,
    PatternInput,
    PatternState,
    RetryPolicies,
    StepTimeout,
    StructuredOutputMethod,
    dual_node,
    llm_output,
    merge_dicts,
    results_block,
    spec_node,
    step_policies,
    structured_llm,
    to_text,
)

__all__ = ["create_map_reduce", "create_parallel", "create_voting"]

Aggregator = Callable[[dict[str, AgentResult], dict[str, Any]], Any]


# ---------------------------------------------------------------- sectioning
class ParallelState(PatternState):
    branch_results: Annotated[dict[str, AgentResult], merge_dicts]


class ParallelOutput(PatternInput):
    structured_response: NotRequired[Any]
    branch_results: dict[str, AgentResult]


def create_parallel(
    branches: Sequence[AgentSpec],
    *,
    aggregator: Aggregator | None = None,
    model: BaseChatModel | None = None,
    synthesizer_prompt: str = "Combine the parallel analyses below into one answer to the user's request.",
    response_format: type | None = None,
    structured_output_method: StructuredOutputMethod = "auto",
    retry_policy: RetryPolicies = None,
    timeout: StepTimeout = None,
    checkpointer: Checkpointer = None,
    name: str = "parallel",
) -> CompiledStateGraph:
    """Run all branches concurrently on the same input, then aggregate.

    Aggregation (first match wins):
        `aggregator(results_by_name, state) -> value` - deterministic code;
        `model` - an LLM synthesizes the results (structured if `response_format`);
        otherwise the branch outputs are concatenated.
    Every branch receives the full input messages. `structured_output_method`
    applies to the synthesizer (see `StructuredOutputMethod`).
    `retry_policy` retries a step that runs an agent or a model when it raises (see
    `RetryPolicies`), `timeout` limits one attempt of such a step in async runs (see
    `StepTimeout`); `checkpointer` checkpoints the graph, only the outermost needs one.
    """
    if len({b.name for b in branches}) != len(branches):
        raise ValueError("Branch names must be unique.")
    synth_llm = structured_llm(model, response_format, structured_output_method) if model is not None else None

    def make_branch(spec: AgentSpec):
        return spec_node(
            spec, lambda state: state["messages"], lambda result, state: {"branch_results": {spec.name: result}}
        )

    def synthesis_messages(state: dict[str, Any]) -> list:
        results = [state["branch_results"][b.name] for b in branches]
        return [
            SystemMessage(synthesizer_prompt),
            *state["messages"],
            HumanMessage(results_block(results, "Parallel results")),
        ]

    def finish(value: Any) -> dict[str, Any]:
        message, structured = llm_output(value, name)
        return {"messages": [message], "structured_response": structured if structured is not None else value}

    def agg_sync(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        results = state["branch_results"]
        if aggregator is not None:
            value = aggregator(results, state)
            return {"messages": [AIMessage(content=to_text(value), name=name)], "structured_response": value}
        if synth_llm is not None:
            return finish(synth_llm.invoke(synthesis_messages(state), config))
        text = "\n\n".join(f"[{b.name}]\n{results[b.name].text}" for b in branches)
        return {"messages": [AIMessage(content=text, name=name)]}

    async def agg_async(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        if aggregator is None and synth_llm is not None:
            return finish(await synth_llm.ainvoke(synthesis_messages(state), config))
        return agg_sync(state, config)

    policies = step_policies(retry_policy, timeout)
    builder = StateGraph(ParallelState, input_schema=PatternInput, output_schema=ParallelOutput)
    for spec in branches:
        builder.add_node(spec.name, make_branch(spec), **policies)
        builder.add_edge(START, spec.name)
    builder.add_node("aggregate", dual_node(agg_sync, agg_async, "aggregate"), **policies)
    builder.add_edge([b.name for b in branches], "aggregate")  # fan-in waits for all
    builder.add_edge("aggregate", END)
    return builder.compile(name=name, checkpointer=checkpointer)


# -------------------------------------------------------------------- voting
class VotingState(PatternState):
    samples: Annotated[list[tuple[int, AgentResult]], operator.add]
    votes: NotRequired[dict[str, int]]


class VotingOutput(PatternInput):
    structured_response: NotRequired[Any]
    votes: dict[str, int]


def create_voting(
    agent: AgentSpec,
    *,
    n: int = 3,
    key: Callable[[AgentResult], Hashable] = lambda r: r.text,
    retry_policy: RetryPolicies = None,
    timeout: StepTimeout = None,
    checkpointer: Checkpointer = None,
    name: str = "voting",
) -> CompiledStateGraph:
    """Run the same agent `n` times in parallel and return the majority answer.

    `key(result)` extracts what is voted on (e.g. `lambda r: r.structured.category`).
    Ties go to the earliest sample. Output: the winning sample's messages and
    structured response plus `votes` (key -> count).
    `retry_policy` retries a step that runs an agent or a model when it raises (see
    `RetryPolicies`), `timeout` limits one attempt of such a step in async runs (see
    `StepTimeout`); `checkpointer` checkpoints the graph, only the outermost needs one.
    """

    def fan_out(state: dict[str, Any]) -> list[Send]:
        return [Send("sample", {"index": i, "messages": state["messages"]}) for i in range(n)]

    def tally(state: dict[str, Any]) -> dict[str, Any]:
        samples = [r for _, r in sorted(state["samples"], key=lambda s: s[0])]
        counts = Counter(key(r) for r in samples)
        best = max(counts.values())
        winner = next(r for r in samples if counts[key(r)] == best)
        update: dict[str, Any] = {
            "messages": [AIMessage(content=winner.text, name=name)],
            "votes": {str(k): v for k, v in counts.items()},
        }
        if winner.structured is not None:
            update["structured_response"] = winner.structured
        return update

    policies = step_policies(retry_policy, timeout)
    builder = StateGraph(VotingState, input_schema=PatternInput, output_schema=VotingOutput)
    sample = spec_node(
        agent,
        lambda state: state["messages"],
        lambda result, state: {"samples": [(state["index"], result)]},
        name="sample",
    )
    builder.add_node("sample", sample, **policies)
    builder.add_node("tally", tally)
    builder.add_conditional_edges(START, fan_out, ["sample"])
    builder.add_edge("sample", "tally")
    builder.add_edge("tally", END)
    return builder.compile(name=name, checkpointer=checkpointer)


# ---------------------------------------------------------------- map-reduce
class MapReduceInput(TypedDict):
    items: list[Any]


class MapReduceState(MapReduceInput):
    mapped: Annotated[list[tuple[int, Any]], operator.add]  # (index, result) from parallel branches
    results: NotRequired[list[Any]]
    output: NotRequired[Any]


class MapReduceOutput(TypedDict):
    results: list[Any]
    output: Any


class _Item(TypedDict):
    index: int
    item: Any


def create_map_reduce(
    mapper: Runnable,
    *,
    reduce: Callable[[list[Any]], Any] = lambda results: results,
    prepare: Callable[[Any], Any] = lambda item: item,
    extract: Callable[[Any], Any] = lambda output: output,
    retry_policy: RetryPolicies = None,
    timeout: StepTimeout = None,
    checkpointer: Checkpointer = None,
    name: str = "map_reduce",
) -> CompiledStateGraph:
    """Map `mapper` over `items` in parallel (one `Send` per item), then reduce.

    Input `{"items": [...]}` -> output `{"results": [...], "output": reduce(results)}`.
    `prepare(item)` builds the mapper input, `extract(mapper_output)` the per-item
    result; `results` keep the order of `items`. Any runnable works as mapper -
    including complete pattern graphs or your own workflows.
    `retry_policy` retries a step that runs an agent or a model when it raises (see
    `RetryPolicies`), `timeout` limits one attempt of such a step in async runs (see
    `StepTimeout`); `checkpointer` checkpoints the graph, only the outermost needs one.
    """

    def fan_out(state: dict[str, Any]) -> list[Send] | str:
        if not state["items"]:
            return "reduce"
        return [Send("map", {"index": i, "item": item}) for i, item in enumerate(state["items"])]

    def map_sync(state: _Item, config: RunnableConfig) -> dict[str, Any]:
        return {"mapped": [(state["index"], extract(mapper.invoke(prepare(state["item"]), config)))]}

    async def map_async(state: _Item, config: RunnableConfig) -> dict[str, Any]:
        return {"mapped": [(state["index"], extract(await mapper.ainvoke(prepare(state["item"]), config)))]}

    def reduce_node(state: dict[str, Any]) -> dict[str, Any]:
        ordered = [r for _, r in sorted(state.get("mapped", []), key=lambda r: r[0])]
        return {"results": ordered, "output": reduce(ordered)}

    policies = step_policies(retry_policy, timeout)
    builder = StateGraph(MapReduceState, input_schema=MapReduceInput, output_schema=MapReduceOutput)
    builder.add_node("map", dual_node(map_sync, map_async, "map"), **policies)
    builder.add_node("reduce", reduce_node)
    builder.add_conditional_edges(START, fan_out, ["map", "reduce"])
    builder.add_edge("map", "reduce")
    builder.add_edge("reduce", END)
    return builder.compile(name=name, checkpointer=checkpointer)
