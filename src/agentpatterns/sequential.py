"""Sequential pipeline (prompt chaining) with gates.

A fixed sequence of steps. Each step reads the original input plus the
outputs of all previous steps and writes its own output into
`outputs[step_name]`. Steps can be

* `llm_step`      - one LLM call, optionally with structured output,
* `agent_step`    - any agent-contract runnable (e.g. a `create_agent` with tools),
* `function_step` - plain, deterministic Python (retrieval, executing actions, ...).

Any step can have a *gate*: a predicate evaluated after the step. If it fails,
the pipeline stops early, optionally producing a fallback result.

    pipeline = create_pipeline([
        llm_step("classify", model, system_prompt="...", output_schema=Category,
                 gate=lambda s: s["outputs"]["classify"].label != "spam",
                 on_gate_fail=lambda s: Resolution(action="ignore")),
        function_step("enrich", lookup_data),
        llm_step("answer", model, system_prompt="...", output_schema=Resolution),
    ])
    pipeline.invoke({"messages": [("user", "...")]})["structured_response"]
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, NotRequired

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from agentpatterns.core import (
    PatternInput,
    PatternState,
    RetryPolicies,
    StepTimeout,
    StructuredOutputMethod,
    agent_as_node,
    dual_node,
    final_text,
    merge_dicts,
    step_policies,
    structured_llm,
    to_text,
)

__all__ = ["PipelineState", "Step", "agent_step", "create_pipeline", "function_step", "llm_step"]


class PipelineInput(PatternInput):
    context: NotRequired[dict[str, Any]]
    """Optional structured input for deterministic steps (not shown to models by default)."""


class PipelineState(PatternState):
    context: NotRequired[dict[str, Any]]
    outputs: Annotated[dict[str, Any], merge_dicts]
    stopped_at: NotRequired[str]


class PipelineOutput(PatternInput):
    structured_response: NotRequired[Any]
    outputs: dict[str, Any]
    stopped_at: NotRequired[str]


StateFn = Callable[[dict[str, Any]], Any]


def default_prompt(state: dict[str, Any]) -> str:
    """Original input + outputs of all previous steps, each in a tagged block."""
    original = "\n\n".join(m.text for m in state["messages"] if isinstance(m, HumanMessage))
    blocks = [f'<step name="{name}">\n{to_text(value)}\n</step>' for name, value in state.get("outputs", {}).items()]
    if not blocks:
        return original
    return original + "\n\nResults of previous steps:\n" + "\n".join(blocks)


@dataclass
class Step:
    """One pipeline step. Prefer the `llm_step` / `agent_step` / `function_step` helpers.

    A step computes its output with `run` (+ optional native `arun`) or runs `agent`.
    """

    name: str
    run: Callable[[dict[str, Any], RunnableConfig], Any] | None = None
    arun: Callable[[dict[str, Any], RunnableConfig], Any] | None = None
    gate: StateFn | None = None
    on_gate_fail: StateFn | None = None
    agent: Runnable | None = None
    """Agent-contract runnable run on `prompt(state)` (shown as a subgraph of the pipeline)."""
    prompt: StateFn = default_prompt

    def __post_init__(self) -> None:
        if (self.run is None) == (self.agent is None):
            raise ValueError(f"Step {self.name!r} needs either `run` or `agent`.")


def _agent_output(result: dict[str, Any]) -> Any:
    """An agent step's output: its structured response, or else its final text."""
    structured = result.get("structured_response")
    return structured if structured is not None else final_text(result)


def llm_step(
    name: str,
    model: BaseChatModel,
    *,
    system_prompt: str,
    output_schema: type | None = None,
    prompt: StateFn = default_prompt,
    gate: StateFn | None = None,
    on_gate_fail: StateFn | None = None,
    structured_output_method: StructuredOutputMethod = "auto",
) -> Step:
    """A single LLM call (structured output if `output_schema` is given, see `StructuredOutputMethod`)."""
    llm = structured_llm(model, output_schema, structured_output_method)

    def messages(state: dict[str, Any]) -> list:
        return [SystemMessage(system_prompt), HumanMessage(prompt(state))]

    def unwrap(value: Any) -> Any:
        return value.text if isinstance(value, AIMessage) else value

    def run(state: dict[str, Any], config: RunnableConfig) -> Any:
        return unwrap(llm.invoke(messages(state), config))

    async def arun(state: dict[str, Any], config: RunnableConfig) -> Any:
        return unwrap(await llm.ainvoke(messages(state), config))

    return Step(name, run, arun, gate, on_gate_fail)


def agent_step(
    name: str,
    agent: Runnable,
    *,
    prompt: StateFn = default_prompt,
    gate: StateFn | None = None,
    on_gate_fail: StateFn | None = None,
) -> Step:
    """Run an agent-contract runnable; its structured response (or final text) is the step output."""
    return Step(name, gate=gate, on_gate_fail=on_gate_fail, agent=agent, prompt=prompt)


def function_step(
    name: str,
    fn: StateFn,
    *,
    gate: StateFn | None = None,
    on_gate_fail: StateFn | None = None,
) -> Step:
    """Deterministic Python step: `fn(state) -> output`."""
    return Step(name, lambda state, config: fn(state), None, gate, on_gate_fail)


def create_pipeline(
    steps: Sequence[Step],
    *,
    output: str | StateFn | None = None,
    retry_policy: RetryPolicies = None,
    timeout: StepTimeout = None,
    checkpointer: Checkpointer = None,
    name: str = "pipeline",
) -> CompiledStateGraph:
    """Build a sequential pipeline following the agent contract.

    Args:
        steps: Steps in execution order (names must be unique).
        output: Which result becomes `structured_response`: a step name, a
            function of the state, or `None` for the last executed step.
        retry_policy: Retries of the steps that run an agent or a model when they raise
            (see `RetryPolicies`).
        timeout: Time limit of one attempt of such a step; async runs only (see
            `StepTimeout`).
        checkpointer: Checkpoints the graph; only the outermost graph needs one.
        name: Graph name.

    Input keys: `messages`, optional `context` (dict for deterministic steps).
    Output keys: `messages` (+ final AIMessage), `structured_response`,
    `outputs` (all step outputs), `stopped_at` (step whose gate failed).
    """
    names = [s.name for s in steps]
    if len(set(names)) != len(names):
        raise ValueError(f"Step names must be unique: {names}")
    if "finish_pipeline" in names or any(n.startswith("gate_failed_") for n in names):
        raise ValueError("Step names 'finish_pipeline' and 'gate_failed_*' are reserved.")

    def make_node(step: Step):
        if step.agent is not None:
            return agent_as_node(
                step.agent,
                input=lambda state: {"messages": [HumanMessage(step.prompt(state))]},
                output=lambda result, state: {"outputs": {step.name: _agent_output(result)}},
                name=step.name,
            )

        def sync(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            return {"outputs": {step.name: step.run(state, config)}}

        async def asynchronous(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            value = await step.arun(state, config) if step.arun else step.run(state, config)
            return {"outputs": {step.name: value}}

        return dual_node(sync, asynchronous, step.name)

    def gate_router(index: int):
        step = steps[index]
        following = names[index + 1] if index + 1 < len(names) else "finish_pipeline"

        def route(state: dict[str, Any]) -> str:
            if step.gate is None or step.gate(state):
                return following
            return f"gate_failed_{step.name}"

        return route

    def make_stop(step: Step):
        def stop(state: dict[str, Any]) -> dict[str, Any]:
            update: dict[str, Any] = {"stopped_at": step.name}
            if step.on_gate_fail is not None:
                update["outputs"] = {"__fallback__": step.on_gate_fail(state)}
            return update

        return stop

    def finish(state: dict[str, Any]) -> dict[str, Any]:
        outputs = state.get("outputs", {})
        if "__fallback__" in outputs:
            result = outputs["__fallback__"]
        elif callable(output):
            result = output(state)
        elif isinstance(output, str):
            result = outputs.get(output)
        else:
            executed = [n for n in names if n in outputs]
            result = outputs[executed[-1]] if executed else None
        return {"messages": [AIMessage(content=to_text(result), name=name)], "structured_response": result}

    policies = step_policies(retry_policy, timeout)
    builder = StateGraph(PipelineState, input_schema=PipelineInput, output_schema=PipelineOutput)
    for index, step in enumerate(steps):
        builder.add_node(step.name, make_node(step), **policies)
        stop = f"gate_failed_{step.name}"
        following = names[index + 1] if index + 1 < len(names) else "finish_pipeline"
        if step.gate is not None:
            builder.add_node(stop, make_stop(step))
            builder.add_edge(stop, "finish_pipeline")
            builder.add_conditional_edges(step.name, gate_router(index), [following, stop])
        else:
            builder.add_edge(step.name, following)
    builder.add_node("finish_pipeline", finish)
    builder.add_edge(START, names[0])
    builder.add_edge("finish_pipeline", END)
    return builder.compile(name=name, checkpointer=checkpointer)
