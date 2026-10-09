"""Evaluator-optimizer (reflection loop).

A generator produces a candidate, an evaluator grades it with structured
output. Failed candidates go back to the generator together with the feedback
until the evaluation passes or `max_iterations` is reached.

    loop = create_evaluator_optimizer(
        drafting_agent,                    # generator: any agent-contract runnable
        model,                             # evaluator: a chat model (may be cheaper) or an agent
        evaluator_prompt="Check the draft against these rules: ...",
        max_iterations=3,
    )
    out = loop.invoke({"messages": [("user", "...")]})
    out["structured_response"], out["evaluation"], out["iterations"]

On a revision the generator continues its own conversation (`keep_history=True`).
For a `create_agent` generator that includes its tool calls, so they are not
repeated. Pattern graphs keep their work outside `messages` (e.g. the
orchestrator's `results`); list those keys in `carry_over` so they are passed
back in, shown to the evaluator via `evaluator_context` and returned:

    reviewed = create_evaluator_optimizer(
        research_orchestrator,
        reviewer_agent,                    # create_agent(..., response_format=Evaluation)
        carry_over=["results"],
        evaluator_context=lambda s: results_block(s.get("results", []), "Research results"),
    )

`evaluator_input` replaces what the evaluator is sent, e.g. a review task from your own
template instead of the request plus the candidate as JSON:

    reviewed = create_evaluator_optimizer(
        writer,
        reviewer_agent,
        evaluator_input=lambda s: [*s["request"], HumanMessage(REVIEW_TASK.format(draft=s["candidate"].text))],
    )
"""

from __future__ import annotations

import types
from collections.abc import Callable, Sequence
from typing import Any, NotRequired

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer
from pydantic import BaseModel, Field

from agentpatterns.core import (
    PatternInput,
    PatternState,
    RetryPolicies,
    StepTimeout,
    StructuredOutputMethod,
    dual_node,
    final_text,
    step_policies,
    structured_llm,
    to_text,
)

__all__ = ["Evaluation", "create_evaluator_optimizer"]


class Evaluation(BaseModel):
    """Verdict on a candidate answer."""

    passed: bool = Field(description="True only if the candidate fully meets all criteria.")
    score: float = Field(ge=0, le=1, description="Quality score between 0 and 1.")
    issues: list[str] = Field(default_factory=list, description="Concrete problems to fix.")
    feedback: str = Field(description="Actionable guidance for the next revision.")


class ReflectionState(PatternState):
    request: NotRequired[list]  # original input messages (what the evaluator judges against)
    candidate: NotRequired[Any]
    evaluation: NotRequired[Any]
    iterations: NotRequired[int]
    generator_messages: NotRequired[list]


class ReflectionOutput(PatternInput):
    structured_response: NotRequired[Any]
    evaluation: Any
    iterations: int


_RESERVED_KEYS = set(ReflectionState.__annotations__)
EvaluatorContext = Callable[[dict[str, Any]], str | Sequence[BaseMessage] | None]
EvaluatorInput = Callable[[dict[str, Any]], str | Sequence[BaseMessage]]


def _with_keys(name: str, base: type, keys: Sequence[str]) -> type:
    """`base` extended by optional, last-value keys (the carried-over state)."""
    if not keys:
        return base

    def body(ns: dict[str, Any]) -> None:
        ns["__module__"] = __name__
        ns["__annotations__"] = {key: NotRequired[Any] for key in keys}

    return types.new_class(name, (base,), {}, body)


def create_evaluator_optimizer(
    generator: Runnable,
    evaluator: BaseChatModel | Runnable,
    *,
    evaluator_prompt: str | None = None,
    evaluation_schema: type[BaseModel] = Evaluation,
    evaluator_context: EvaluatorContext | None = None,
    evaluator_input: EvaluatorInput | None = None,
    carry_over: Sequence[str] = (),
    passed: Callable[[Any], bool] = lambda evaluation: evaluation.passed,
    feedback: Callable[[Any], str] | None = None,
    max_iterations: int = 3,
    on_max_iterations: Callable[[Any, Any], Any] | None = None,
    keep_history: bool = True,
    structured_output_method: StructuredOutputMethod = "auto",
    retry_policy: RetryPolicies = None,
    timeout: StepTimeout = None,
    checkpointer: Checkpointer = None,
    name: str = "evaluator_optimizer",
) -> CompiledStateGraph:
    """Build a generate -> evaluate -> revise loop following the agent contract.

    Args:
        generator: Agent-contract runnable producing the candidate (structured
            response or final text).
        evaluator: The judge. Either a chat model (one structured-output call per
            review; `evaluator_prompt` is its system prompt) or an agent-contract
            runnable that returns the verdict as `structured_response`, e.g.
            `create_agent(model, tools=[...], system_prompt=RUBRIC, response_format=Evaluation)`
            for a reviewer that checks facts with tools.
        evaluator_prompt: Criteria / rubric. Required for a model evaluator; for an
            agent evaluator it is optional and added to the review request.
        evaluation_schema: Structured verdict of a model evaluator (default `Evaluation`).
        evaluator_context: `fn(state) -> str | messages | None`, extra material for
            the evaluator besides request and candidate (e.g. the research results
            of a carried-over key, so the reviewer can check claims against sources).
        evaluator_input: `fn(state) -> str | messages`, what the evaluator is sent
            instead of the request and the candidate (a string is one human
            message), e.g. a review task rendered from `state["candidate"]` with
            your own template. `state["request"]` holds the loop's input messages.
            Excludes `evaluator_context`; a model evaluator still gets
            `evaluator_prompt` as its system prompt.
        carry_over: State keys of the generator's output to keep between iterations
            (e.g. `["results"]` for an orchestrator). They are passed back into the
            generator on each revision (if its input schema accepts them), are
            visible to `evaluator_context`, may be given as input to the loop and
            are part of its output.
        passed: Extracts pass/fail from a verdict (for custom schemas).
        feedback: Formats the revision request (default: feedback + issues).
        max_iterations: Hard cap on generator runs.
        on_max_iterations: `fn(candidate, evaluation) -> replacement` applied when
            the cap is hit without passing (e.g. escalate to a human). Default:
            return the last candidate unchanged (check `evaluation` in the output).
        keep_history: Generator continues its own conversation (True) or restarts
            from the request + last candidate + feedback (False).
        structured_output_method: How a model evaluator produces the verdict
            (see `StructuredOutputMethod`; "auto" = native where supported).
        retry_policy: Retries of the generate and evaluate steps when they raise
            (see `RetryPolicies`), so a failed review does not redo the generation.
        timeout: Time limit of one attempt of the generate or evaluate step; async runs
            only (see `StepTimeout`).
        checkpointer: Checkpoints the loop; only the outermost graph needs one.
        name: Graph name.
    """
    if clash := _RESERVED_KEYS.intersection(carry_over):
        raise ValueError(f"carry_over keys {clash} are used by the loop itself.")
    if evaluator_input is not None and evaluator_context is not None:
        raise ValueError(
            "evaluator_input replaces the evaluator's whole input; pass evaluator_context or it, not both."
        )
    model_judge = isinstance(evaluator, BaseChatModel)
    if model_judge and not evaluator_prompt:
        raise ValueError("A model evaluator needs an evaluator_prompt (its rubric).")
    judge = structured_llm(evaluator, evaluation_schema, structured_output_method) if model_judge else evaluator
    carried = tuple(carry_over)

    def default_feedback(evaluation: Any) -> str:
        issues = "\n".join(f"- {i}" for i in getattr(evaluation, "issues", []) or [])
        text = getattr(evaluation, "feedback", "") or to_text(evaluation)
        return f"Reviewer feedback: {text}\n{issues}\nPlease revise your answer accordingly."

    make_feedback = feedback or default_feedback

    def generator_messages(state: dict[str, Any]) -> list:
        request = state.get("request") or state["messages"]
        if not state.get("evaluation"):
            return list(request)
        revision = HumanMessage(make_feedback(state["evaluation"]))
        if keep_history:
            return [*state["generator_messages"], revision]
        return [*request, AIMessage(to_text(state["candidate"])), revision]

    def generator_input(state: dict[str, Any]) -> dict[str, Any]:
        kept = {key: state[key] for key in carried if state.get(key) is not None}
        return {"messages": generator_messages(state), **kept}

    def generated(state: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        candidate = result.get("structured_response")
        return {
            "request": state.get("request") or list(state["messages"]),
            "candidate": candidate if candidate is not None else final_text(result),
            "generator_messages": result["messages"],
            "iterations": state.get("iterations", 0) + 1,
            **{key: result[key] for key in carried if key in result},
        }

    def generate_sync(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        return generated(state, generator.invoke(generator_input(state), config))

    async def generate_async(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        return generated(state, await generator.ainvoke(generator_input(state), config))

    def judge_messages(state: dict[str, Any]) -> list:
        if evaluator_input is not None:
            given = evaluator_input(state)
            sent = [HumanMessage(given)] if isinstance(given, str) else list(given)
            return [*([SystemMessage(evaluator_prompt)] if model_judge else []), *sent]
        context = evaluator_context(state) if evaluator_context is not None else None
        parts = [] if model_judge or not evaluator_prompt else [f"Review criteria:\n{evaluator_prompt}"]
        if isinstance(context, str):
            parts.append(context)
        parts.append(f"Candidate to evaluate:\n{to_text(state['candidate'])}")
        return [
            *([SystemMessage(evaluator_prompt)] if model_judge else []),
            *state["request"],
            *(context if context and not isinstance(context, str) else []),
            HumanMessage("\n\n".join(parts)),
        ]

    def verdict(result: Any) -> dict[str, Any]:
        if model_judge:
            return {"evaluation": result}
        if result.get("structured_response") is None:
            raise ValueError(
                "The evaluator agent returned no structured_response; "
                f"create it with response_format={evaluation_schema.__name__} (or your verdict schema)."
            )
        return {"evaluation": result["structured_response"]}

    def judge_input(state: dict[str, Any]) -> Any:
        return judge_messages(state) if model_judge else {"messages": judge_messages(state)}

    def evaluate_sync(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        return verdict(judge.invoke(judge_input(state), config))

    async def evaluate_async(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        return verdict(await judge.ainvoke(judge_input(state), config))

    def route(state: dict[str, Any]) -> str:
        if passed(state["evaluation"]) or state["iterations"] >= max_iterations:
            return "finish"
        return "generate"

    def finish(state: dict[str, Any]) -> dict[str, Any]:
        candidate = state["candidate"]
        if not passed(state["evaluation"]) and on_max_iterations is not None:
            candidate = on_max_iterations(candidate, state["evaluation"])
        update: dict[str, Any] = {"messages": [AIMessage(content=to_text(candidate), name=name)]}
        if not isinstance(candidate, str):
            update["structured_response"] = candidate
        return update

    policies = step_policies(retry_policy, timeout)
    builder = StateGraph(
        _with_keys("ReflectionState", ReflectionState, carried),
        input_schema=_with_keys("ReflectionInput", PatternInput, carried),
        output_schema=_with_keys("ReflectionOutput", ReflectionOutput, carried),
    )
    builder.add_node("generate", dual_node(generate_sync, generate_async, "generate"), **policies)
    builder.add_node("evaluate", dual_node(evaluate_sync, evaluate_async, "evaluate"), **policies)
    builder.add_node("finish", finish)
    builder.add_edge(START, "generate")
    builder.add_edge("generate", "evaluate")
    builder.add_conditional_edges("evaluate", route, ["generate", "finish"])
    builder.add_edge("finish", END)
    return builder.compile(name=name, checkpointer=checkpointer)
