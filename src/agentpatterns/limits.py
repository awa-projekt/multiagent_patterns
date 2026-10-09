"""Limits - cap agent loops, participants and whole runs.

Four levels, from the inside out:

1. **Agent loop** - `max_model_calls`, `max_tool_calls` and `on_limit` on every
   factory that builds a `create_agent` loop (single agent, supervisor, hierarchy,
   swarm, state machine, skills). `loop_limits(...)` returns the same native
   middleware for your own `create_agent` graphs. The count is per run *of that
   agent graph*: a subagent called three times gets three fresh budgets, a swarm
   agent a fresh one each time it takes control.
2. **Participants** - how often one agent or step may run per run:
   `max_calls_per_agent` (supervisor), `SwarmAgent.max_activations`,
   `Step.max_visits`, next to the pattern loops `max_rounds`, `max_iterations`
   and `max_handoffs`. These limits are soft: the model is told the limit is
   reached and continues without that participant.
3. **Whole run** - `RunBudget`, one hard budget for everything a run does,
   including nested agents, parallel branches and pattern-internal LLM calls.
4. **Graph steps** - LangGraph's `recursion_limit` (run config). It counts
   super-steps, not model calls, and every middleware hook is a node: a
   backstop, not a budget.

    agent = create_single_agent(model, tools, max_model_calls=10, on_limit="end")
    researcher = create_agent(model, tools, middleware=loop_limits(max_model_calls=8))

    budget = RunBudget(max_model_calls=60)
    graph.invoke(inputs, {"callbacks": [budget]})
"""

from __future__ import annotations

import threading
from typing import Any, Literal

from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.callbacks import BaseCallbackHandler

__all__ = ["BudgetExceededError", "ModelCallLimitExceededError", "OnLimit", "RunBudget", "loop_limits"]

OnLimit = Literal["error", "end"]
"""What happens when an agent loop reaches `max_model_calls`.

"error" (default) raises `ModelCallLimitExceededError`. "end" stops the loop with a
final `AIMessage` explaining why (no `structured_response`); as a subagent, the
caller receives that text as the tool result and can react.
"""


def loop_limits(
    *,
    max_model_calls: int | None = None,
    max_tool_calls: int | None = None,
    on_limit: OnLimit = "error",
) -> list[AgentMiddleware]:
    """Native call-limit middleware for one agent loop, counted per run of the agent graph.

    Args:
        max_model_calls: Hard cap on model calls; `on_limit` decides what happens.
        max_tool_calls: Soft cap on tool calls: further calls are refused with an
            error `ToolMessage`, so the model can still answer with what it has.
        on_limit: "error" or "end" (see `OnLimit`).

    For thread-level limits (across conversation turns) or limits on one tool,
    add `ModelCallLimitMiddleware(thread_limit=...)` /
    `ToolCallLimitMiddleware(tool_name=..., ...)` yourself.
    """
    guards: list[AgentMiddleware] = []
    if max_model_calls is not None:
        guards.append(ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior=on_limit))
    if max_tool_calls is not None:
        guards.append(ToolCallLimitMiddleware(run_limit=max_tool_calls))
    return guards


class BudgetExceededError(RuntimeError):
    """Raised when a `RunBudget` is used up. Nothing was called beyond the budget."""

    def __init__(self, kind: Literal["model", "tool"], limit: int) -> None:
        self.kind = kind
        self.limit = limit
        super().__init__(f"Run budget exceeded: the run may make at most {limit} {kind} calls.")


class RunBudget(BaseCallbackHandler):
    """Hard cap on the model and tool calls of a whole run, across all nested agents.

    Pass it as a callback. LangChain hands callbacks down to every nested
    runnable: subagents behind tools, swarm peers, parallel branches and the
    pattern-internal calls (routing, planning, judging).

        budget = RunBudget(max_model_calls=60, max_tool_calls=100)
        try:
            out = graph.invoke(inputs, {"callbacks": [budget]})
        except BudgetExceededError:
            ...  # escalate; with a checkpointer the thread can be resumed later
        budget.model_calls, budget.tool_calls

    The call that would exceed the budget is not made. Counts accumulate over
    every run the instance is passed to: create one per run, reuse one for a
    per-session budget, or `reset()` it.
    """

    raise_error = True  # LangChain only re-raises callback errors with this flag
    run_inline = True  # count in the caller's thread / event loop, no executor hop

    def __init__(self, *, max_model_calls: int | None = None, max_tool_calls: int | None = None) -> None:
        if max_model_calls is None and max_tool_calls is None:
            raise ValueError("Set max_model_calls and/or max_tool_calls.")
        self.limits = {"model": max_model_calls, "tool": max_tool_calls}
        self.used = {"model": 0, "tool": 0}
        self._lock = threading.Lock()  # parallel branches report from several threads

    @property
    def model_calls(self) -> int:
        return self.used["model"]

    @property
    def tool_calls(self) -> int:
        return self.used["tool"]

    def reset(self) -> None:
        with self._lock:
            self.used = {"model": 0, "tool": 0}

    def _charge(self, kind: Literal["model", "tool"]) -> None:
        with self._lock:
            limit = self.limits[kind]
            if limit is not None and self.used[kind] >= limit:
                raise BudgetExceededError(kind, limit)
            self.used[kind] += 1

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list, **kwargs: Any) -> None:
        self._charge("model")

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        self._charge("model")  # non-chat LLMs

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        self._charge("tool")
