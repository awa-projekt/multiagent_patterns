# 12. Other patterns and variants

These patterns show up in the literature and in other frameworks. We did not
implement them as separate workflows, but here is how they relate to ours and
how to build them with LangGraph.

## Plan-and-execute (and ReWOO, LLMCompiler)

A planner writes an explicit multi-step plan, executors run the steps, and a
re-planner revises or finishes. That is our
[orchestrator-workers](05-orchestrator-workers.md) with `max_rounds > 1`.

- **ReWOO:** the plan uses variables for earlier results and runs without re-planning.
- **LLMCompiler:** the plan is a DAG of tasks streamed to parallel executors
  (the paper claims a 3.6× speedup).
- In LangChain v1, planning inside a single agent is available as
  `TodoListMiddleware` (a `write_todos` tool). Deep Agents combine planning,
  subagents and a virtual filesystem.

Use it for long tasks where a strong planner coordinates cheaper executors.
Watch out for stale plans; re-plan after surprises.

## Blackboard / shared state

Agents coordinate by reading and writing a shared store instead of messaging
each other. A controller (or the agents themselves) decides who acts next
based on the board. In LangGraph the board is the graph state (keys with
reducers) or a `Store`. Google ADK calls session state "your whiteboard".
Salemi et al. (2025) report 13-57% relative improvement with a blackboard
design where agents volunteer for posted requests. No coordinator needs to know
every agent's expertise.

Risks: write conflicts and inconsistent state. Microsoft lists shared mutable
state between concurrent agents as an anti-pattern. Use reducers and
single-writer keys.

## Group chat / debate / council

Several agents discuss on a shared thread; a manager picks the next speaker
and a termination condition ends the chat. Debate variants have agents propose
and critique answers over several rounds. Du et al. (2023) report better
reasoning and factuality. Microsoft recommends three agents or fewer and
read-only participants.

LangGraph implementation: a `StateGraph` with a `speaker_selector` node (LLM or
round-robin) routing to agent nodes that share `messages`, plus a turn counter.
Our [swarm](08-swarm-handoffs.md) is the decentralized cousin, and
[voting](04-parallelization.md) is the parallel, non-interactive cousin.

Costs grow with agents × rounds; use it for ideation, reviews and adversarial
checks, not for routine processing.

## "Smart friend" / advisor

A worker agent consults a stronger model (or a forked copy with full context)
for hard decisions only (Cognition 2026). It is a supervisor inverted: the
worker keeps control. Build it with `agent_as_tool(..., input_mode="fork")`
around a stronger model.

## Deep Agents

LangChain's higher-level harness (`deepagents`) packages planning (todos),
subagents (sync and async, isolated or forked), skills and a virtual
filesystem for context offloading. It is the batteries-included option when
you want a general long-running agent rather than a specific business
workflow.

## Composite ("custom workflow")

Real systems combine patterns. LangChain calls this *custom workflow*:
deterministic graph structure with agentic nodes, and other patterns embedded
as nodes. Our [composite example](../library.md#composition) nests parallel
triage, a spam gate, a router with specialists, an evaluator loop, human
approval and map-reduce.
