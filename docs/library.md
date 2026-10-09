English | [Deutsch](de/library.md)

# `agentpatterns`: reusable multi-agent patterns for LangGraph

`src/agentpatterns/` packages every pattern from this study as a factory
function, so developers can use a pattern in their own workflows without
re-implementing it. The library is a thin layer over LangChain v1
(`create_agent`, middleware, structured output) and LangGraph 1.x
(`StateGraph`, `Send`, `Command`). There is no custom runtime and nothing
hides the underlying graphs.

- [Design](#design): the agent contract, `AgentSpec`, principles
- [Quickstart](#quickstart)
- [API reference](#api-reference): one section per pattern
- [Limits and budgets](#limits-and-budgets): agent loops, participants, whole runs, graph steps
- [Messaging between agents](#messaging-between-agents): `MessagingMiddleware`, `Mailbox`, delivery, where it pays off
- [Integration guide](#integration-guide): subgraphs, tools, persistence, run context, retries and timeouts, MCP tools, HITL, streaming, async, real models
- [Composition](#composition): nesting patterns (the composite example)
- [Recipe: research with review](#recipe-research-with-review): orchestrator + researchers inside a review loop, and how state flows
- [Testing](#testing): `ScriptedChatModel`, `UsageTracker`, real-model smoke tests
- [Caveats](#caveats)

| Module | Pattern | Factory |
|---|---|---|
| `single_agent.py` | [Single agent](patterns/01-single-agent.md) | [`create_single_agent`](#create_single_agent) |
| `sequential.py` | [Sequential pipeline](patterns/02-sequential-pipeline.md) | [`create_pipeline`](#create_pipeline), `llm_step`, `agent_step`, `function_step` |
| `router.py` | [Router](patterns/03-router.md) | [`create_router`](#create_router) |
| `parallel.py` | [Parallelization](patterns/04-parallelization.md) | [`create_parallel`](#create_parallel), [`create_voting`](#create_voting), [`create_map_reduce`](#create_map_reduce) |
| `orchestrator.py` | [Orchestrator-workers](patterns/05-orchestrator-workers.md) | [`create_orchestrator`](#create_orchestrator) |
| `supervisor.py` | [Supervisor](patterns/06-supervisor.md) | [`create_supervisor`](#create_supervisor), `create_task_tool` |
| `hierarchical.py` | [Hierarchical teams](patterns/07-hierarchical.md) | [`create_hierarchy`](#create_hierarchy), `Team`, `build_team` |
| `swarm.py` | [Swarm / handoffs](patterns/08-swarm-handoffs.md) | [`create_swarm`](#create_swarm), `SwarmAgent`, `create_handoff_tool` |
| `state_machine.py` | [State machine](patterns/09-state-machine.md) | [`create_state_machine_agent`](#create_state_machine_agent), `Step`, `StateMachineMiddleware`, `transition` |
| `skills.py` | [Skills](patterns/10-skills.md) | [`create_skills_agent`](#create_skills_agent), `Skill`, `SkillsMiddleware` |
| `evaluator_optimizer.py` | [Evaluator-optimizer](patterns/11-evaluator-optimizer.md) | [`create_evaluator_optimizer`](#create_evaluator_optimizer), `Evaluation` |
| `limits.py` | guardrails | [`loop_limits`, `RunBudget`](#limits-and-budgets), `BudgetExceededError` (+ `DelegationLimitMiddleware` in `supervisor.py`) |
| `messaging.py` | agents message each other | [`MessagingMiddleware`, `Mailbox`](#messaging-between-agents), `AgentMessage` |
| `core.py` | building blocks | `AgentSpec`, `agent_as_tool`, `agent_as_node`, `invoke_agent`, `make_serializer`, `structured_llm`, `results_block` |
| `mcp.py` | MCP tools per run | [`McpTools`, `unreachable`](#mcp-tools-from-the-runs-server) (`mcp` extra) |
| `testing.py` | test utilities | `ScriptedChatModel`, `ModelTurn`, `UsageTracker`, `serve_mcp` |

---

## Design

### The agent contract

Every factory returns a **compiled LangGraph graph** that behaves like a
`langchain.agents.create_agent` graph:

```python
input  = {"messages": [...]}                        # plus optional pattern-specific keys
output = {"messages": [..., AIMessage(final)],      # final answer appended
          "structured_response": ...}               # if a response_format / schema was given
```

That is the whole trick. Because all patterns speak the same contract:

- **any pattern can be a participant of any other pattern**: a router can route
  to a supervisor, a supervisor can delegate to a pipeline, and the generator of
  an evaluator loop can be a router;
- **any pattern can be embedded in your own `StateGraph`**, either directly as a
  subgraph node (shared `messages` key) or via `agent_as_node` (different schema);
- **any pattern can be exposed as a tool** (`agent_as_tool`);
- **`create_agent` graphs and your own graphs are first-class participants.**

Inside a pattern, participants are *called* from a graph node (`agent_as_node`)
rather than added as subgraph nodes that share state with the pattern. A shared
`messages` channel would append every participant's transcript to the pattern's
conversation. Parallel copies of one agent (voting, map-reduce, several tasks
for one worker) would also collide on `structured_response`, and a participant
called again would see its previous run. Called participants start fresh on
every call and return only their result. LangGraph still recognises them as
subgraphs, so they appear in `get_graph(xray=True)` and `get_state(subgraphs=True)`
(see [graph view](#6-streaming-and-observability)).

### `AgentSpec`: participants of a pattern

```python
from agentpatterns import AgentSpec

spec = AgentSpec(
    name="billing_specialist",                       # node / tool name: alphanumeric, _ and -
    description="Invoices, duplicate charges, refunds.",   # routing signal for routers & supervisors
    agent=billing_agent,                             # anything following the agent contract
)
```

Names and descriptions are **prompting levers**: routers, supervisors,
orchestrators and swarms use them to decide whom to involve.

### Principles

1. **Native primitives only.** `create_agent`, `AgentMiddleware`,
   `response_format`, `with_structured_output`, `Send`, `Command`,
   `StateGraph`. Everything you know about LangGraph (checkpointers, streaming,
   interrupts, LangSmith) keeps working.
2. **Structured output first.** Routing decisions, plans, verdicts and final
   answers are schemas (dynamic `Literal` enums constrain agent names).
3. **Guardrails built in.** Every agent loop is capped (`max_model_calls`,
   `max_tool_calls`, `on_limit`), participants can be capped
   (`max_calls_per_agent`, `max_activations`, `max_visits`), pattern loops have
   `max_rounds`, `max_handoffs` and `max_iterations`, and `RunBudget` caps a
   whole run. Gates have fallbacks. See [Limits and budgets](#limits-and-budgets).
4. **Sync and async.** Every node that calls a model or agent has a native
   async implementation; delegation tools have `coroutine`s.
5. **Escape hatches.** Deterministic routing (`route_fn`), code aggregators,
   custom transition tools, middleware pass-through, and `**agent_kwargs`
   forwarded to `create_agent`.

## Quickstart

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from agentpatterns import AgentSpec, create_supervisor

model = init_chat_model("anthropic:claude-opus-5")

researcher = create_agent(model, tools=[web_search], system_prompt="You research facts.")
writer = create_agent(model, tools=[], system_prompt="You write concise summaries.")

team = create_supervisor(
    model,
    [AgentSpec("researcher", "Finds facts on the web.", researcher),
     AgentSpec("writer", "Turns notes into a summary.", writer)],
    system_prompt="You coordinate a small research team.",
    response_format=Report,           # optional Pydantic schema
)
result = team.invoke({"messages": [("user", "Summarize the state of EU AI regulation")]})
result["structured_response"]
```

---

## API reference

All factories accept `name=` (graph name, used in traces and as the subgraph
name). Parameters are keyword-only except the first ones shown.

### `create_single_agent`

```python
create_single_agent(model, tools, *, system_prompt=None, response_format=None,
                    max_model_calls=25, max_tool_calls=None, on_limit="error", middleware=(),
                    name="single_agent", **agent_kwargs) -> CompiledStateGraph
```

`create_agent` with guardrails: `ModelCallLimitMiddleware` (`on_limit`: raise
or end the loop) and optionally `ToolCallLimitMiddleware` (further tool calls are
refused), see [agent loops](#agent-loops). `agent_kwargs` go to `create_agent`
(`checkpointer`, `store`, `context_schema`, ...).

### `create_pipeline`

```python
create_pipeline(steps, *, output=None, name="pipeline") -> CompiledStateGraph
llm_step(name, model, *, system_prompt, output_schema=None, prompt=default_prompt, gate=None, on_gate_fail=None,
         structured_output_method="auto")
agent_step(name, agent, *, prompt=default_prompt, gate=None, on_gate_fail=None)
function_step(name, fn, *, gate=None, on_gate_fail=None)
```

- Each step writes `outputs[step_name]`; later steps and gates read `state["outputs"]`.
- `llm_step` makes one model call (structured if `output_schema`); `agent_step`
  runs any agent-contract runnable (its structured response or final text is
  the output); `function_step` runs deterministic code `fn(state) -> output`.
- `default_prompt(state)` = the original human messages plus
  `<step name="...">` blocks with all previous outputs. Pass
  `prompt=lambda state: ...` for custom prompts.
- Steps are `Step` objects (`PipelineStep` in the package root). A step computes
  its output with `run(state, config)` (+ optional native `arun`), or runs
  `agent` on the task `prompt(state)`. `agent_step` builds the latter, so
  the agent shows up as a subgraph of the pipeline.
- **Gate:** `gate(state) -> bool` is evaluated after the step. If it returns
  False, the pipeline stops; `on_gate_fail(state)` may supply the final result.
- `output`: which value becomes `structured_response` (a step name, a function,
  or `None` for the last executed step).
- **Input:** `messages`, optional `context` (a dict for deterministic steps,
  for example the original domain object). **Output:** `messages`,
  `structured_response`, `outputs`, `stopped_at`.

### `create_router`

```python
create_router(model, routes, *, system_prompt=DEFAULT, allow_multiple=True, route_fn=None,
              synthesizer_prompt=DEFAULT, response_format=None, synthesize=None,
              structured_output_method="auto", name="router") -> CompiledStateGraph
```

- Routing: one structured-output call returning `RoutingDecision`, where
  `RoutingDecision.routes: list[{agent: Literal[names], task: str}]`. The agent
  catalog is appended to `system_prompt`. `allow_multiple=False` means at most one route.
- `route_fn(state) -> [{"agent", "task"}]` replaces the LLM with deterministic routing.
- Selected routes run in parallel (`Send`); each agent receives its `task` as a
  human message.
- Synthesis: an LLM merges the results (structured with `response_format`).
  By default it is skipped when exactly one route answered and no
  `response_format` is set; `synthesize=True/False` forces it either way.
- **Output:** `messages`, `structured_response` (if `response_format`), `routes`.
- Reserved route names: `route`, `synthesize`.

### `create_parallel`

```python
create_parallel(branches, *, aggregator=None, model=None, synthesizer_prompt=DEFAULT,
                response_format=None, structured_output_method="auto", name="parallel") -> CompiledStateGraph
```

All branches get the full input messages concurrently. Aggregation, in order of
precedence: `aggregator(results_by_name: dict[str, AgentResult], state) -> value`
(code), an LLM synthesis with `model` (+ `response_format`), or
concatenation. **Output:** `messages`, `structured_response`, `branch_results`.

Tip: an agent without tools plus `response_format` is a single structured LLM
call, which makes a perfect analyst branch:
`create_agent(model, tools=[], system_prompt=..., response_format=Schema)`.

### `create_voting`

```python
create_voting(agent: AgentSpec, *, n=3, key=lambda r: r.text, name="voting")
```

Runs the agent `n` times in parallel and returns the majority answer
(`key(AgentResult)` selects what is compared; ties go to the earliest sample).
**Output:** the winner's `messages` / `structured_response`, plus `votes`. Use a
sampling temperature above 0.

### `create_map_reduce`

```python
create_map_reduce(mapper, *, reduce=lambda results: results, prepare=lambda item: item,
                  extract=lambda output: output, name="map_reduce")
```

Input `{"items": [...]}`, output `{"results": [...], "output": reduce(results)}`.
`mapper` is any runnable (a pattern graph, your own workflow, a chain). Results
keep the order of the items; empty input is fine.

### `create_orchestrator`

```python
create_orchestrator(model, workers, *, planner_prompt=DEFAULT, synthesizer_prompt=DEFAULT,
                    response_format=None, max_rounds=1, max_tasks_per_round=10,
                    structured_output_method="auto", name="orchestrator") -> CompiledStateGraph
```

- Planner: one structured-output call returning `Plan`, where
  `Plan.tasks: list[{worker: Literal[names], instruction: str}]`. An empty list
  means done.
- Tasks run in parallel (`Send`); the same worker can get several tasks.
- `max_rounds > 1` turns on re-planning: after each round the planner sees
  all results ("Results so far") and plans again.
- **Input:** `messages`, optionally `results` from earlier work. The planner then
  sees them, plans only what is still missing (an empty plan goes straight to
  synthesis), and the synthesizer writes from old and new results. `max_rounds`
  counts the rounds of one run; round numbers continue after the earlier ones.
  This is what a review loop with `carry_over=["results"]` uses.
- **Output:** `messages`, `structured_response`, `results`
  (`[{round, worker, task, output}]`, including earlier work). Reserved names:
  `prepare`, `plan`, `synthesize`, `collect`.

### `create_supervisor`

```python
create_supervisor(model, subagents, *, system_prompt, tools=(), response_format=None,
                  delegation="tool_per_agent", input_mode="task", max_model_calls=25,
                  max_tool_calls=None, on_limit="error", max_calls_per_agent=None,
                  middleware=(), name="supervisor", **agent_kwargs) -> CompiledStateGraph
DelegationLimitMiddleware(limits, *, delegation="tool_per_agent")   # use with any create_agent supervisor
```

- `delegation="tool_per_agent"`: one tool per subagent (`name(task: str)`).
  `"task_tool"`: a single `task(agent_name: Literal[...], description: str)`
  tool, convenient for many or team-owned agents.
- `input_mode="task"`: isolated subagent context. `"fork"`: the subagent also
  receives the supervisor's conversation (human and AI text, no tool traffic).
- Tools return the subagent's structured response as JSON, or its final text.
- `tools`: extra tools the supervisor uses itself. `middleware` and
  `agent_kwargs` (checkpointer, store, ...) go to `create_agent`.
- `max_model_calls` / `max_tool_calls` / `on_limit` cap the supervisor's own
  loop; subagents have their own limits. `max_calls_per_agent` (an int for all,
  or `{"billing": 2}`) caps delegations per subagent and run, see
  [participants](#participants).

### `create_hierarchy`

```python
Team(name, description, system_prompt, members=[AgentSpec | Team, ...], response_format=None, model=None)
create_hierarchy(model, teams, *, system_prompt, response_format=None, name="hierarchy",
                 **supervisor_kwargs) -> CompiledStateGraph
build_team(model, team, **supervisor_kwargs) -> AgentSpec
```

Every `Team` becomes a supervisor exposed as a tool to its parent (at any
depth). A `Team.response_format` makes team leads return structured reports.
`delegation`, `input_mode` and the limits (`max_model_calls`, `max_tool_calls`,
`on_limit`, `max_calls_per_agent`) apply to all levels; a `max_calls_per_agent`
mapping may name members of any level. Other kwargs (`middleware`,
`checkpointer`, ...) go to the top level only.

### `create_swarm`

```python
SwarmAgent(name, description, system_prompt, tools=(), handoffs=None, model=None,
           max_activations=None, max_model_calls=None, max_tool_calls=None)
create_swarm(model, agents, *, default_agent=None, response_format=None, history="full",
             max_handoffs=8, max_model_calls=25, max_tool_calls=None, on_limit="error",
             middleware=(), retry_policy=None, timeout=None, checkpointer=None,
             name="swarm") -> CompiledStateGraph
create_handoff_tool(agent_name, *, description=None, history="full", max_handoffs=None,
                    max_activations=None, name=None)
```

- Each agent gets `transfer_to_<peer>(note)` tools for its `handoffs` (default:
  all peers) and a roster in its prompt.
- `history="full"` forwards the whole conversation; `"handoff_only"` forwards
  only the handoff call and its `ToolMessage` (containing the note).
- `max_handoffs` per run: when reached, the handoff tool refuses and tells the
  agent to finish. `SwarmAgent.max_activations`: how often that agent may take
  control per run (the start counts).
- `max_model_calls` / `max_tool_calls` / `on_limit` cap each agent's loop *per
  activation*; `SwarmAgent.max_model_calls` / `max_tool_calls` override them for
  one agent. See [Limits and budgets](#limits-and-budgets).
- **State / output:** `messages`, `structured_response`, `active_agent`,
  `handoff_count`, `activations`. With a `checkpointer`, the next turn starts
  with `active_agent`.

### `create_state_machine_agent`

```python
Step(name, system_prompt, tools=(), transitions=(), final=False, tool_filter=None, max_visits=None)
create_state_machine_agent(model, steps, *, initial_step=None, response_format=None,
                           state_schema=None, tools=(), max_model_calls=25, max_tool_calls=None,
                           on_limit="error", middleware=(), name="state_machine",
                           **agent_kwargs) -> CompiledStateGraph
StateMachineMiddleware(steps, *, initial_step=None)      # use with any create_agent
transition(target, tool_call_id, message=None, **updates) -> Command   # for custom tools
create_transition_tool(target, description=None)          # go_to_<target>(reason)
```

- `system_prompt` is a template: `{key}` is filled from the state (dicts and
  models are rendered as JSON). It can also be a callable `state -> str`.
- `transitions` auto-generate `go_to_<step>(reason)` tools. Custom tools move
  steps with `return transition("next", runtime.tool_call_id, my_key=...)`.
- `tool_filter(state, tools) -> tools` narrows a step's tools at runtime.
- Only `final=True` steps keep `response_format`, so the agent cannot finish early.
- Extend `StateMachineState` for extra keys your prompts or tools use.
- `Step.max_visits`: how often a run may enter the step (the start counts);
  further transitions are refused. `step_visits` in the output shows the counts.
  `max_model_calls` caps the loop across all steps.

### `create_skills_agent`

```python
Skill(name, description, instructions, tools=())
create_skills_agent(model, skills, *, system_prompt, tools=(), response_format=None,
                    max_model_calls=25, max_tool_calls=None, on_limit="error",
                    middleware=(), name="skills_agent", **agent_kwargs) -> CompiledStateGraph
SkillsMiddleware(skills, *, always_available=(), show_catalog=True)   # use with any create_agent
```

- Adds `load_skill(skill_name: Literal[...])`, which returns the instructions
  as a tool result and records the skill in `loaded_skills` (the merge reducer
  makes parallel loads safe).
- Tools of unloaded skills are hidden. `tools` (or `always_available` names)
  stay visible.
- The catalog (name: description) is appended to the system prompt.

### `create_evaluator_optimizer`

```python
create_evaluator_optimizer(generator, evaluator, *, evaluator_prompt=None,
                           evaluation_schema=Evaluation, evaluator_context=None,
                           evaluator_input=None, carry_over=(),
                           passed=lambda e: e.passed, feedback=None, max_iterations=3,
                           on_max_iterations=None, keep_history=True,
                           structured_output_method="auto", retry_policy=None, timeout=None,
                           checkpointer=None, name="evaluator_optimizer") -> CompiledStateGraph
```

- `generator`: any agent-contract runnable, including other patterns.
- `evaluator`: either a **chat model** (one structured-output call per review,
  `evaluator_prompt` is its system prompt and required) or an **agent** that
  returns the verdict as `structured_response`, e.g.
  `create_agent(model, tools=[fetch_url], system_prompt=RUBRIC, response_format=Evaluation)`.
  An agent reviewer can check facts with tools; `evaluator_prompt` is optional
  for it and is added to the review request.
- Default schema `Evaluation(passed, score, issues, feedback)`; use `passed=`
  for custom schemas.
- `evaluator_context(state) -> str | messages | None`: extra material for the
  reviewer besides the request and the candidate, typically the evidence
  (`results_block(state.get("results", []), "Research results")`).
- `evaluator_input(state) -> str | messages`: what the reviewer is sent instead,
  the whole review request, e.g. a task from your own template that shows the
  candidate as facts without its reasoning:
  `lambda s: [*s["request"], HumanMessage(REVIEW.format(work=s["candidate"]))]`.
  `state["request"]` holds the loop's input messages. A string is one human
  message; a model evaluator still gets `evaluator_prompt` as its system prompt.
  Excludes `evaluator_context`.
- `carry_over=["results", ...]`: generator output keys kept between
  iterations. They are passed back into the generator on each revision (if its
  input schema accepts them), are visible to `evaluator_context`, may be given
  as input to the loop and are part of its output. Pattern graphs keep their
  work outside `messages`, so this is how a revision builds on it instead of
  starting over.
- `keep_history=True`: the generator continues its own conversation. For a
  `create_agent` generator that includes its tool calls, so revisions don't redo
  them; for pattern graphs use `carry_over`. `False`: it restarts from the
  request, the last candidate and the feedback.
- `on_max_iterations(candidate, evaluation) -> replacement` sets a safe
  fallback, such as escalating to a human.
- `retry_policy=RetryPolicy(...)` retries the generate and the evaluate step on
  their own, so a failed review does not redo the generation; `timeout=` limits
  each attempt in async runs (see [retries and timeouts](#retries-timeouts-checkpointer-run-context)).
- **Output:** `messages`, `structured_response`, `evaluation`, `iterations`,
  plus the `carry_over` keys.

### Core helpers

| Helper | Purpose |
|---|---|
| `agent_as_tool(spec, *, name=None, description=None, input_mode="task")` | Expose an agent or pattern as a tool (`task: str`); sync + async; `"fork"` forwards the caller's conversation. |
| `agent_as_node(agent, *, input=None, output=None, name=None)` | Adapt an agent-contract graph to a parent graph with a different schema. The node calls the graph directly, so it stays a visible subgraph (graph view, `get_state(subgraphs=True)`). |
| `invoke_agent(spec, task, config=None) -> AgentResult` | Invoke with a string or messages; returns `name`, `text`, `structured`, `messages`. |
| `final_text(result)` | Structured response as JSON, or the last message's text. |
| `make_serializer(*types)` | Checkpoint serializer allow-listing your Pydantic types (LangGraph ≥ 1.2). |
| `structured_llm(model, schema, method="auto")` | The structured-output call the patterns use internally (see [real models](#8-real-models-and-structured-output)). |
| `results_block(results, header)` | Render `AgentResult`s or orchestrator `results` as a prompt block, e.g. for `evaluator_context`. |
| `PatternState`, `PatternInput`, `PatternOutput` | State schemas of the contract, for your own compatible graphs. |

---

## Limits and budgets

Every loop in a multi-agent system needs a cap, and the loops nest: a
supervisor's loop calls subagents that each run their own loop. The library
caps four levels (`agentpatterns.limits`):

| Level | Caps | Settings | When reached |
|---|---|---|---|
| [Agent loop](#agent-loops) | model and tool calls of one agent run | `max_model_calls`, `max_tool_calls`, `on_limit` on every agent factory; `loop_limits(...)` for your own `create_agent` | model cap: raise (`"error"`) or stop with a final message (`"end"`); tool cap: further calls are refused, the model finishes |
| [Participants](#participants) | how often one agent or step runs | `max_calls_per_agent`, `SwarmAgent.max_activations`, `Step.max_visits`; pattern loops `max_rounds`, `max_tasks_per_round`, `max_iterations`, `max_handoffs` | refused with a message to the model, which continues without it |
| [Whole run](#whole-run-runbudget) | all model and tool calls of a run, nested agents included | `RunBudget(...)` as a callback | `BudgetExceededError`; the call over budget is not made |
| [Graph steps](#graph-steps-recursion_limit) | super-steps of each graph | `recursion_limit` in the run config | `GraphRecursionError` |

### Agent loops

```python
from langchain.agents import create_agent
from agentpatterns import SwarmAgent, create_single_agent, create_swarm, loop_limits

agent = create_single_agent(model, tools, max_model_calls=10, max_tool_calls=20, on_limit="end")

# Your own create_agent graphs get the same native middleware:
researcher = create_agent(model, [web_search], middleware=loop_limits(max_model_calls=8, on_limit="end"))

swarm = create_swarm(
    model,
    [SwarmAgent("triage", "Front desk", TRIAGE, max_model_calls=3),             # override for one agent
     SwarmAgent("research", "Deep research", RESEARCH, tools=[web_search])],     # swarm default
    max_model_calls=25,
    on_limit="end",
)
```

Available on `create_single_agent`, `create_supervisor`, `create_hierarchy`
(every level), `create_swarm` (plus per `SwarmAgent`),
`create_state_machine_agent` and `create_skills_agent`. Default:
`max_model_calls=25`, no tool cap.

- **Counted per run of that agent graph** (the native `run_limit`). A subagent
  called three times gets three fresh budgets, and a swarm agent gets a fresh one
  each time it takes control. Totals therefore multiply: a swarm can make about
  `(max_handoffs + 1) × max_model_calls` model calls, and a hierarchy multiplies
  per level. Use a [`RunBudget`](#whole-run-runbudget) for one total.
- `on_limit="error"` (default) raises `ModelCallLimitExceededError` (re-exported
  by `agentpatterns`). `"end"` stops the loop with a final `AIMessage` that says
  why, and without `structured_response`. As a subagent, the caller receives that
  text as the tool result and can react.
- `max_tool_calls` is soft: calls beyond it get an error `ToolMessage` ("Do not
  make additional tool calls"), so the model can still give a proper answer.
- For limits across conversation turns (with a checkpointer) or on a single
  tool, add the native middleware yourself:
  `middleware=[ModelCallLimitMiddleware(thread_limit=100), ToolCallLimitMiddleware(tool_name="web_search", run_limit=5)]`.

### Participants

```python
supervisor = create_supervisor(model, specialists, system_prompt=SUPERVISOR,
                               max_calls_per_agent={"research": 2, "billing": 1})   # or one int for all
swarm = create_swarm(model, [SwarmAgent("triage", "Front desk", TRIAGE, max_activations=1), ...])
agent = create_state_machine_agent(model, [Step("collect", COLLECT, transitions=["resolve"], max_visits=2), ...])
```

- `max_calls_per_agent` (supervisor, hierarchy) counts delegations per subagent
  and run, including parallel calls in one turn, for both delegation styles. A
  call over the limit is not executed; the supervisor gets an error
  `ToolMessage` and continues with the others. The logic lives in
  `DelegationLimitMiddleware`, which also works on a hand-built `create_agent`
  supervisor.
- `SwarmAgent.max_activations`: how often the agent may take control per run,
  starting included. Further handoffs to it are refused. `activations` in the
  output shows the counts.
- `Step.max_visits`: how often the run may enter the step, starting included.
  A refused `go_to_<step>` is not executed. A custom tool's transition
  (`transition(...)`) is undone, but its other updates stay because the tool
  already ran; its `ToolMessage` gets the refusal appended. `step_visits` in the
  output shows the counts.
- These counts reset with every run (`invoke`), so earlier turns of a
  conversation don't block later ones.

### Whole run: `RunBudget`

```python
from agentpatterns import BudgetExceededError, RunBudget

budget = RunBudget(max_model_calls=60, max_tool_calls=120)
try:
    result = hierarchy.invoke(inputs, {"callbacks": [budget]})
except BudgetExceededError as error:          # error.kind ("model" | "tool"), error.limit
    escalate_to_human(error)
budget.model_calls, budget.tool_calls        # what the run used
```

- A LangChain callback handler. LangChain passes callbacks down to every nested
  runnable, so the budget sees subagents behind tools, swarm peers, parallel
  branches (sync and async) and the pattern-internal calls (routing, planning,
  synthesis, judging).
- Hard: the call that would exceed the budget is not made, and the whole run
  stops with `BudgetExceededError`. With a checkpointer, the thread keeps its
  last completed step and can be resumed.
- Use one instance per run, or one per session to budget several runs together
  (`reset()` starts over). `graph.with_config(callbacks=[budget])` binds it to a
  graph.
- LangChain logs "Error in RunBudget.on_chat_model_start callback" before it
  re-raises; that line is expected.

### Graph steps: `recursion_limit`

LangGraph stops a graph after `recursion_limit` super-steps
(`GraphRecursionError`): `graph.invoke(inputs, {"recursion_limit": 200})`. The
default in LangGraph 1.2 is 10,007 (`LANGGRAPH_DEFAULT_RECURSION_LIMIT`). It is a
backstop, not a budget:

- It counts graph steps, not model calls. In a `create_agent` loop every
  middleware hook is its own node, so one model turn takes several steps.
- Nested graphs inherit the value from the run config but count their own
  steps, so it does not bound a run's total either.

Set it above what the real limits allow, so that it never fires first.

---

## Messaging between agents

The agents of a run can send each other messages, much like Claude Code
sessions do. An agent opts in with `MessagingMiddleware`
(`agentpatterns.messaging`), and a `Mailbox` carries the messages of one run:

```python
from langchain.agents import create_agent
from agentpatterns import AgentSpec, Mailbox, MessagingMiddleware, create_parallel

researcher = create_agent(model, [web_search], name="researcher", system_prompt=RESEARCH,
                          middleware=[MessagingMiddleware(description="Finds facts on the web.")])
analyst = create_agent(model, [], name="analyst", system_prompt=ANALYSIS,
                       middleware=[MessagingMiddleware(can_message=["researcher"])])
team = create_parallel([AgentSpec("researcher", "...", researcher), AgentSpec("analyst", "...", analyst)],
                       model=model)

mailbox = Mailbox(max_messages_per_agent=10)
result = await team.ainvoke(inputs, {"configurable": {"mailbox": mailbox}})
mailbox.log           # [AgentMessage(id=1, sender="researcher", to="analyst", text="...")]
mailbox.undelivered   # messages that no agent received
```

**What the agent gets**

- `list_agents`: the agents of the run it may message, with address, status
  (`running`, `finished`, or `not started` for names in `can_message`) and
  description.
- `send_message(to, message)`: the result says what happened to the message
  (see Delivery below). A refused message (not in `can_message`, over the limit,
  an unknown or ambiguous address, its own address) comes back as an error tool
  result, and the agent goes on without it.
- Messages from other agents, delivered before its next model call as a user
  message `<agent-message from="researcher" to="analyst">...</agent-message>`.
  They stay in the agent's conversation and its checkpoints.
- A short addition to the system prompt: its own address, that other agents
  see neither its answer nor its tool results, and that peer messages come from
  colleagues, not from the user.
- If it is about to finish with unread messages, the middleware sends it back
  to the model with them. `max_model_calls` still applies.

**How it works**

- **One `Mailbox` per run**, passed as `config["configurable"]["mailbox"]`. The
  config reaches every nested agent, including subagents behind delegation
  tools. A mailbox bound to the graph would mix up the messages of concurrent
  runs. In a run without a mailbox the middleware does nothing and hides its
  tools, so the same agent works with and without messaging.
- **Outside the graph state.** Parallel branches only see each other's state
  updates after the step, which is too late for agents that run at the same
  time. The mailbox is an in-memory object, safe to use from parallel branches
  (threads or async).
- **Addresses.** An agent's address is its name: `MessagingMiddleware("x")`,
  or by default `create_agent(name=...)`. A single middleware without a name can
  therefore serve every agent of a swarm (`create_swarm(..., middleware=[MessagingMiddleware()])`).
  Copies of one agent that run in the same run (map-reduce items, an
  orchestrator worker with several tasks, a subagent called in parallel) get
  `researcher`, `researcher#2`, ... in the order they start. Each call of an
  agent is a new copy: a supervisor that calls a subagent again starts another
  run of it, and so does a swarm agent that takes control again.
- **Delivery.**
  - A message to a running agent is queued for its next model call.
  - A message to an agent that is not running goes to the running agent with
    the same name. If there is none, it waits for the next agent with that name
    that starts in the run: a later pipeline step, the next swarm activation,
    the next call of a subagent. If several are running, the send is refused and
    the sender picks one.
  - `send_message` tells the sender which of these happened, e.g. "Sent to
    analyst. It is running and gets the message before its next step." or "early
    has finished. The message waits and is delivered if an agent named early
    starts again in this run."
- **Retries and resumes.** A retried step (`retry_policy`) keeps its agent's
  address, and the retry gets all the agent's messages again. With a
  checkpointer, the agent's state records how many messages it has read, so a
  resumed agent does not get them twice; pass the same `Mailbox` when you resume.
  On a checkpointed thread, each turn can use a new mailbox; the agent then
  starts reading at that mailbox's first message.
- **From code.** `mailbox.send("researcher", "Also check 2025.", sender="ops")`
  sends a message from outside the agents, for example from a person watching
  the run or from a test.

**Where it pays off.** Messages only help agents that run at the same time.
Everywhere else the patterns already pass results along:

| Pattern | Agents run at the same time? | Messaging |
|---|---|---|
| `create_parallel`, `create_map_reduce`, orchestrator workers in one round, a supervisor's parallel delegations | yes | Useful: share findings, avoid duplicate work, ask a peer |
| Pipeline, evaluator-optimizer | no | Notes for later steps only; step outputs already do that |
| Swarm | one at a time | The handoff note already does that |
| Supervisor and the subagent it calls | the supervisor waits | The message arrives along with the tool result |
| Router, single agent, state machine, skills | one agent | Nobody to talk to |
| Voting | yes | Leave it off: the samples should stay independent |

**Safety and limits**

- Peer messages arrive as user messages, but they are marked and explained as
  coming from colleagues, not from the user. A prompt injection in data that an
  agent reads (an e-mail, a web page) can still make that agent send messages.
  Restrict who may message whom with `can_message`, and keep side effects behind
  checks in the tools or human approval on the receiving agent, as with
  delegation.
- `input_mode="fork"` forwards the caller's conversation without peer messages.
- `Mailbox(max_messages_per_agent=10)` (the default) caps the messages one
  agent run may send. Each delivered message costs the receiver a model call;
  `max_model_calls` and `RunBudget` cap the total.

**Not covered (yet)**

- An agent cannot block to wait for a reply. If the reply comes after the agent
  has finished, it waits for that agent's next run.
- The mailbox lives in memory. Runs that are resumed in another process lose
  their messages.
- An agent run that fails without a retry keeps the status `running`.

---

## Integration guide

### 1. Embed a pattern in your workflow (subgraph)

**Same key (`messages`):** add the compiled pattern directly as a node.

```python
from langgraph.graph import StateGraph, MessagesState, START

team = create_parallel([...])
graph = StateGraph(MessagesState).add_node("team", team).add_edge(START, "team").compile()
```

**Different schema:** map inputs and outputs with `agent_as_node`. This is how
all e-mail workflows embed their pattern (`email_assistant/common.py`):

```python
from agentpatterns import agent_as_node

node = agent_as_node(
    router,
    input=lambda state: {"messages": [HumanMessage(state["email"].as_prompt())]},
    output=lambda result, state: {"resolution": result["structured_response"]},
)
builder.add_node("resolve", node)
```

### 2. Expose a pattern as a tool

```python
from agentpatterns import AgentSpec, agent_as_tool

help_desk = agent_as_tool(AgentSpec("help_desk", "Answers product questions.", router))
chat = create_agent(model, tools=[help_desk], checkpointer=InMemorySaver())
```

This is also how you make a stateless router or pipeline usable in a
multi-turn chat: the chat agent keeps the memory.

### 3. Nest patterns

Every factory accepts `AgentSpec`s whose `agent` is another pattern:

```python
desk = create_router(model, specialists)
checked = create_evaluator_optimizer(desk, model, evaluator_prompt=RULES)
top = create_supervisor(model, [AgentSpec("desk", "Routed help desk", checked), ...], system_prompt=...)
```

### 4. Persistence, memory, multi-turn

- Compile or create the **outermost** graph with a checkpointer and pass a
  `thread_id`:
  `graph.invoke(input, {"configurable": {"thread_id": "customer-42"}})`.
- Inner agents and subagents use the default per-invocation persistence
  (they inherit the parent's checkpointer within one call). That is right for
  stateless specialists and supports interrupts.
- Swarm and state machine keep `active_agent` / `current_step` in state, so
  multi-turn conversations continue where they stopped.
- LangGraph ≥ 1.2 checkpoints only allow-listed classes without warnings: use
  `InMemorySaver(serde=make_serializer(MySchema, ...))` (or the same `serde`
  on Postgres/SQLite savers).

### Retries, timeouts, checkpointer, run context

Every factory takes the same things, so a composition behaves like one graph:

- **`checkpointer=`** on every factory. Only the outermost graph needs one; the
  graphs inside it checkpoint on it.
- **`retry_policy=`** (a LangGraph `RetryPolicy`, or several) and **`timeout=`**
  on every factory that builds its own graph (`create_evaluator_optimizer`,
  `create_orchestrator`, `create_router`, `create_parallel`, `create_voting`,
  `create_map_reduce`, `create_pipeline`, `create_swarm`). Both apply to each step
  that runs an agent or a model, not to the pattern's bookkeeping steps:
  - A step that raises is retried on its own, so one transient error does not
    rerun the whole pattern.
  - `timeout` (seconds, a `timedelta`, or a `TimeoutPolicy` that can also cap
    idle time) cancels an attempt that takes longer, with `NodeTimeoutError`. The
    default `RetryPolicy` retries it, so `retry_policy=RetryPolicy(max_attempts=3),
    timeout=30` gives a hung model call two more tries of 30 s each. It limits one
    attempt of one step, not the run.
  - Timeouts are LangGraph node timeouts and need an async run (`ainvoke`,
    `astream`). LangGraph cannot cancel sync code, so `invoke` refuses a step
    that has one (`ValueError`).
  - Retries cost calls: a [`RunBudget`](#whole-run-runbudget) counts every
    attempt. `BudgetExceededError` is a `RuntimeError`, which the default
    `RetryPolicy` does not retry; keep it that way when you pass `retry_on=`.
- **The `create_agent` factories** (single agent, supervisor, hierarchy, state
  machine, skills) retry with LangChain's middleware instead:
  `middleware=[ModelRetryMiddleware(on_failure="error"), ToolRetryMiddleware(...)]`.
  Pass `on_failure="error"`: the default, `"continue"`, ends the agent loop with
  the error text as its answer, which a caller (a supervisor, a review loop) then
  takes for a real one. They have no steps of their own to time out; use the chat
  model's request timeout (e.g. `ChatAnthropic(timeout=...)`), or the `timeout=`
  of the pattern or graph node that runs them.
- **The run context** (`graph.invoke(input, context=...)`) reaches every agent
  inside a pattern, subagents behind a delegation tool included, as
  `runtime.context` / `ToolRuntime.context`. The patterns declare no context type
  of their own; build your agents with `context_schema=` if you want it typed.
  So an agent is built once, and what differs between runs (a tenant, a server
  URL, an eval case's environment) comes with each run.

### MCP tools from the run's server

`agentpatterns.mcp` (needs the `mcp` extra, `multiagent-patterns[mcp]`) gives an
agent the tools of an MCP server the run names, so the agent is built once even
when every run talks to another server, e.g. one per eval case:

```python
from agentpatterns.mcp import McpTools, unreachable
from langchain.agents.middleware import ToolRetryMiddleware

agent = create_agent(
    model,
    tools=[],
    middleware=[
        ToolRetryMiddleware(retry_on=unreachable, on_failure=lambda e: "The records server is down."),
        McpTools(lambda ctx: ctx.mcp_url, tools=["search", "get_record"], server="records"),
    ],
    context_schema=Context,
)
await agent.ainvoke({"messages": [...]}, context=Context(mcp_url=...))
```

It is built on LangChain's `langchain.mcp` (beta, on FastMCP), which replaces the
archived `langchain-mcp-adapters`. Importing it warns once that the API may change.

- **The target** is anything `MCPAdapter` accepts: an http(s) URL, or a
  `fastmcp.Client` when the run needs more than an address, such as auth
  (`Client(url, auth=token)`), a request timeout, or a tool-list cache shared
  between runs (below). `target(ctx)` is called at the start and for every model
  and tool call, so it should not do I/O.
- When the agent starts, `McpTools` lists the server's tools once (only those
  named in `tools=`; a missing one fails the start) and keeps their definitions
  in the agent's state; every model call is offered them, and every call of one
  connects to the run's server.
- An error the server reports reaches the model as the tool's answer.
  `unreachable(error)` tells a server that could not be reached apart (a
  transport failure, or 502/503/504 from a gateway in front of it): retry tool
  calls with `ToolRetryMiddleware(retry_on=unreachable, ...)` (its `on_failure`
  message tells the model), and a failed start with the enclosing step's
  `retry_policy=RetryPolicy(retry_on=unreachable)`. Put `ToolRetryMiddleware`
  **before** `McpTools`: `McpTools` connects around each call, so only an outer
  retry covers a failed connect.
- **A server that asks for input** during a call (MCP elicitation) interrupts
  the run. The interrupt's value is `{"type": "mcp_elicitation", "tool_name",
  "requests": [{"key", "message", ...}]}`; resume with
  `Command(resume={"responses": {key: {"action": "accept", "content": {...}}}})`.
  Like every interrupt, it needs a checkpointer.
- **Caching the tool list across runs:** pass a `fastmcp.Client` with a shared
  cache store, partitioned per tenant so no tenant sees another's list:

  ```python
  from fastmcp import Client
  from fastmcp.client.caching import KeyValueResponseCacheStore
  from mcp.client.caching import CacheConfig

  store = KeyValueResponseCacheStore()     # in memory; pass storage=RedisStore(...) to share it between processes
  McpTools(lambda ctx: Client(ctx.mcp_url, cache=CacheConfig(store=store, partition=ctx.tenant, target_id="records")))
  ```

  The server decides how long a list may be cached (`ttlMs`; a FastMCP server
  sets it with `FastMCP(..., cache_ttl=60)`). Without a TTL from the server,
  nothing is cached and every run lists again.
- Several servers on one agent: one `McpTools` each, with its own `server=`.
- Async only, like the MCP client.

### 5. Human in the loop

- **Before risky tools:** pass
  `middleware=[HumanInTheLoopMiddleware(interrupt_on={"issue_refund": True})]`
  to `create_single_agent`, `create_supervisor`, `create_swarm`,
  `create_state_machine_agent` or `create_skills_agent`.
- **Before side effects in a workflow:** add a node that calls `interrupt(...)`
  and resume with `graph.invoke(Command(resume="approve"), config)`. See
  `human_approval` in `email_assistant/library_based/composite.py`.
- Interrupts raised inside subagents propagate to the top-level graph when only
  that graph has a checkpointer.

### 6. Streaming and observability

- `graph.stream(input, stream_mode="updates", subgraphs=True)` streams nested
  pattern steps. Subagents called inside tools are not statically visible
  subgraphs, but their model calls still appear in callbacks and traces.
- **Graph view:** `graph.get_graph(xray=True)` (what LangGraph Studio and
  similar tools draw) expands every participant of the router, parallel,
  voting, map-reduce, orchestrator, pipeline (`agent_step`), evaluator-optimizer
  and swarm into its own nodes, level by level (`xray=N` limits the depth).
  `get_state(config, subgraphs=True)` shows the state of an interrupted
  participant. Limits:
  - Supervisor and hierarchy subagents are tools, so they only appear as the
    `tools` node.
  - An embedded swarm stays one box: LangGraph only inlines subgraphs with a
    single exit, and any swarm agent may end the run.
  - `draw_mermaid()` needs subgraph node names that are unique across all levels
    (a `langchain_core` restriction; the JSON graph has no such limit).
- LangSmith tracing works out of the box; agent names (`name=`) appear as
  `lc_agent_name` metadata.
- `agentpatterns.testing.UsageTracker` is a callback that counts model calls,
  tokens (from `usage_metadata`) and tool calls, per agent.

### 7. Async

All factories work with `ainvoke` / `astream`. Model and agent calls inside
nodes and delegation tools run natively async, and parallel branches run
concurrently.

### 8. Real models and structured output

- Pass any LangChain chat model (`init_chat_model("anthropic:claude-opus-5")`).
- **Agent-based patterns** (single agent, supervisor, hierarchy, swarm, state
  machine, skills, and every `create_agent` participant): `response_format=Schema`
  lets LangChain pick `ProviderStrategy` (native structured output) when the
  model's profile declares support, and `ToolStrategy` otherwise. Pass
  `ProviderStrategy(Schema)` or `ToolStrategy(Schema)` to decide yourself.
- **Graph patterns** make their own structured calls (routing decision, plan,
  synthesis, `llm_step`, model evaluator). They follow the same rule through
  `structured_output_method="auto"`: `with_structured_output(schema,
  method="json_schema")` when the model's profile declares native structured
  output, the provider default otherwise. Pass `"function_calling"` (or any
  other `method=` value of your provider) to override.
- Why it matters for Claude: `with_structured_output`'s default for Claude is
  forced tool calling. Claude Opus 5.5, Claude Sonnet 5.5 and Claude Fable 5.1
  reject forced `tool_choice` (`any`/`tool`), and so does any Claude model with extended
  thinking enabled. `langchain-anthropic` then falls back to an *unforced* tool
  call and raises `OutputParserException` if the model answers in text instead.
  Native structured output (`output_config.format`) has no such gap, which is why
  `"auto"` prefers it. (Checked against `langchain-anthropic` 1.7.4: every current
  Claude model's profile declares native support.)
- Run `tests/test_real_model.py` against your model before relying on it (see
  [Testing](#testing)).
- Use different models per role by building participants with their own models
  (`AgentSpec(..., create_agent(small_model, ...))`, `Team(model=...)`,
  `SwarmAgent(model=...)`, and the `evaluator` of a review loop).

---

## Composition

`src/email_assistant/library_based/composite.py` shows the kind of system the
library is meant for, built entirely from factories:

```
e-mail workflow (StateGraph)
└── pipeline                                   create_pipeline
    ├── triage       = four parallel analysts  create_parallel      (gate: spam -> ignore)
    └── quality_loop = reflection loop         create_evaluator_optimizer
                       └── generator = router  create_router
                                       └── routes = specialist agents (create_agent)
├── human_approval (interrupt() for refunds and escalations)
└── dispatch (deterministic send)

inbox digest = create_map_reduce(workflow over all e-mails)
```

```python
triage = create_parallel(analysts, aggregator=aggregate)
router = create_router(model, specialist_specs(model), system_prompt=ROUTER,
                       synthesizer_prompt=REPLY_WRITER, response_format=EmailResolution)
checked = create_evaluator_optimizer(router, model, evaluator_prompt=REVIEWER,
                                     evaluation_schema=QualityReview, max_iterations=2, on_max_iterations=escalate)
pipeline = create_pipeline([
    agent_step("triage", triage, gate=lambda s: s["outputs"]["triage"]["category"] != "spam",
               on_gate_fail=ignore_spam),
    agent_step("quality_loop", checked),
])
```

Each of the eleven library-based workflows in `email_assistant/library_based/`
is a 20-75 line example of one factory, and each produces exactly the same
results as its hand-written counterpart in `email_assistant/native/` (asserted
by the tests).

## Recipe: research with review

A common combination: an orchestrator that fans out research tasks to
researcher subagents, and an independent reviewer that checks the result and
sends it back with feedback. Model it as **the orchestrator inside a review
loop**. The loop is the outer pattern because reviewing the finished deliverable
is its own concern, ideally done by a judge with a fresh context and its own
criteria.

```python
from langchain.agents import create_agent
from agentpatterns import AgentSpec, Evaluation, create_evaluator_optimizer, create_orchestrator, results_block

researcher = create_agent(model, tools=[web_search], name="researcher",
                          system_prompt="Research the question you are given. Cite source URLs.")
research = create_orchestrator(
    model,
    [AgentSpec("researcher", "Researches one question on the web", researcher)],
    planner_prompt=PLANNER,
    response_format=Report,                     # e.g. Report(summary, sources)
)
reviewer = create_agent(model, tools=[fetch_url], name="reviewer",   # tools: optional fact checking
                        system_prompt=REVIEW_RUBRIC, response_format=Evaluation)
reviewed = create_evaluator_optimizer(
    research,
    reviewer,
    carry_over=["results"],                     # keep the research between revisions
    evaluator_context=lambda s: results_block(s.get("results", []), "Research results"),
    max_iterations=3,
)
out = reviewed.invoke({"messages": [("user", "Brief me on the EU e-bike market")]})
out["structured_response"], out["results"], out["evaluation"], out["iterations"]
```

The two options that make this work:

- `carry_over=["results"]`: without it, every revision restarts the
  orchestrator from scratch and all research runs again. With it, the planner
  sees the earlier results next to the feedback and plans only the gaps (or
  nothing, if the feedback is about writing).
- `evaluator_context`: without it, the reviewer only sees the request and the
  report, so it can judge structure and completeness but not whether the
  claims are backed by the research.

### How the state flows

Three graphs are involved: the review loop (outer), the orchestrator (the
loop's generator) and the researcher agents (the orchestrator's workers). Each
has its own state; they exchange only what their input and output schemas
declare.

```
review loop state                  orchestrator state (one per iteration)       researcher (one per task)
─────────────────                  ──────────────────────────────────────       ─────────────────────────
messages     (input)          ─┐
request      (= input messages)│
                               └─► messages = request (+ old report + feedback)
results      (carried) ───────────► results  = earlier results
                                   prepare:  round = last round, last_round = round + max_rounds
                                   plan:     plan = [{worker, instruction}, ...]
                                             ── Send(instruction) ─────────────► messages = [instruction]
                                                                                 tool loop (search, ...)
                                   results += [{round, worker, task, output}] ◄── final text
                                   synthesize: structured_response = Report
candidate          ◄────────────── structured_response
results            ◄────────────── results (earlier + new)
generator_messages ◄────────────── messages (request, report)
evaluate: reviewer gets request + evaluator_context(state) + candidate
evaluation = Evaluation(passed, feedback, ...)
passed? → finish: messages += [report], structured_response = report
failed? → next iteration: generator_messages + "Reviewer feedback: ..." and results go back in
```

Traced with the scripted model (`test_review_loop_revises_research_without_redoing_it`):

| Step | Graph | What changes |
|---|---|---|
| 1 | orchestrator | `plan`: 2 tasks (market, rivals); two researchers run in parallel; `results` = 2 entries (round 1) |
| 2 | orchestrator | `synthesize`: Report with 2 sources |
| 3 | loop | `candidate` = Report, `results` = 2 entries, `iterations` = 1 |
| 4 | loop | reviewer sees the 2 results and the report: `passed=False`, "Add pricing research." |
| 5 | orchestrator | input: request, report v1, feedback, and the 2 results. `plan`: 1 task (pricing, round 2) |
| 6 | orchestrator | `synthesize` from 3 results: Report with 3 sources |
| 7 | loop | reviewer: `passed=True`; `finish` returns the report, 3 `results`, `iterations=2` |

Three searches in total. Without `carry_over`, the second iteration starts
from scratch: the planner doesn't see the earlier results and runs the first
two searches again.

What stays private: the orchestrator's `plan`, `round` and `last_round`, the
loop's `request`, `candidate` and `generator_messages`, and each researcher's
own conversation (only its final text reaches `results`).

### What the surrounding workflow needs to define

Only what it wants to read or pass in. There are two ways to embed the loop.

**Shared keys:** add the compiled loop directly as a node. The parent state
needs `messages` plus the loop's output keys you care about:

```python
class ResearchWorkflow(MessagesState):
    results: list[dict]               # NO reducer: the loop returns the complete list
    structured_response: Report
    evaluation: Evaluation
    iterations: int

graph = (StateGraph(ResearchWorkflow)
         .add_node("research", reviewed)
         .add_edge(START, "research")
         .compile(checkpointer=InMemorySaver(serde=make_serializer(Report, Evaluation))))
```

Because `results` is also an input of the loop, a follow-up question on the
same thread starts from the stored research
(`test_review_loop_as_subgraph_builds_on_earlier_research`). Don't give
`results` an `operator.add` reducer in the parent: the loop already returns
earlier plus new results, so appending would duplicate them.

**Your own schema:** map in and out with `agent_as_node`:

```python
class Briefing(TypedDict, total=False):
    topic: str
    report: Report
    sources: list[dict]
    review: Evaluation

builder.add_node("research", agent_as_node(
    reviewed,
    input=lambda s: {"messages": [HumanMessage(s["topic"])], "results": s.get("sources", [])},
    output=lambda r, s: {"report": r["structured_response"], "sources": r["results"], "review": r["evaluation"]},
))
```

Either way, register your schemas with `make_serializer` if the workflow has a
checkpointer, and set `on_max_iterations` if a report that never passes
review should be escalated rather than returned.

## Testing

`agentpatterns.testing` lets you unit-test orchestration without API calls:

```python
from agentpatterns.testing import ScriptedChatModel, ModelTurn, UsageTracker

def policy(turn: ModelTurn):
    if "role=router" in turn.system:
        return turn.structured({"routes": [{"agent": "billing", "task": "check invoice"}]})
    if not turn.called("get_invoice"):
        return turn.call("get_invoice", invoice_id="INV-1")        # or turn.call_many(...)
    return turn.say(f"Invoice: {turn.result('get_invoice')}")

model = ScriptedChatModel(policy=policy)
```

- The policy sees the messages and bound tools (`turn.system`, `turn.tool_names`,
  `turn.tool_results()`, `turn.call_results()`, `turn.structured_tool_schema()`, ...)
  and returns text, tool calls (parallel via `call_many`) or structured output.
- The model raises `ScriptError` on things a real model cannot do: calling an
  unbound tool, or answering in text when a tool call is forced.
- Policies are pure functions of the conversation, so they stay deterministic
  under parallel execution, unlike list-based fake models.
- `UsageTracker` (a callback) asserts call counts and cost regressions.
- `turn.agent_messages` lists the messages from other agents in the conversation
  (`AgentMessage`: `sender`, `to`, `text`), for agents with `MessagingMiddleware`
  (`tests/test_messaging.py`).
- `serve_mcp(server)` serves an MCP server (`fastmcp.FastMCP`, or the MCP SDK's
  `MCPServer`) over HTTP for the length of a `with` block and yields its URL, so
  agents with `McpTools` run against a real MCP server in tests (`tests/test_mcp.py`).
- `ScriptedChatModel(policy=..., profile={"structured_output": True})` simulates
  a model with native structured output: `create_agent` then uses
  `ProviderStrategy` and the library `method="json_schema"`, as with current
  Claude models. `turn.structured(...)` answers correctly on both paths, so the
  same policy covers both (the e-mail tests run every workflow both ways).

`tests/test_library.py` has a compact test for every factory, which doubles as
usage examples.

**Real-model smoke tests.** `tests/test_real_model.py` runs the patterns
against a real model and checks that each returns its declared structure (not
answer quality). They are skipped unless you opt in:

```bash
uv sync --extra anthropic
AGENTPATTERNS_TEST_MODEL=anthropic:claude-opus-5 uv run pytest -m real_model
```

Run them for every model you plan to use, especially models without forced
tool use (Claude Opus 5.5, Claude Sonnet 5.5, Claude Fable 5.1). One run makes
roughly 40-60 model calls. The Anthropic provider reads its key from
`ANTHROPIC_API_KEY`.

## Caveats

- **Handoffs and parallel tool calls:** a handoff tool called in the same turn
  as other tools produces an unpaired history (a LangGraph limitation shared by
  all Command-based handoffs). Prompt agents to hand off alone.
- **Subagent state inspection:** subagents invoked inside tools (supervisor,
  hierarchy) are not visible to `get_state(subgraphs=True)` or the graph view.
  Use graph-node patterns (router, orchestrator, parallel) when you need to
  inspect or draw nested state.
- **Router, parallel and orchestrator state** stores `AgentResult` objects.
  With a checkpointer, register them via `make_serializer` (already included).
- **Carried-over keys** are last-value in the review loop: the generator is
  expected to return the complete value (as the orchestrator does with
  `results`). A generator that doesn't accept the key as input still has it
  exposed in the output, but can't build on it.
- **Cost:** multi-agent patterns multiply tokens; see the measured comparison
  in [use-case.md](use-case.md#measured-comparison) and the per-pattern reports.
