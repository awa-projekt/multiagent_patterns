[English](../library.md) | Deutsch

# `agentpatterns`: wiederverwendbare Multi-Agent-Patterns für LangGraph

`src/agentpatterns/` stellt jedes Pattern dieser Studie als Factory-Funktion
bereit. So können Entwickler ein Pattern in ihren eigenen Workflows nutzen,
ohne es neu zu implementieren. Die Bibliothek ist eine dünne Schicht über
LangChain v1 (`create_agent`, Middleware, strukturierte Ausgabe) und
LangGraph 1.x (`StateGraph`, `Send`, `Command`). Es gibt keine eigene
Laufzeitumgebung, und nichts verbirgt die zugrunde liegenden Graphen.

- [Design](#design): der Agent-Vertrag, `AgentSpec`, Prinzipien
- [Schnellstart](#schnellstart)
- [API-Referenz](#api-referenz): ein Abschnitt pro Pattern
- [Limits und Budgets](#limits-und-budgets): Agent-Schleifen, Teilnehmer, ganze Läufe, Graph-Schritte
- [Nachrichten zwischen Agenten](#nachrichten-zwischen-agenten): `MessagingMiddleware`, `Mailbox`, Zustellung, wann es sich lohnt
- [Integrationsleitfaden](#integrationsleitfaden): Subgraphen, Tools, Persistenz, Laufkontext, Retries und Timeouts, MCP-Tools, HITL, Streaming, Async, echte Modelle
- [Komposition](#komposition): Patterns verschachteln (das Composite-Beispiel)
- [Rezept: Recherche mit Review](#rezept-recherche-mit-review): Orchestrator + Researcher in einer Review-Schleife und wie der State fließt
- [Testen](#testen): `ScriptedChatModel`, `UsageTracker`, Smoke-Tests mit echten Modellen
- [Einschränkungen](#einschränkungen)

| Modul | Pattern | Factory |
|---|---|---|
| `single_agent.py` | [Single Agent](patterns/01-single-agent.md) | [`create_single_agent`](#create_single_agent) |
| `sequential.py` | [Sequenzielle Pipeline](patterns/02-sequential-pipeline.md) | [`create_pipeline`](#create_pipeline), `llm_step`, `agent_step`, `function_step` |
| `router.py` | [Router](patterns/03-router.md) | [`create_router`](#create_router) |
| `parallel.py` | [Parallelisierung](patterns/04-parallelization.md) | [`create_parallel`](#create_parallel), [`create_voting`](#create_voting), [`create_map_reduce`](#create_map_reduce) |
| `orchestrator.py` | [Orchestrator-Worker](patterns/05-orchestrator-workers.md) | [`create_orchestrator`](#create_orchestrator) |
| `supervisor.py` | [Supervisor](patterns/06-supervisor.md) | [`create_supervisor`](#create_supervisor), `create_task_tool` |
| `hierarchical.py` | [Hierarchische Teams](patterns/07-hierarchical.md) | [`create_hierarchy`](#create_hierarchy), `Team`, `build_team` |
| `swarm.py` | [Swarm / Handoffs](patterns/08-swarm-handoffs.md) | [`create_swarm`](#create_swarm), `SwarmAgent`, `create_handoff_tool` |
| `state_machine.py` | [State Machine](patterns/09-state-machine.md) | [`create_state_machine_agent`](#create_state_machine_agent), `Step`, `StateMachineMiddleware`, `transition` |
| `skills.py` | [Skills](patterns/10-skills.md) | [`create_skills_agent`](#create_skills_agent), `Skill`, `SkillsMiddleware` |
| `evaluator_optimizer.py` | [Evaluator-Optimizer](patterns/11-evaluator-optimizer.md) | [`create_evaluator_optimizer`](#create_evaluator_optimizer), `Evaluation` |
| `limits.py` | Guardrails | [`loop_limits`, `RunBudget`](#limits-und-budgets), `BudgetExceededError` (+ `DelegationLimitMiddleware` in `supervisor.py`) |
| `messaging.py` | Agenten schreiben einander Nachrichten | [`MessagingMiddleware`, `Mailbox`](#nachrichten-zwischen-agenten), `AgentMessage` |
| `core.py` | Bausteine | `AgentSpec`, `agent_as_tool`, `agent_as_node`, `invoke_agent`, `make_serializer`, `structured_llm`, `results_block` |
| `mcp.py` | MCP-Tools pro Lauf | [`McpTools`, `unreachable`](#mcp-tools-vom-server-des-laufs) (Extra `mcp`) |
| `testing.py` | Test-Hilfsmittel | `ScriptedChatModel`, `ModelTurn`, `UsageTracker`, `serve_mcp` |

---

## Design

### Der Agent-Vertrag

Jede Factory gibt einen **kompilierten LangGraph-Graphen** zurück, der sich wie
ein `langchain.agents.create_agent`-Graph verhält:

```python
input  = {"messages": [...]}                        # plus optional pattern-specific keys
output = {"messages": [..., AIMessage(final)],      # final answer appended
          "structured_response": ...}               # if a response_format / schema was given
```

Das ist schon der ganze Trick. Weil alle Patterns denselben Vertrag einhalten,
gilt:

- **Jedes Pattern kann Teilnehmer jedes anderen Patterns sein**: Ein Router kann
  an einen Supervisor routen, ein Supervisor kann an eine Pipeline delegieren,
  und der Generator einer Evaluator-Schleife kann ein Router sein;
- **jedes Pattern lässt sich in Ihren eigenen `StateGraph` einbetten**, entweder
  direkt als Subgraph-Knoten (gemeinsamer Schlüssel `messages`) oder über
  `agent_as_node` (anderes Schema);
- **jedes Pattern kann als Tool bereitgestellt werden** (`agent_as_tool`);
- **`create_agent`-Graphen und Ihre eigenen Graphen sind vollwertige Teilnehmer.**

Innerhalb eines Patterns werden Teilnehmer aus einem Graph-Knoten heraus
*aufgerufen* (`agent_as_node`). Sie werden nicht als Subgraph-Knoten
hinzugefügt, die ihren State mit dem Pattern teilen. Ein gemeinsamer
`messages`-Kanal würde das Transkript jedes Teilnehmers an die Konversation des
Patterns anhängen. Parallele Kopien eines Agenten (Voting, Map-Reduce, mehrere
Aufgaben für einen Worker) würden außerdem bei `structured_response`
kollidieren, und ein erneut aufgerufener Teilnehmer sähe seinen vorherigen
Lauf. Aufgerufene Teilnehmer starten bei jedem Aufruf frisch und geben nur ihr
Ergebnis zurück. LangGraph erkennt sie trotzdem als Subgraphen, daher erscheinen
sie in `get_graph(xray=True)` und `get_state(subgraphs=True)`
(siehe [Graph-Ansicht](#6-streaming-und-observability)).

### `AgentSpec`: Teilnehmer eines Patterns

```python
from agentpatterns import AgentSpec

spec = AgentSpec(
    name="billing_specialist",                       # node / tool name: alphanumeric, _ and -
    description="Invoices, duplicate charges, refunds.",   # routing signal for routers & supervisors
    agent=billing_agent,                             # anything following the agent contract
)
```

Namen und Beschreibungen sind **Prompting-Hebel**: Router, Supervisor,
Orchestratoren und Swarms entscheiden damit, wen sie einbeziehen.

### Prinzipien

1. **Nur native Primitive.** `create_agent`, `AgentMiddleware`,
   `response_format`, `with_structured_output`, `Send`, `Command`,
   `StateGraph`. Alles, was Sie über LangGraph wissen (Checkpointer, Streaming,
   Interrupts, LangSmith), funktioniert weiterhin.
2. **Strukturierte Ausgabe zuerst.** Routing-Entscheidungen, Pläne, Urteile und
   finale Antworten sind Schemas (dynamische `Literal`-Enums schränken die
   Agent-Namen ein).
3. **Eingebaute Guardrails.** Jede Agent-Schleife ist begrenzt (`max_model_calls`,
   `max_tool_calls`, `on_limit`), Teilnehmer lassen sich begrenzen
   (`max_calls_per_agent`, `max_activations`, `max_visits`), Pattern-Schleifen
   haben `max_rounds`, `max_handoffs` und `max_iterations`, und `RunBudget`
   begrenzt einen ganzen Lauf. Gates haben Fallbacks. Siehe
   [Limits und Budgets](#limits-und-budgets).
4. **Sync und Async.** Jeder Knoten, der ein Modell oder einen Agenten aufruft,
   hat eine native Async-Implementierung; Delegations-Tools haben `coroutine`s.
5. **Hintertüren.** Deterministisches Routing (`route_fn`), Aggregatoren in
   Code, eigene Übergangs-Tools, durchgereichte Middleware und
   `**agent_kwargs`, die an `create_agent` weitergegeben werden.

## Schnellstart

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

## API-Referenz

Alle Factories akzeptieren `name=` (Graph-Name, verwendet in Traces und als
Subgraph-Name). Die Parameter sind Keyword-only, mit Ausnahme der jeweils
zuerst gezeigten.

### `create_single_agent`

```python
create_single_agent(model, tools, *, system_prompt=None, response_format=None,
                    max_model_calls=25, max_tool_calls=None, on_limit="error", middleware=(),
                    name="single_agent", **agent_kwargs) -> CompiledStateGraph
```

`create_agent` mit Guardrails: `ModelCallLimitMiddleware` (`on_limit`: Exception
auslösen oder die Schleife beenden) und optional `ToolCallLimitMiddleware`
(weitere Tool-Aufrufe werden abgelehnt), siehe [Agent-Schleifen](#agent-schleifen).
`agent_kwargs` gehen an `create_agent` (`checkpointer`, `store`,
`context_schema`, ...).

### `create_pipeline`

```python
create_pipeline(steps, *, output=None, name="pipeline") -> CompiledStateGraph
llm_step(name, model, *, system_prompt, output_schema=None, prompt=default_prompt, gate=None, on_gate_fail=None,
         structured_output_method="auto")
agent_step(name, agent, *, prompt=default_prompt, gate=None, on_gate_fail=None)
function_step(name, fn, *, gate=None, on_gate_fail=None)
```

- Jeder Schritt schreibt `outputs[step_name]`; spätere Schritte und Gates lesen
  `state["outputs"]`.
- `llm_step` macht einen Modellaufruf (strukturiert, wenn `output_schema`
  gesetzt ist); `agent_step` führt ein beliebiges Runnable nach dem
  Agent-Vertrag aus (seine strukturierte Antwort oder sein finaler Text ist die
  Ausgabe); `function_step` führt deterministischen Code `fn(state) -> output`
  aus.
- `default_prompt(state)` = die ursprünglichen Human-Nachrichten plus
  `<step name="...">`-Blöcke mit allen bisherigen Ausgaben. Übergeben Sie
  `prompt=lambda state: ...` für eigene Prompts.
- Schritte sind `Step`-Objekte (`PipelineStep` im Paket-Root). Ein Schritt
  berechnet seine Ausgabe mit `run(state, config)` (+ optional nativem `arun`)
  oder führt `agent` mit der Aufgabe `prompt(state)` aus. `agent_step` erzeugt
  die zweite Variante, sodass der Agent als Subgraph der Pipeline erscheint.
- **Gate:** `gate(state) -> bool` wird nach dem Schritt ausgewertet. Gibt es
  False zurück, stoppt die Pipeline; `on_gate_fail(state)` kann das Endergebnis
  liefern.
- `output`: welcher Wert zu `structured_response` wird (ein Schrittname, eine
  Funktion oder `None` für den zuletzt ausgeführten Schritt).
- **Eingabe:** `messages`, optional `context` (ein Dict für deterministische
  Schritte, zum Beispiel das ursprüngliche Domänenobjekt). **Ausgabe:**
  `messages`, `structured_response`, `outputs`, `stopped_at`.

### `create_router`

```python
create_router(model, routes, *, system_prompt=DEFAULT, allow_multiple=True, route_fn=None,
              synthesizer_prompt=DEFAULT, response_format=None, synthesize=None,
              structured_output_method="auto", name="router") -> CompiledStateGraph
```

- Routing: ein Aufruf mit strukturierter Ausgabe, der `RoutingDecision`
  zurückgibt, wobei
  `RoutingDecision.routes: list[{agent: Literal[names], task: str}]`. Der
  Agent-Katalog wird an `system_prompt` angehängt. `allow_multiple=False`
  bedeutet höchstens eine Route.
- `route_fn(state) -> [{"agent", "task"}]` ersetzt das LLM durch
  deterministisches Routing.
- Die ausgewählten Routen laufen parallel (`Send`); jeder Agent erhält seine
  `task` als Human-Nachricht.
- Synthese: Ein LLM führt die Ergebnisse zusammen (strukturiert mit
  `response_format`). Standardmäßig entfällt sie, wenn genau eine Route
  geantwortet hat und kein `response_format` gesetzt ist; `synthesize=True/False`
  erzwingt das eine oder das andere.
- **Ausgabe:** `messages`, `structured_response` (bei `response_format`), `routes`.
- Reservierte Routennamen: `route`, `synthesize`.

### `create_parallel`

```python
create_parallel(branches, *, aggregator=None, model=None, synthesizer_prompt=DEFAULT,
                response_format=None, structured_output_method="auto", name="parallel") -> CompiledStateGraph
```

Alle Zweige erhalten gleichzeitig die vollständigen Eingabenachrichten. Die
Aggregation, in der Reihenfolge des Vorrangs:
`aggregator(results_by_name: dict[str, AgentResult], state) -> value`
(Code), eine LLM-Synthese mit `model` (+ `response_format`) oder
Verkettung. **Ausgabe:** `messages`, `structured_response`, `branch_results`.

Tipp: Ein Agent ohne Tools plus `response_format` ist ein einzelner
strukturierter LLM-Aufruf und damit ein idealer Analysten-Zweig:
`create_agent(model, tools=[], system_prompt=..., response_format=Schema)`.

### `create_voting`

```python
create_voting(agent: AgentSpec, *, n=3, key=lambda r: r.text, name="voting")
```

Führt den Agenten `n`-mal parallel aus und gibt die Mehrheitsantwort zurück
(`key(AgentResult)` wählt aus, was verglichen wird; bei Gleichstand gewinnt die
früheste Stichprobe). **Ausgabe:** `messages` / `structured_response` des
Gewinners, plus `votes`. Verwenden Sie eine Sampling-Temperatur über 0.

### `create_map_reduce`

```python
create_map_reduce(mapper, *, reduce=lambda results: results, prepare=lambda item: item,
                  extract=lambda output: output, name="map_reduce")
```

Eingabe `{"items": [...]}`, Ausgabe `{"results": [...], "output": reduce(results)}`.
`mapper` ist ein beliebiges Runnable (ein Pattern-Graph, Ihr eigener Workflow,
eine Chain). Die Ergebnisse behalten die Reihenfolge der Items; eine leere
Eingabe ist kein Problem.

### `create_orchestrator`

```python
create_orchestrator(model, workers, *, planner_prompt=DEFAULT, synthesizer_prompt=DEFAULT,
                    response_format=None, max_rounds=1, max_tasks_per_round=10,
                    structured_output_method="auto", name="orchestrator") -> CompiledStateGraph
```

- Planer: ein Aufruf mit strukturierter Ausgabe, der `Plan` zurückgibt, wobei
  `Plan.tasks: list[{worker: Literal[names], instruction: str}]`. Eine leere
  Liste bedeutet: fertig.
- Aufgaben laufen parallel (`Send`); derselbe Worker kann mehrere Aufgaben
  erhalten.
- `max_rounds > 1` aktiviert die Neuplanung: Nach jeder Runde sieht der Planer
  alle Ergebnisse („Results so far“) und plant erneut.
- **Eingabe:** `messages`, optional `results` aus früherer Arbeit. Der Planer
  sieht sie dann und plant nur, was noch fehlt (ein leerer Plan geht direkt zur
  Synthese). Der Synthesizer schreibt aus alten und neuen Ergebnissen.
  `max_rounds` zählt die Runden eines Laufs; die Rundennummern setzen die
  früheren fort. Das nutzt eine Review-Schleife mit `carry_over=["results"]`.
- **Ausgabe:** `messages`, `structured_response`, `results`
  (`[{round, worker, task, output}]`, einschließlich früherer Arbeit).
  Reservierte Namen: `prepare`, `plan`, `synthesize`, `collect`.

### `create_supervisor`

```python
create_supervisor(model, subagents, *, system_prompt, tools=(), response_format=None,
                  delegation="tool_per_agent", input_mode="task", max_model_calls=25,
                  max_tool_calls=None, on_limit="error", max_calls_per_agent=None,
                  middleware=(), name="supervisor", **agent_kwargs) -> CompiledStateGraph
DelegationLimitMiddleware(limits, *, delegation="tool_per_agent")   # use with any create_agent supervisor
```

- `delegation="tool_per_agent"`: ein Tool pro Subagent (`name(task: str)`).
  `"task_tool"`: ein einzelnes Tool `task(agent_name: Literal[...], description: str)`,
  praktisch bei vielen Agenten oder bei Agenten, die einem Team gehören.
- `input_mode="task"`: isolierter Subagent-Kontext. `"fork"`: Der Subagent
  erhält zusätzlich die Konversation des Supervisors (Human- und AI-Text, kein
  Tool-Verkehr).
- Tools geben die strukturierte Antwort des Subagenten als JSON zurück oder
  seinen finalen Text.
- `tools`: zusätzliche Tools, die der Supervisor selbst nutzt. `middleware` und
  `agent_kwargs` (Checkpointer, Store, ...) gehen an `create_agent`.
- `max_model_calls` / `max_tool_calls` / `on_limit` begrenzen die eigene
  Schleife des Supervisors; Subagenten haben ihre eigenen Limits.
  `max_calls_per_agent` (ein int für alle oder `{"billing": 2}`) begrenzt die
  Delegationen pro Subagent und Lauf, siehe [Teilnehmer](#teilnehmer).

### `create_hierarchy`

```python
Team(name, description, system_prompt, members=[AgentSpec | Team, ...], response_format=None, model=None)
create_hierarchy(model, teams, *, system_prompt, response_format=None, name="hierarchy",
                 **supervisor_kwargs) -> CompiledStateGraph
build_team(model, team, **supervisor_kwargs) -> AgentSpec
```

Jedes `Team` wird zu einem Supervisor, der seiner übergeordneten Ebene als Tool
bereitgestellt wird (in beliebiger Tiefe). Mit `Team.response_format` geben
Teamleiter strukturierte Berichte zurück. `delegation`, `input_mode` und die
Limits (`max_model_calls`, `max_tool_calls`, `on_limit`, `max_calls_per_agent`)
gelten für alle Ebenen; ein `max_calls_per_agent`-Mapping kann Mitglieder jeder
Ebene benennen. Andere kwargs (`middleware`, `checkpointer`, ...) gehen nur an
die oberste Ebene.

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

- Jeder Agent erhält `transfer_to_<peer>(note)`-Tools für seine `handoffs`
  (Standard: alle Peers) und eine Teilnehmerliste in seinem Prompt.
- `history="full"` gibt die ganze Konversation weiter; `"handoff_only"` gibt
  nur den Handoff-Aufruf und seine `ToolMessage` (mit der Notiz) weiter.
- `max_handoffs` pro Lauf: Ist das Limit erreicht, lehnt das Handoff-Tool ab und
  fordert den Agenten auf, zum Ende zu kommen. `SwarmAgent.max_activations`: wie
  oft dieser Agent pro Lauf die Kontrolle übernehmen darf (der Start zählt mit).
- `max_model_calls` / `max_tool_calls` / `on_limit` begrenzen die Schleife jedes
  Agenten *pro Aktivierung*; `SwarmAgent.max_model_calls` / `max_tool_calls`
  überschreiben sie für einen einzelnen Agenten. Siehe
  [Limits und Budgets](#limits-und-budgets).
- **State / Ausgabe:** `messages`, `structured_response`, `active_agent`,
  `handoff_count`, `activations`. Mit einem `checkpointer` beginnt die nächste
  Gesprächsrunde beim `active_agent`.

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

- `system_prompt` ist ein Template: `{key}` wird aus dem State befüllt (Dicts
  und Modelle werden als JSON dargestellt). Es kann auch ein Callable
  `state -> str` sein.
- `transitions` erzeugen automatisch `go_to_<step>(reason)`-Tools. Eigene Tools
  wechseln den Schritt mit
  `return transition("next", runtime.tool_call_id, my_key=...)`.
- `tool_filter(state, tools) -> tools` schränkt die Tools eines Schritts zur
  Laufzeit ein.
- Nur Schritte mit `final=True` behalten `response_format`, damit der Agent
  nicht vorzeitig abschließen kann.
- Erweitern Sie `StateMachineState` um zusätzliche Schlüssel, die Ihre Prompts
  oder Tools verwenden.
- `Step.max_visits`: wie oft ein Lauf den Schritt betreten darf (der Start
  zählt mit); weitere Übergänge werden abgelehnt. `step_visits` in der Ausgabe
  zeigt die Zählerstände. `max_model_calls` begrenzt die Schleife über alle
  Schritte hinweg.

### `create_skills_agent`

```python
Skill(name, description, instructions, tools=())
create_skills_agent(model, skills, *, system_prompt, tools=(), response_format=None,
                    max_model_calls=25, max_tool_calls=None, on_limit="error",
                    middleware=(), name="skills_agent", **agent_kwargs) -> CompiledStateGraph
SkillsMiddleware(skills, *, always_available=(), show_catalog=True)   # use with any create_agent
```

- Fügt `load_skill(skill_name: Literal[...])` hinzu. Das Tool gibt die
  Anweisungen als Tool-Ergebnis zurück und vermerkt den Skill in
  `loaded_skills` (der Merge-Reducer macht paralleles Laden sicher).
- Tools nicht geladener Skills sind verborgen. `tools` (oder die Namen in
  `always_available`) bleiben sichtbar.
- Der Katalog (Name: Beschreibung) wird an den System-Prompt angehängt.

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

- `generator`: ein beliebiges Runnable nach dem Agent-Vertrag, auch andere
  Patterns.
- `evaluator`: entweder ein **Chat-Modell** (ein Aufruf mit strukturierter
  Ausgabe pro Review; `evaluator_prompt` ist sein System-Prompt und
  erforderlich) oder ein **Agent**, der das Urteil als `structured_response`
  zurückgibt, z. B.
  `create_agent(model, tools=[fetch_url], system_prompt=RUBRIC, response_format=Evaluation)`.
  Ein Agent als Reviewer kann Fakten mit Tools prüfen; `evaluator_prompt` ist
  für ihn optional und wird der Review-Anfrage hinzugefügt.
- Standardschema `Evaluation(passed, score, issues, feedback)`; verwenden Sie
  `passed=` für eigene Schemas.
- `evaluator_context(state) -> str | messages | None`: zusätzliches Material für
  den Reviewer neben der Anfrage und dem Kandidaten, typischerweise die Belege
  (`results_block(state.get("results", []), "Research results")`).
- `evaluator_input(state) -> str | messages`: was der Reviewer stattdessen
  erhält, also die gesamte Review-Anfrage. Ein Beispiel ist eine Aufgabe aus
  Ihrem eigenen Template, die den Kandidaten als Fakten ohne seine Begründung
  zeigt:
  `lambda s: [*s["request"], HumanMessage(REVIEW.format(work=s["candidate"]))]`.
  `state["request"]` enthält die Eingabenachrichten der Schleife. Ein String ist
  eine Human-Nachricht; ein Modell-Evaluator erhält weiterhin `evaluator_prompt`
  als System-Prompt. Schließt `evaluator_context` aus.
- `carry_over=["results", ...]`: Ausgabeschlüssel des Generators, die zwischen
  den Iterationen erhalten bleiben. Sie werden bei jeder Überarbeitung an den
  Generator zurückgegeben (sofern sein Eingabeschema sie akzeptiert), sind für
  `evaluator_context` sichtbar, können der Schleife als Eingabe übergeben werden
  und sind Teil ihrer Ausgabe. Pattern-Graphen halten ihre Arbeit außerhalb von
  `messages`. Auf diese Weise baut eine Überarbeitung darauf auf, statt von
  vorn zu beginnen.
- `keep_history=True`: Der Generator setzt seine eigene Konversation fort. Bei
  einem `create_agent`-Generator gehören seine Tool-Aufrufe dazu, sodass
  Überarbeitungen sie nicht wiederholen; für Pattern-Graphen verwenden Sie
  `carry_over`. `False`: Er beginnt neu mit der Anfrage, dem letzten Kandidaten
  und dem Feedback.
- `on_max_iterations(candidate, evaluation) -> replacement` legt einen sicheren
  Fallback fest, etwa die Eskalation an einen Menschen.
- `retry_policy=RetryPolicy(...)` wiederholt den Generierungs- und den
  Bewertungsschritt jeweils für sich, sodass ein fehlgeschlagenes Review die
  Generierung nicht wiederholt; `timeout=` begrenzt jeden Versuch in Async-Läufen
  (siehe [Retries und Timeouts](#retries-timeouts-checkpointer-laufkontext)).
- **Ausgabe:** `messages`, `structured_response`, `evaluation`, `iterations`
  sowie die `carry_over`-Schlüssel.

### Core-Hilfsfunktionen

| Hilfsfunktion | Zweck |
|---|---|
| `agent_as_tool(spec, *, name=None, description=None, input_mode="task")` | Stellt einen Agenten oder ein Pattern als Tool bereit (`task: str`); sync + async; `"fork"` gibt die Konversation des Aufrufers weiter. |
| `agent_as_node(agent, *, input=None, output=None, name=None)` | Passt einen Graphen nach dem Agent-Vertrag an einen übergeordneten Graphen mit anderem Schema an. Der Knoten ruft den Graphen direkt auf, daher bleibt er ein sichtbarer Subgraph (Graph-Ansicht, `get_state(subgraphs=True)`). |
| `invoke_agent(spec, task, config=None) -> AgentResult` | Aufruf mit einem String oder mit Nachrichten; gibt `name`, `text`, `structured`, `messages` zurück. |
| `final_text(result)` | Strukturierte Antwort als JSON oder der Text der letzten Nachricht. |
| `make_serializer(*types)` | Checkpoint-Serializer, der Ihre Pydantic-Typen auf die Allowlist setzt (LangGraph ≥ 1.2). |
| `structured_llm(model, schema, method="auto")` | Der Aufruf mit strukturierter Ausgabe, den die Patterns intern verwenden (siehe [echte Modelle](#8-echte-modelle-und-strukturierte-ausgabe)). |
| `results_block(results, header)` | Stellt `AgentResult`s oder Orchestrator-`results` als Prompt-Block dar, z. B. für `evaluator_context`. |
| `PatternState`, `PatternInput`, `PatternOutput` | State-Schemas des Vertrags, für Ihre eigenen kompatiblen Graphen. |

---

## Limits und Budgets

Jede Schleife in einem Multi-Agent-System braucht eine Obergrenze, und die
Schleifen sind verschachtelt: Die Schleife eines Supervisors ruft Subagenten
auf, die jeweils ihre eigene Schleife ausführen. Die Bibliothek begrenzt vier
Ebenen (`agentpatterns.limits`):

| Ebene | Begrenzt | Einstellungen | Wenn erreicht |
|---|---|---|---|
| [Agent-Schleife](#agent-schleifen) | Modell- und Tool-Aufrufe eines Agent-Laufs | `max_model_calls`, `max_tool_calls`, `on_limit` bei jeder Agent-Factory; `loop_limits(...)` für Ihr eigenes `create_agent` | Modell-Limit: Exception (`"error"`) oder Stopp mit einer finalen Nachricht (`"end"`); Tool-Limit: weitere Aufrufe werden abgelehnt, das Modell kommt zum Ende |
| [Teilnehmer](#teilnehmer) | wie oft ein Agent oder Schritt läuft | `max_calls_per_agent`, `SwarmAgent.max_activations`, `Step.max_visits`; Pattern-Schleifen `max_rounds`, `max_tasks_per_round`, `max_iterations`, `max_handoffs` | Ablehnung mit einer Nachricht an das Modell, das ohne den Teilnehmer weitermacht |
| [Ganzer Lauf](#ganzer-lauf-runbudget) | alle Modell- und Tool-Aufrufe eines Laufs, verschachtelte Agenten eingeschlossen | `RunBudget(...)` als Callback | `BudgetExceededError`; der Aufruf, der das Budget überschreitet, wird nicht ausgeführt |
| [Graph-Schritte](#graph-schritte-recursion_limit) | Super-Steps jedes Graphen | `recursion_limit` in der Lauf-Konfiguration | `GraphRecursionError` |

### Agent-Schleifen

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

Verfügbar bei `create_single_agent`, `create_supervisor`, `create_hierarchy`
(auf jeder Ebene), `create_swarm` (zusätzlich pro `SwarmAgent`),
`create_state_machine_agent` und `create_skills_agent`. Standard:
`max_model_calls=25`, kein Tool-Limit.

- **Gezählt pro Lauf des jeweiligen Agent-Graphen** (das native `run_limit`).
  Ein dreimal aufgerufener Subagent erhält drei frische Budgets, und ein
  Swarm-Agent erhält jedes Mal ein frisches, wenn er die Kontrolle übernimmt.
  Die Gesamtsummen multiplizieren sich daher: Ein Swarm kann etwa
  `(max_handoffs + 1) × max_model_calls` Modellaufrufe machen, und eine
  Hierarchie multipliziert pro Ebene. Verwenden Sie ein
  [`RunBudget`](#ganzer-lauf-runbudget) für eine Gesamtgrenze.
- `on_limit="error"` (Standard) löst `ModelCallLimitExceededError` aus (von
  `agentpatterns` re-exportiert). `"end"` stoppt die Schleife mit einer finalen
  `AIMessage`, die den Grund nennt, und ohne `structured_response`. Bei einem
  Subagenten erhält der Aufrufer diesen Text als Tool-Ergebnis und kann
  reagieren.
- `max_tool_calls` ist weich: Aufrufe darüber hinaus erhalten eine
  Fehler-`ToolMessage` („Do not make additional tool calls“), sodass das Modell
  trotzdem eine ordentliche Antwort geben kann.
- Für Limits über Gesprächsrunden hinweg (mit einem Checkpointer) oder für ein
  einzelnes Tool fügen Sie die native Middleware selbst hinzu:
  `middleware=[ModelCallLimitMiddleware(thread_limit=100), ToolCallLimitMiddleware(tool_name="web_search", run_limit=5)]`.

### Teilnehmer

```python
supervisor = create_supervisor(model, specialists, system_prompt=SUPERVISOR,
                               max_calls_per_agent={"research": 2, "billing": 1})   # or one int for all
swarm = create_swarm(model, [SwarmAgent("triage", "Front desk", TRIAGE, max_activations=1), ...])
agent = create_state_machine_agent(model, [Step("collect", COLLECT, transitions=["resolve"], max_visits=2), ...])
```

- `max_calls_per_agent` (Supervisor, Hierarchie) zählt die Delegationen pro
  Subagent und Lauf, einschließlich paralleler Aufrufe in einer Runde, für beide
  Delegationsstile. Ein Aufruf über dem Limit wird nicht ausgeführt; der
  Supervisor erhält eine Fehler-`ToolMessage` und macht mit den anderen weiter.
  Die Logik steckt in `DelegationLimitMiddleware`, die auch bei einem von Hand
  gebauten `create_agent`-Supervisor funktioniert.
- `SwarmAgent.max_activations`: wie oft der Agent pro Lauf die Kontrolle
  übernehmen darf, der Start eingeschlossen. Weitere Handoffs an ihn werden
  abgelehnt. `activations` in der Ausgabe zeigt die Zählerstände.
- `Step.max_visits`: wie oft der Lauf den Schritt betreten darf, der Start
  eingeschlossen. Ein abgelehntes `go_to_<step>` wird nicht ausgeführt. Der
  Übergang eines eigenen Tools (`transition(...)`) wird rückgängig gemacht,
  seine anderen Updates bleiben aber bestehen, weil das Tool bereits gelaufen
  ist; an seine `ToolMessage` wird die Ablehnung angehängt. `step_visits` in der
  Ausgabe zeigt die Zählerstände.
- Diese Zähler werden bei jedem Lauf (`invoke`) zurückgesetzt, sodass frühere
  Runden einer Konversation spätere nicht blockieren.

### Ganzer Lauf: `RunBudget`

```python
from agentpatterns import BudgetExceededError, RunBudget

budget = RunBudget(max_model_calls=60, max_tool_calls=120)
try:
    result = hierarchy.invoke(inputs, {"callbacks": [budget]})
except BudgetExceededError as error:          # error.kind ("model" | "tool"), error.limit
    escalate_to_human(error)
budget.model_calls, budget.tool_calls        # what the run used
```

- Ein LangChain-Callback-Handler. LangChain reicht Callbacks an jedes
  verschachtelte Runnable weiter. Daher sieht das Budget Subagenten hinter
  Tools, Swarm-Peers, parallele Zweige (sync und async) und die
  Pattern-internen Aufrufe (Routing, Planung, Synthese, Bewertung).
- Hart: Der Aufruf, der das Budget überschreiten würde, wird nicht ausgeführt,
  und der ganze Lauf stoppt mit `BudgetExceededError`. Mit einem Checkpointer
  behält der Thread seinen letzten abgeschlossenen Schritt und kann fortgesetzt
  werden.
- Verwenden Sie eine Instanz pro Lauf oder eine pro Sitzung, um mehrere Läufe
  gemeinsam zu budgetieren (`reset()` beginnt von vorn).
  `graph.with_config(callbacks=[budget])` bindet das Budget an einen Graphen.
- LangChain protokolliert „Error in RunBudget.on_chat_model_start callback“,
  bevor es die Exception erneut auslöst; diese Zeile ist zu erwarten.

### Graph-Schritte: `recursion_limit`

LangGraph stoppt einen Graphen nach `recursion_limit` Super-Steps
(`GraphRecursionError`): `graph.invoke(inputs, {"recursion_limit": 200})`. Der
Standardwert in LangGraph 1.2 ist 10.007 (`LANGGRAPH_DEFAULT_RECURSION_LIMIT`).
Er ist eine Rückfallsicherung, kein Budget:

- Er zählt Graph-Schritte, nicht Modellaufrufe. In einer `create_agent`-Schleife
  ist jeder Middleware-Hook ein eigener Knoten, daher braucht eine Modellrunde
  mehrere Schritte.
- Verschachtelte Graphen erben den Wert aus der Lauf-Konfiguration, zählen aber
  ihre eigenen Schritte. Er begrenzt also auch nicht die Gesamtzahl eines Laufs.

Setzen Sie ihn höher als das, was die echten Limits zulassen, damit er nie
zuerst greift.

---

## Nachrichten zwischen Agenten

Die Agenten eines Laufs können einander Nachrichten senden, ähnlich wie
Claude-Code-Sitzungen. Ein Agent aktiviert das mit `MessagingMiddleware`
(`agentpatterns.messaging`), und eine `Mailbox` transportiert die Nachrichten
eines Laufs:

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

**Was der Agent bekommt**

- `list_agents`: die Agenten des Laufs, denen er schreiben darf, mit Adresse,
  Status (`running`, `finished` oder `not started` für Namen in `can_message`)
  und Beschreibung.
- `send_message(to, message)`: Das Ergebnis sagt, was mit der Nachricht
  geschehen ist (siehe Zustellung unten). Eine abgelehnte Nachricht (nicht in
  `can_message`, über dem Limit, eine unbekannte oder mehrdeutige Adresse, die
  eigene Adresse) kommt als Fehler-Tool-Ergebnis zurück, und der Agent macht
  ohne sie weiter.
- Nachrichten anderer Agenten, zugestellt vor seinem nächsten Modellaufruf als
  User-Nachricht `<agent-message from="researcher" to="analyst">...</agent-message>`.
  Sie bleiben in der Konversation des Agenten und in seinen Checkpoints.
- Eine kurze Ergänzung des System-Prompts: seine eigene Adresse, dass andere
  Agenten weder seine Antwort noch seine Tool-Ergebnisse sehen und dass
  Peer-Nachrichten von Kollegen kommen, nicht vom Nutzer.
- Will er mit ungelesenen Nachrichten abschließen, schickt die Middleware ihn
  mit diesen Nachrichten zurück zum Modell. `max_model_calls` gilt weiterhin.

**So funktioniert es**

- **Eine `Mailbox` pro Lauf**, übergeben als `config["configurable"]["mailbox"]`.
  Die Config erreicht jeden verschachtelten Agenten, auch Subagenten hinter
  Delegations-Tools. Eine an den Graphen gebundene Mailbox würde die Nachrichten
  gleichzeitiger Läufe vermischen. In einem Lauf ohne Mailbox tut die Middleware
  nichts und verbirgt ihre Tools, sodass derselbe Agent mit und ohne Messaging
  funktioniert.
- **Außerhalb des Graph-States.** Parallele Zweige sehen die State-Updates der
  anderen erst nach dem Schritt. Das ist zu spät für Agenten, die gleichzeitig
  laufen. Die Mailbox ist ein In-Memory-Objekt, das sich sicher aus parallelen
  Zweigen verwenden lässt (Threads oder async).
- **Adressen.** Die Adresse eines Agenten ist sein Name: `MessagingMiddleware("x")`
  oder standardmäßig `create_agent(name=...)`. Eine einzelne Middleware ohne
  Namen kann daher jedem Agenten eines Swarms dienen
  (`create_swarm(..., middleware=[MessagingMiddleware()])`). Kopien eines
  Agenten, die im selben Lauf laufen (Map-Reduce-Items, ein Orchestrator-Worker
  mit mehreren Aufgaben, ein parallel aufgerufener Subagent), erhalten
  `researcher`, `researcher#2`, ... in der Reihenfolge ihres Starts. Jeder
  Aufruf eines Agenten ist eine neue Kopie: Ein Supervisor, der einen Subagenten
  erneut aufruft, startet einen weiteren Lauf davon. Das gilt auch für einen
  Swarm-Agenten, der die Kontrolle erneut übernimmt.
- **Zustellung.**
  - Eine Nachricht an einen laufenden Agenten wird für seinen nächsten
    Modellaufruf eingereiht.
  - Eine Nachricht an einen Agenten, der nicht läuft, geht an den laufenden
    Agenten mit demselben Namen. Gibt es keinen, wartet sie auf den nächsten
    Agenten mit diesem Namen, der im Lauf startet: ein späterer
    Pipeline-Schritt, die nächste Swarm-Aktivierung, der nächste Aufruf eines
    Subagenten. Laufen mehrere, wird das Senden abgelehnt, und der Absender
    wählt einen aus.
  - `send_message` teilt dem Absender mit, welcher dieser Fälle eingetreten ist,
    z. B. „Sent to analyst. It is running and gets the message before its next
    step.“ oder „early has finished. The message waits and is delivered if an
    agent named early starts again in this run.“
- **Retries und Fortsetzungen.** Ein wiederholter Schritt (`retry_policy`)
  behält die Adresse seines Agenten, und die Wiederholung erhält alle Nachrichten
  des Agenten erneut. Mit einem Checkpointer vermerkt der State des Agenten, wie
  viele Nachrichten er gelesen hat, sodass ein fortgesetzter Agent sie nicht
  doppelt erhält; übergeben Sie beim Fortsetzen dieselbe `Mailbox`. Auf einem
  Thread mit Checkpoints kann jede Gesprächsrunde eine neue Mailbox verwenden;
  der Agent beginnt dann bei der ersten Nachricht dieser Mailbox zu lesen.
- **Aus Code.** `mailbox.send("researcher", "Also check 2025.", sender="ops")`
  sendet eine Nachricht von außerhalb der Agenten, zum Beispiel von einer
  Person, die den Lauf beobachtet, oder aus einem Test.

**Wann es sich lohnt.** Nachrichten helfen nur Agenten, die gleichzeitig laufen.
Überall sonst reichen die Patterns Ergebnisse bereits weiter:

| Pattern | Laufen die Agenten gleichzeitig? | Messaging |
|---|---|---|
| `create_parallel`, `create_map_reduce`, Orchestrator-Worker in einer Runde, die parallelen Delegationen eines Supervisors | ja | Nützlich: Erkenntnisse teilen, doppelte Arbeit vermeiden, einen Peer fragen |
| Pipeline, Evaluator-Optimizer | nein | Nur Notizen für spätere Schritte; das leisten bereits die Schrittausgaben |
| Swarm | einer nach dem anderen | Das leistet bereits die Handoff-Notiz |
| Supervisor und der Subagent, den er aufruft | der Supervisor wartet | Die Nachricht kommt zusammen mit dem Tool-Ergebnis an |
| Router, Single Agent, State Machine, Skills | ein Agent | Niemand zum Reden da |
| Voting | ja | Weglassen: Die Stichproben sollen unabhängig bleiben |

**Sicherheit und Limits**

- Peer-Nachrichten kommen als User-Nachrichten an. Sie sind aber als Nachrichten
  von Kollegen markiert und erklärt, nicht als Nachrichten des Nutzers. Eine
  Prompt-Injection in Daten, die ein Agent liest (eine E-Mail, eine Webseite),
  kann diesen Agenten trotzdem dazu bringen, Nachrichten zu senden. Legen Sie
  mit `can_message` fest, wer wem schreiben darf, und sichern Sie Seiteneffekte
  durch Prüfungen in den Tools oder durch menschliche Freigabe beim empfangenden
  Agenten ab, wie bei der Delegation.
- `input_mode="fork"` gibt die Konversation des Aufrufers ohne Peer-Nachrichten
  weiter.
- `Mailbox(max_messages_per_agent=10)` (der Standard) begrenzt die Nachrichten,
  die ein Agent-Lauf senden darf. Jede zugestellte Nachricht kostet den
  Empfänger einen Modellaufruf; `max_model_calls` und `RunBudget` begrenzen die
  Gesamtzahl.

**(Noch) nicht abgedeckt**

- Ein Agent kann nicht blockieren, um auf eine Antwort zu warten. Kommt die
  Antwort, nachdem der Agent fertig ist, wartet sie auf den nächsten Lauf
  dieses Agenten.
- Die Mailbox lebt im Arbeitsspeicher. Läufe, die in einem anderen Prozess
  fortgesetzt werden, verlieren ihre Nachrichten.
- Ein Agent-Lauf, der ohne Retry fehlschlägt, behält den Status `running`.

---

## Integrationsleitfaden

### 1. Ein Pattern in Ihren Workflow einbetten (Subgraph)

**Gleicher Schlüssel (`messages`):** Fügen Sie das kompilierte Pattern direkt
als Knoten hinzu.

```python
from langgraph.graph import StateGraph, MessagesState, START

team = create_parallel([...])
graph = StateGraph(MessagesState).add_node("team", team).add_edge(START, "team").compile()
```

**Anderes Schema:** Bilden Sie Ein- und Ausgaben mit `agent_as_node` ab. So
betten alle E-Mail-Workflows ihr Pattern ein (`email_assistant/common.py`):

```python
from agentpatterns import agent_as_node

node = agent_as_node(
    router,
    input=lambda state: {"messages": [HumanMessage(state["email"].as_prompt())]},
    output=lambda result, state: {"resolution": result["structured_response"]},
)
builder.add_node("resolve", node)
```

### 2. Ein Pattern als Tool bereitstellen

```python
from agentpatterns import AgentSpec, agent_as_tool

help_desk = agent_as_tool(AgentSpec("help_desk", "Answers product questions.", router))
chat = create_agent(model, tools=[help_desk], checkpointer=InMemorySaver())
```

So machen Sie auch einen zustandslosen Router oder eine zustandslose Pipeline
in einem Multi-Turn-Chat nutzbar: Der Chat-Agent behält das Gedächtnis.

### 3. Patterns verschachteln

Jede Factory akzeptiert `AgentSpec`s, deren `agent` ein anderes Pattern ist:

```python
desk = create_router(model, specialists)
checked = create_evaluator_optimizer(desk, model, evaluator_prompt=RULES)
top = create_supervisor(model, [AgentSpec("desk", "Routed help desk", checked), ...], system_prompt=...)
```

### 4. Persistenz, Gedächtnis, Multi-Turn

- Kompilieren oder erstellen Sie den **äußersten** Graphen mit einem
  Checkpointer, und übergeben Sie eine `thread_id`:
  `graph.invoke(input, {"configurable": {"thread_id": "customer-42"}})`.
- Innere Agenten und Subagenten verwenden die standardmäßige Persistenz pro
  Aufruf (innerhalb eines Aufrufs erben sie den Checkpointer des übergeordneten
  Graphen). Das ist für zustandslose Spezialisten richtig und unterstützt
  Interrupts.
- Swarm und State Machine halten `active_agent` / `current_step` im State,
  sodass Multi-Turn-Konversationen dort weitergehen, wo sie aufgehört haben.
- LangGraph ≥ 1.2 checkpointet ohne Warnungen nur Klassen auf der Allowlist:
  Verwenden Sie `InMemorySaver(serde=make_serializer(MySchema, ...))` (oder
  dasselbe `serde` bei Postgres-/SQLite-Savern).

### Retries, Timeouts, Checkpointer, Laufkontext

Jede Factory nimmt dieselben Dinge entgegen, sodass sich eine Komposition wie
ein einziger Graph verhält:

- **`checkpointer=`** bei jeder Factory. Nur der äußerste Graph braucht einen;
  die Graphen darin schreiben ihre Checkpoints in ihn.
- **`retry_policy=`** (eine LangGraph-`RetryPolicy` oder mehrere) und
  **`timeout=`** bei jeder Factory, die ihren eigenen Graphen baut
  (`create_evaluator_optimizer`, `create_orchestrator`, `create_router`,
  `create_parallel`, `create_voting`, `create_map_reduce`, `create_pipeline`,
  `create_swarm`). Beide gelten für jeden Schritt, der einen Agenten oder ein
  Modell ausführt, nicht für die Verwaltungsschritte des Patterns:
  - Ein Schritt, der eine Exception auslöst, wird für sich wiederholt. Ein
    einzelner vorübergehender Fehler startet also nicht das ganze Pattern neu.
  - `timeout` (Sekunden, ein `timedelta` oder eine `TimeoutPolicy`, die auch die
    Leerlaufzeit begrenzen kann) bricht einen Versuch, der länger dauert, mit
    `NodeTimeoutError` ab. Die Standard-`RetryPolicy` wiederholt ihn. So gibt
    `retry_policy=RetryPolicy(max_attempts=3),
    timeout=30` einem hängenden Modellaufruf zwei weitere Versuche von je 30 s.
    Das Timeout begrenzt einen Versuch eines Schritts, nicht den Lauf.
  - Timeouts sind LangGraph-Knoten-Timeouts und brauchen einen Async-Lauf
    (`ainvoke`, `astream`). LangGraph kann synchronen Code nicht abbrechen,
    daher verweigert `invoke` einen Schritt mit Timeout (`ValueError`).
  - Retries kosten Aufrufe: Ein [`RunBudget`](#ganzer-lauf-runbudget) zählt jeden
    Versuch. `BudgetExceededError` ist ein `RuntimeError`, den die
    Standard-`RetryPolicy` nicht wiederholt; belassen Sie es dabei, wenn Sie
    `retry_on=` übergeben.
- **Die `create_agent`-Factories** (Single Agent, Supervisor, Hierarchie, State
  Machine, Skills) wiederholen stattdessen über LangChains Middleware:
  `middleware=[ModelRetryMiddleware(on_failure="error"), ToolRetryMiddleware(...)]`.
  Übergeben Sie `on_failure="error"`: Der Standard, `"continue"`, beendet die
  Agent-Schleife mit dem Fehlertext als Antwort, die ein Aufrufer (ein
  Supervisor, eine Review-Schleife) dann für eine echte hält. Diese Factories
  haben keine eigenen Schritte, die ein Timeout haben könnten. Verwenden Sie das
  Request-Timeout des Chat-Modells (z. B. `ChatAnthropic(timeout=...)`) oder das
  `timeout=` des Patterns oder Graph-Knotens, der sie ausführt.
- **Der Laufkontext** (`graph.invoke(input, context=...)`) erreicht jeden
  Agenten innerhalb eines Patterns, Subagenten hinter einem Delegations-Tool
  eingeschlossen, als `runtime.context` / `ToolRuntime.context`. Die Patterns
  deklarieren keinen eigenen Kontexttyp; bauen Sie Ihre Agenten mit
  `context_schema=`, wenn Sie ihn typisiert haben möchten. So wird ein Agent
  einmal gebaut, und was sich zwischen Läufen unterscheidet (ein Mandant, eine
  Server-URL, die Umgebung eines Eval-Falls), kommt mit jedem Lauf.

### MCP-Tools vom Server des Laufs

`agentpatterns.mcp` (benötigt das Extra `mcp`, `multiagent-patterns[mcp]`) gibt
einem Agenten die Tools eines MCP-Servers, den der Lauf benennt. So wird der
Agent nur einmal gebaut, auch wenn jeder Lauf mit einem anderen Server spricht,
z. B. mit einem pro Eval-Fall:

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

Es baut auf LangChains `langchain.mcp` auf (Beta, auf Basis von FastMCP), das
das archivierte `langchain-mcp-adapters` ersetzt. Beim Import wird einmalig
gewarnt, dass sich die API ändern kann.

- **Das Ziel** ist alles, was `MCPAdapter` akzeptiert: eine http(s)-URL oder ein
  `fastmcp.Client`, wenn der Lauf mehr als eine Adresse braucht, etwa
  Authentifizierung (`Client(url, auth=token)`), ein Request-Timeout oder einen
  zwischen Läufen geteilten Cache für die Tool-Liste (siehe unten). `target(ctx)`
  wird beim Start und bei jedem Modell- und Tool-Aufruf aufgerufen und sollte
  daher kein I/O ausführen.
- Wenn der Agent startet, listet `McpTools` die Tools des Servers einmal auf
  (nur die in `tools=` genannten; fehlt eines, schlägt der Start fehl) und hält
  ihre Definitionen im State des Agenten. Sie werden jedem Modellaufruf
  angeboten, und jeder Aufruf eines dieser Tools verbindet sich mit dem Server
  des Laufs.
- Ein Fehler, den der Server meldet, erreicht das Modell als Antwort des Tools.
  `unreachable(error)` unterscheidet davon einen Server, der nicht erreichbar
  war (ein Transportfehler oder 502/503/504 von einem vorgeschalteten Gateway):
  Wiederholen Sie Tool-Aufrufe mit `ToolRetryMiddleware(retry_on=unreachable, ...)`
  (ihre `on_failure`-Nachricht informiert das Modell) und einen fehlgeschlagenen
  Start mit der `retry_policy=RetryPolicy(retry_on=unreachable)` des umgebenden
  Schritts. Setzen Sie `ToolRetryMiddleware` **vor** `McpTools`: `McpTools`
  verbindet sich um jeden Aufruf herum, daher deckt nur ein äußerer Retry einen
  fehlgeschlagenen Verbindungsaufbau ab.
- **Ein Server, der während eines Aufrufs nach Eingaben fragt** (MCP-Elicitation),
  unterbricht den Lauf. Der Wert des Interrupts ist `{"type": "mcp_elicitation", "tool_name",
  "requests": [{"key", "message", ...}]}`; setzen Sie fort mit
  `Command(resume={"responses": {key: {"action": "accept", "content": {...}}}})`.
  Wie jeder Interrupt braucht er einen Checkpointer.
- **Tool-Liste über Läufe hinweg cachen:** Übergeben Sie einen `fastmcp.Client`
  mit einem gemeinsamen Cache-Store, partitioniert pro Mandant, damit kein
  Mandant die Liste eines anderen sieht:

  ```python
  from fastmcp import Client
  from fastmcp.client.caching import KeyValueResponseCacheStore
  from mcp.client.caching import CacheConfig

  store = KeyValueResponseCacheStore()     # in memory; pass storage=RedisStore(...) to share it between processes
  McpTools(lambda ctx: Client(ctx.mcp_url, cache=CacheConfig(store=store, partition=ctx.tenant, target_id="records")))
  ```

  Der Server entscheidet, wie lange eine Liste gecacht werden darf (`ttlMs`; ein
  FastMCP-Server setzt das mit `FastMCP(..., cache_ttl=60)`). Ohne TTL vom
  Server wird nichts gecacht, und jeder Lauf listet erneut.
- Mehrere Server an einem Agenten: jeweils ein `McpTools` mit eigenem `server=`.
- Nur async, wie der MCP-Client.

### 5. Human-in-the-Loop

- **Vor riskanten Tools:** Übergeben Sie
  `middleware=[HumanInTheLoopMiddleware(interrupt_on={"issue_refund": True})]`
  an `create_single_agent`, `create_supervisor`, `create_swarm`,
  `create_state_machine_agent` oder `create_skills_agent`.
- **Vor Seiteneffekten in einem Workflow:** Fügen Sie einen Knoten hinzu, der
  `interrupt(...)` aufruft, und setzen Sie mit
  `graph.invoke(Command(resume="approve"), config)` fort. Siehe
  `human_approval` in `email_assistant/library_based/composite.py`.
- Interrupts, die in Subagenten ausgelöst werden, propagieren zum
  Top-Level-Graphen, wenn nur dieser Graph einen Checkpointer hat.

### 6. Streaming und Observability

- `graph.stream(input, stream_mode="updates", subgraphs=True)` streamt
  verschachtelte Pattern-Schritte. Subagenten, die innerhalb von Tools
  aufgerufen werden, sind keine statisch sichtbaren Subgraphen. Ihre
  Modellaufrufe erscheinen aber trotzdem in Callbacks und Traces.
- **Graph-Ansicht:** `graph.get_graph(xray=True)` (das, was LangGraph Studio und
  ähnliche Tools zeichnen) klappt jeden Teilnehmer von Router, Parallel,
  Voting, Map-Reduce, Orchestrator, Pipeline (`agent_step`), Evaluator-Optimizer
  und Swarm in eigene Knoten auf, Ebene für Ebene (`xray=N` begrenzt die Tiefe).
  `get_state(config, subgraphs=True)` zeigt den State eines unterbrochenen
  Teilnehmers. Einschränkungen:
  - Subagenten von Supervisor und Hierarchie sind Tools und erscheinen daher nur
    als Knoten `tools`.
  - Ein eingebetteter Swarm bleibt ein einzelner Kasten: LangGraph löst nur
    Subgraphen mit einem einzigen Ausgang inline auf, und jeder Swarm-Agent kann
    den Lauf beenden.
  - `draw_mermaid()` braucht Subgraph-Knotennamen, die über alle Ebenen hinweg
    eindeutig sind (eine Einschränkung von `langchain_core`; der JSON-Graph hat
    kein solches Limit).
- LangSmith-Tracing funktioniert ohne weiteres Zutun; Agent-Namen (`name=`)
  erscheinen als Metadaten `lc_agent_name`.
- `agentpatterns.testing.UsageTracker` ist ein Callback, der Modellaufrufe,
  Token (aus `usage_metadata`) und Tool-Aufrufe pro Agent zählt.

### 7. Async

Alle Factories funktionieren mit `ainvoke` / `astream`. Modell- und
Agent-Aufrufe in Knoten und Delegations-Tools laufen nativ async, und parallele
Zweige laufen gleichzeitig.

### 8. Echte Modelle und strukturierte Ausgabe

- Übergeben Sie ein beliebiges LangChain-Chat-Modell
  (`init_chat_model("anthropic:claude-opus-5")`).
- **Agent-basierte Patterns** (Single Agent, Supervisor, Hierarchie, Swarm,
  State Machine, Skills und jeder `create_agent`-Teilnehmer): Mit
  `response_format=Schema` wählt LangChain `ProviderStrategy` (native
  strukturierte Ausgabe), wenn das Profil des Modells Unterstützung deklariert,
  und sonst `ToolStrategy`. Übergeben Sie `ProviderStrategy(Schema)` oder
  `ToolStrategy(Schema)`, um selbst zu entscheiden.
- **Graph-Patterns** machen ihre eigenen strukturierten Aufrufe
  (Routing-Entscheidung, Plan, Synthese, `llm_step`, Modell-Evaluator). Sie
  folgen derselben Regel über `structured_output_method="auto"`:
  `with_structured_output(schema,
  method="json_schema")`, wenn das Profil des Modells native strukturierte
  Ausgabe deklariert, sonst der Standard des Providers. Übergeben Sie
  `"function_calling"` (oder einen anderen `method=`-Wert Ihres Providers), um
  das zu überschreiben.
- Warum das für Claude wichtig ist: Der Standard von `with_structured_output`
  für Claude ist erzwungenes Tool Calling. Claude Opus 5.5, Claude Sonnet 5.5
  und Claude Fable 5.1 lehnen ein erzwungenes `tool_choice` (`any`/`tool`) ab,
  ebenso jedes Claude-Modell mit aktiviertem Extended Thinking.
  `langchain-anthropic` weicht dann auf einen *nicht erzwungenen* Tool-Aufruf
  aus und löst `OutputParserException` aus, wenn das Modell stattdessen mit Text
  antwortet. Native strukturierte Ausgabe (`output_config.format`) hat diese
  Lücke nicht, deshalb bevorzugt `"auto"` sie. (Geprüft mit
  `langchain-anthropic` 1.7.4: Das Profil jedes aktuellen Claude-Modells
  deklariert native Unterstützung.)
- Führen Sie `tests/test_real_model.py` gegen Ihr Modell aus, bevor Sie sich
  darauf verlassen (siehe [Testen](#testen)).
- Verwenden Sie pro Rolle unterschiedliche Modelle, indem Sie Teilnehmer mit
  eigenen Modellen bauen (`AgentSpec(..., create_agent(small_model, ...))`,
  `Team(model=...)`, `SwarmAgent(model=...)` und der `evaluator` einer
  Review-Schleife).

---

## Komposition

`src/email_assistant/library_based/composite.py` zeigt die Art von System, für
die die Bibliothek gedacht ist, vollständig aus Factories gebaut:

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

Jeder der elf bibliotheksbasierten Workflows in `email_assistant/library_based/`
ist ein 20-75 Zeilen langes Beispiel für eine Factory. Jeder liefert exakt
dieselben Ergebnisse wie sein handgeschriebenes Gegenstück in
`email_assistant/native/` (durch die Tests sichergestellt).

## Rezept: Recherche mit Review

Eine häufige Kombination: ein Orchestrator, der Rechercheaufgaben per Fan-out an
Researcher-Subagenten verteilt, und ein unabhängiger Reviewer, der das Ergebnis
prüft und es mit Feedback zurückschickt. Modellieren Sie das als **Orchestrator
innerhalb einer Review-Schleife**. Die Schleife ist das äußere Pattern, weil die
Prüfung des fertigen Ergebnisses ein eigenes Anliegen ist. Sie erledigt
idealerweise ein Bewerter mit frischem Kontext und eigenen Kriterien.

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

Die beiden Optionen, die das möglich machen:

- `carry_over=["results"]`: Ohne diese Option startet jede Überarbeitung den
  Orchestrator von vorn, und die gesamte Recherche läuft erneut. Mit ihr sieht
  der Planer die früheren Ergebnisse neben dem Feedback und plant nur die Lücken
  (oder nichts, wenn es im Feedback um das Schreiben geht).
- `evaluator_context`: Ohne diese Option sieht der Reviewer nur die Anfrage und
  den Bericht. Er kann dann Struktur und Vollständigkeit beurteilen, aber nicht,
  ob die Aussagen durch die Recherche gedeckt sind.

### Wie der State fließt

Drei Graphen sind beteiligt: die Review-Schleife (außen), der Orchestrator (der
Generator der Schleife) und die Researcher-Agenten (die Worker des
Orchestrators). Jeder hat seinen eigenen State. Sie tauschen nur aus, was ihre
Ein- und Ausgabeschemas deklarieren.

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

Nachverfolgt mit dem Scripted Model (`test_review_loop_revises_research_without_redoing_it`):

| Schritt | Graph | Was sich ändert |
|---|---|---|
| 1 | Orchestrator | `plan`: 2 Aufgaben (Markt, Wettbewerber); zwei Researcher laufen parallel; `results` = 2 Einträge (Runde 1) |
| 2 | Orchestrator | `synthesize`: Report mit 2 Quellen |
| 3 | Schleife | `candidate` = Report, `results` = 2 Einträge, `iterations` = 1 |
| 4 | Schleife | Reviewer sieht die 2 Ergebnisse und den Report: `passed=False`, „Add pricing research.“ |
| 5 | Orchestrator | Eingabe: Anfrage, Report v1, Feedback und die 2 Ergebnisse. `plan`: 1 Aufgabe (Preise, Runde 2) |
| 6 | Orchestrator | `synthesize` aus 3 Ergebnissen: Report mit 3 Quellen |
| 7 | Schleife | Reviewer: `passed=True`; `finish` gibt den Report, 3 `results` und `iterations=2` zurück |

Insgesamt drei Suchen. Ohne `carry_over` beginnt die zweite Iteration von vorn:
Der Planer sieht die früheren Ergebnisse nicht und führt die ersten beiden
Suchen erneut aus.

Was privat bleibt: `plan`, `round` und `last_round` des Orchestrators,
`request`, `candidate` und `generator_messages` der Schleife sowie die eigene
Konversation jedes Researchers (nur sein finaler Text gelangt in `results`).

### Was der umgebende Workflow definieren muss

Nur das, was er lesen oder hineingeben will. Es gibt zwei Wege, die Schleife
einzubetten.

**Gemeinsame Schlüssel:** Fügen Sie die kompilierte Schleife direkt als Knoten
hinzu. Der übergeordnete State braucht `messages` plus die Ausgabeschlüssel der
Schleife, die Sie interessieren:

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

Weil `results` auch eine Eingabe der Schleife ist, setzt eine Folgefrage im
selben Thread auf der gespeicherten Recherche auf
(`test_review_loop_as_subgraph_builds_on_earlier_research`). Geben Sie
`results` im übergeordneten State keinen `operator.add`-Reducer: Die Schleife
gibt bereits frühere plus neue Ergebnisse zurück, ein Anhängen würde sie also
duplizieren.

**Ihr eigenes Schema:** Bilden Sie Ein- und Ausgabe mit `agent_as_node` ab:

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

In beiden Fällen gilt: Registrieren Sie Ihre Schemas mit `make_serializer`, wenn
der Workflow einen Checkpointer hat. Setzen Sie `on_max_iterations`, wenn ein
Bericht, der das Review nie besteht, eskaliert statt zurückgegeben werden soll.

## Testen

Mit `agentpatterns.testing` können Sie die Orchestrierung ohne API-Aufrufe per
Unit-Test prüfen:

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

- Die Policy sieht die Nachrichten und die gebundenen Tools (`turn.system`,
  `turn.tool_names`, `turn.tool_results()`, `turn.call_results()`,
  `turn.structured_tool_schema()`, ...) und gibt Text, Tool-Aufrufe (parallel
  über `call_many`) oder strukturierte Ausgabe zurück.
- Das Modell löst `ScriptError` bei Dingen aus, die ein echtes Modell nicht
  kann: ein nicht gebundenes Tool aufrufen oder mit Text antworten, wenn ein
  Tool-Aufruf erzwungen ist.
- Policies sind reine Funktionen der Konversation. Daher bleiben sie bei
  paralleler Ausführung deterministisch, anders als listenbasierte Fake-Modelle.
- `UsageTracker` (ein Callback) prüft Aufrufzahlen und Kostenregressionen per
  Assertion.
- `turn.agent_messages` listet die Nachrichten anderer Agenten in der
  Konversation auf (`AgentMessage`: `sender`, `to`, `text`), für Agenten mit
  `MessagingMiddleware` (`tests/test_messaging.py`).
- `serve_mcp(server)` stellt einen MCP-Server (`fastmcp.FastMCP` oder
  `MCPServer` aus dem MCP-SDK) für die Dauer eines `with`-Blocks über HTTP bereit
  und liefert seine URL. So laufen Agenten mit `McpTools` in Tests gegen einen
  echten MCP-Server (`tests/test_mcp.py`).
- `ScriptedChatModel(policy=..., profile={"structured_output": True})` simuliert
  ein Modell mit nativer strukturierter Ausgabe: `create_agent` verwendet dann
  `ProviderStrategy` und die Bibliothek `method="json_schema"`, wie bei aktuellen
  Claude-Modellen. `turn.structured(...)` antwortet auf beiden Wegen korrekt,
  sodass dieselbe Policy beide abdeckt (die E-Mail-Tests führen jeden Workflow
  auf beide Arten aus).

`tests/test_library.py` enthält einen kompakten Test für jede Factory, der
zugleich als Anwendungsbeispiel dient.

**Smoke-Tests mit echten Modellen.** `tests/test_real_model.py` führt die
Patterns gegen ein echtes Modell aus und prüft, ob jedes seine deklarierte
Struktur zurückgibt (nicht die Qualität der Antworten). Die Tests werden
übersprungen, sofern Sie sie nicht ausdrücklich aktivieren:

```bash
uv sync --extra anthropic
AGENTPATTERNS_TEST_MODEL=anthropic:claude-opus-5 uv run pytest -m real_model
```

Führen Sie sie für jedes Modell aus, das Sie verwenden wollen, besonders für
Modelle ohne erzwungene Tool-Nutzung (Claude Opus 5.5, Claude Sonnet 5.5,
Claude Fable 5.1). Ein Lauf macht etwa 40-60 Modellaufrufe. Der
Anthropic-Provider liest seinen Schlüssel aus `ANTHROPIC_API_KEY`.

## Einschränkungen

- **Handoffs und parallele Tool-Aufrufe:** Ein Handoff-Tool, das in derselben
  Runde wie andere Tools aufgerufen wird, erzeugt eine History mit unvollständig
  gepaarten Einträgen (eine Einschränkung von LangGraph, die alle
  Command-basierten Handoffs teilen). Weisen Sie die Agenten im Prompt an,
  Handoffs allein auszuführen.
- **Inspektion des Subagent-States:** Subagenten, die innerhalb von Tools
  aufgerufen werden (Supervisor, Hierarchie), sind für
  `get_state(subgraphs=True)` und die Graph-Ansicht nicht sichtbar. Verwenden
  Sie Patterns mit Graph-Knoten (Router, Orchestrator, Parallel), wenn Sie
  verschachtelten State inspizieren oder zeichnen müssen.
- **Der State von Router, Parallel und Orchestrator** speichert
  `AgentResult`-Objekte. Registrieren Sie sie bei einem Checkpointer über
  `make_serializer` (bereits enthalten).
- **Übernommene Schlüssel** gelten in der Review-Schleife nach dem
  Last-Value-Prinzip: Vom Generator wird erwartet, dass er den vollständigen
  Wert zurückgibt (wie der Orchestrator bei `results`). Ein Generator, der den
  Schlüssel nicht als Eingabe akzeptiert, hat ihn zwar in seiner Ausgabe, kann
  aber nicht darauf aufbauen.
- **Kosten:** Multi-Agent-Patterns vervielfachen die Token; siehe den
  gemessenen Vergleich in [use-case.md](use-case.md#gemessener-vergleich) und die
  Berichte pro Pattern.
