Deutsch | [English](README.en.md)

# Multi-Agent-Patterns mit LangGraph

Recherche, Referenzimplementierungen und eine wiederverwendbare Bibliothek für
Multi-Agent-Architekturen mit LLMs, gebaut auf **LangGraph 1.2 / LangChain 1.4**.

Wir haben einen einzigen fachlichen Anwendungsfall (**Kunden-E-Mails
kategorisieren, bearbeiten und beantworten**) mit elf Patterns umgesetzt:

Single Agent · Sequential Pipeline · Router · Parallelisierung (Sectioning /
Voting / Map-Reduce) · Orchestrator-Workers · Supervisor (Subagents als Tools) ·
hierarchische Teams · Swarm (Handoffs) · State Machine (Handoffs per
Middleware) · Skills (Progressive Disclosure) · Evaluator-Optimizer

Alles läuft **ohne API-Key**. Ein geskriptetes Mock-LLM beherrscht Tool Calling,
parallele Tool-Aufrufe und Structured Output, sodass sich jeder Workflow
deterministisch ausführen und testen lässt. Mit einer einzigen Zeile lässt es
sich gegen ein echtes Modell austauschen.

## Einstieg

| Sie möchten... | Dann lesen Sie |
|---|---|
| entscheiden, welches Pattern zu Ihrem Problem passt | [docs/de/README.md](docs/de/README.md): Zusammenfassung, Vergleich, Entscheidungshilfe |
| ein Pattern im Detail verstehen | [docs/de/patterns/](docs/de/patterns/): ein ausführlicher Bericht pro Pattern |
| ein Pattern in Ihrem eigenen Workflow nutzen | [docs/de/library.md](docs/de/library.md): die Bibliothek `agentpatterns` |
| den Anwendungsfall, die Daten und die Messungen sehen | [docs/de/use-case.md](docs/de/use-case.md) |
| die Quellen prüfen | [docs/de/sources.md](docs/de/sources.md) |

Die Dokumentation gibt es auf Deutsch unter `docs/de/` und auf Englisch unter `docs/`.

## Schnellstart

Voraussetzungen: [uv](https://docs.astral.sh/uv/) und Python 3.12 oder neuer
(uv installiert bei Bedarf eine passende Python-Version). Ein API-Key ist nicht
nötig.

```bash
git clone https://github.com/awa-projekt/multiagent_patterns.git
cd multiagent_patterns
uv sync
uv run email-demo                               # run all patterns on the inbox, print comparison
uv run email-demo --impl library                # same, built with the agentpatterns library
uv run email-demo -p supervisor -e E-1004 -v    # one pattern / one e-mail, show reply and calls per agent
uv run pytest                                   # 178 tests, offline (+11 opt-in real-model tests, skipped)
```

Ein Pattern aus der Bibliothek verwenden:

```python
from langchain.agents import create_agent
from agentpatterns import AgentSpec, create_supervisor

billing = create_agent(model, tools=[get_invoice, issue_refund], system_prompt="You handle billing...")
tech = create_agent(model, tools=[check_status, create_ticket], system_prompt="You handle tech issues...")

supervisor = create_supervisor(
    model,
    [AgentSpec("billing", "Invoices and refunds", billing), AgentSpec("tech", "Outages and bugs", tech)],
    system_prompt="You coordinate customer service.",
    response_format=Resolution,
)
supervisor.invoke({"messages": [("user", "I was charged twice and the dashboard is down")]})
```

Jede Factory liefert einen kompilierten LangGraph-Graphen mit demselben Vertrag
wie `create_agent` (rein: `messages`, raus: `messages` + `structured_response`).
Deshalb lassen sich Patterns verschachteln, als Subgraphen einbetten oder als
Tools bereitstellen.

## Projektstruktur

```
src/
├── agentpatterns/                 # reusable library: one module per pattern
│   ├── core.py                    #   AgentSpec, agent_as_tool, agent_as_node, ...
│   ├── single_agent.py  sequential.py  router.py  parallel.py  orchestrator.py
│   ├── supervisor.py  hierarchical.py  swarm.py  state_machine.py  skills.py
│   ├── evaluator_optimizer.py
│   ├── limits.py  messaging.py    #   loop limits / RunBudget, messages between agents
│   ├── mcp.py                     #   MCP tools per run (`mcp` extra)
│   └── testing.py                 #   ScriptedChatModel (mock LLM), UsageTracker, serve_mcp
└── email_assistant/               # the business use case
    ├── schemas.py  data.py  tools.py  prompts.py   # domain: e-mails, mock systems, tools, prompts
    ├── brain.py  mock_llm.py      #   deterministic "reasoning" behind the mock LLM
    ├── common.py                  #   workflow contract + deterministic dispatch step
    ├── native/                    #   hand-written LangGraph implementation per pattern
    ├── library_based/             #   the same workflows built with agentpatterns (+ composite.py)
    └── run.py                     #   CLI (email-demo)
tests/                             # library unit tests + end-to-end tests of all 23 workflows
docs/                              # summary, per-pattern reports, library docs, sources
```

## Mit einem echten Modell arbeiten

Die Workflows akzeptieren jedes LangChain-Chat-Modell. Die Beispiele nutzen
Anthropics Claude-Modelle über das Extra `anthropic`, das den Key aus
`ANTHROPIC_API_KEY` liest:

```bash
uv sync --extra anthropic
export ANTHROPIC_API_KEY=...
```

```python
from langchain.chat_models import init_chat_model
from email_assistant.native import PATTERNS

graph = PATTERNS["router"](init_chat_model("anthropic:claude-opus-5"))
```

Hinweise zu den Strategien für Structured Output stehen in
[docs/de/use-case.md](docs/de/use-case.md#auf-ein-echtes-modell-umstellen). Mit den
optionalen Smoke-Tests prüfen Sie Ihr Modell (ein Lauf macht etwa 40 bis 60
Modellaufrufe):

```bash
AGENTPATTERNS_TEST_MODEL=anthropic:claude-opus-5 uv run pytest -m real_model
```

Andere Anbieter funktionieren genauso: das passende LangChain-Paket installieren
(zum Beispiel `uv add langchain-openai`), den API-Key des Anbieters setzen und
dessen `init_chat_model`-Kennung übergeben.

## Entwicklung

```bash
uv run pytest              # offline test suite
uvx ruff check .           # lint (configured in pyproject.toml)
uvx ruff format src tests  # format (Python sources only)
```

Die Dependency-Gruppe `dev` (wird von `uv sync` installiert) enthält das
MCP-Extra, sodass die MCP-Tests ohne weitere Flags laufen.

## Lizenz

Lizenziert unter der [Apache License 2.0](LICENSE). Copyright 2026 awa-projekt.
