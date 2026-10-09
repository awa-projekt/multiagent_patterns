[Deutsch](README.md) | English

# Multi-agent patterns with LangGraph

Research, reference implementations and a reusable library for multi-agent
LLM architectures, built on **LangGraph 1.2 / LangChain 1.4**.

We took one business use case (**categorizing, processing and answering
customer e-mails**) and implemented it with eleven patterns:

single agent · sequential pipeline · router · parallelization (sectioning /
voting / map-reduce) · orchestrator-workers · supervisor (subagents as tools) ·
hierarchical teams · swarm (handoffs) · state machine (handoffs via
middleware) · skills (progressive disclosure) · evaluator-optimizer

Everything runs **without an API key**. A scripted mock LLM supports tool
calling, parallel tool calls and structured output, so every workflow can be
executed and tested deterministically. It can be swapped for a real model with
one line.

## Where to start

| You want to... | Go to |
|---|---|
| Decide which pattern fits your problem | [docs/README.md](docs/README.md): summary, comparison, decision guide |
| Understand one pattern in depth | [docs/patterns/](docs/patterns/): one detailed report per pattern |
| Use a pattern in your own workflow | [docs/library.md](docs/library.md): the `agentpatterns` library |
| See the use case, data and measurements | [docs/use-case.md](docs/use-case.md) |
| Check sources | [docs/sources.md](docs/sources.md) |

## Quickstart

Prerequisites: [uv](https://docs.astral.sh/uv/) and Python 3.12 or newer
(uv installs a matching Python if none is found). No API key is needed.

```bash
git clone https://github.com/awa-projekt/multiagent_patterns.git
cd multiagent_patterns
uv sync
uv run email-demo                               # run all patterns on the inbox, print comparison
uv run email-demo --impl library                # same, built with the agentpatterns library
uv run email-demo -p supervisor -e E-1004 -v    # one pattern / one e-mail, show reply and calls per agent
uv run pytest                                   # 178 tests, offline (+11 opt-in real-model tests, skipped)
```

Use a pattern from the library:

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

Every factory returns a compiled LangGraph graph with the same contract as
`create_agent` (`messages` in, `messages` + `structured_response` out). You can
therefore nest patterns, embed them as subgraphs or expose them as tools.

## Project layout

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

## Using a real model

The workflows take any LangChain chat model. The examples use Anthropic's
Claude models through the `anthropic` extra, which reads your key from
`ANTHROPIC_API_KEY`:

```bash
uv sync --extra anthropic
export ANTHROPIC_API_KEY=...
```

```python
from langchain.chat_models import init_chat_model
from email_assistant.native import PATTERNS

graph = PATTERNS["router"](init_chat_model("anthropic:claude-opus-5"))
```

See [docs/use-case.md](docs/use-case.md#switching-to-a-real-model) for notes on
structured-output strategies, and run the opt-in smoke tests against your model
(they make roughly 40-60 model calls):

```bash
AGENTPATTERNS_TEST_MODEL=anthropic:claude-opus-5 uv run pytest -m real_model
```

Other providers work the same way: install their LangChain package (for
example `uv add langchain-openai`), set their API key variable and pass their
`init_chat_model` identifier.

## Development

```bash
uv run pytest              # offline test suite
uvx ruff check .           # lint (configured in pyproject.toml)
uvx ruff format src tests  # format (Python sources only)
```

The `dev` dependency group (installed by `uv sync`) includes the MCP extra, so
the MCP tests run without extra flags.

## License

Licensed under the [Apache License 2.0](LICENSE). Copyright 2026 awa-projekt.
