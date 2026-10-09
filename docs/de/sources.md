[English](../sources.md) | Deutsch

# Recherchequellen

Wir haben die Primärquellen im September 2026 gelesen. Die Dokumentation von
LangChain und LangGraph haben wir als rohes Markdown gelesen und die APIs gegen
die installierten Pakete geprüft (LangGraph 1.2.12, LangChain 1.4.2,
`langchain-core` 1.6.5).

## Framework-Dokumentation (LangChain / LangGraph)

- Multi-Agent-Überblick (Patterns, Wahl eines Patterns, Performance-Vergleich): https://docs.langchain.com/oss/python/langchain/multi-agent
- Subagents: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents
- Handoffs: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs, mit dem Customer-Support-Tutorial
- Router: https://docs.langchain.com/oss/python/langchain/multi-agent/router, mit dem Knowledge-Base-Tutorial
- Skills: https://docs.langchain.com/oss/python/langchain/multi-agent/skills, mit dem SQL-Assistant-Tutorial
- Custom workflow: https://docs.langchain.com/oss/python/langchain/multi-agent/custom-workflow
- Migrate from langgraph-supervisor: https://docs.langchain.com/oss/python/migrate/langgraph-supervisor
- Migration auf LangGraph v1 (`create_react_agent` → `create_agent`): https://docs.langchain.com/oss/python/migrate/langgraph-v1
- Structured output (`ToolStrategy`, `ProviderStrategy`): https://docs.langchain.com/oss/python/langchain/structured-output
- Custom middleware (`wrap_model_call`, eigener State, dynamische Tools): https://docs.langchain.com/oss/python/langchain/middleware/custom
- Unit testing (Fake-Chat-Modelle): https://docs.langchain.com/oss/python/langchain/test/unit-testing
- Workflows and agents (Prompt Chaining, Parallelisierung, Routing, Orchestrator-Worker, Evaluator-Optimizer): https://docs.langchain.com/oss/python/langgraph/workflows-agents
- Graph API (`Send`, `Command`, `Command.PARENT`, Reducer): https://docs.langchain.com/oss/python/langgraph/graph-api
- Subgraphs (Kommunikation, Persistenzmodi): https://docs.langchain.com/oss/python/langgraph/use-subgraphs
- Repositories: langgraph-supervisor-py (archiviert) https://github.com/langchain-ai/langgraph-supervisor-py, langgraph-swarm-py https://github.com/langchain-ai/langgraph-swarm-py

**Von uns geprüfter API-Stand:**

- `langchain.agents.create_agent` ersetzt `langgraph.prebuilt.create_react_agent`.
- `langgraph-supervisor` wird nicht mehr gepflegt. Die Dokumentation empfiehlt
  das Pattern „Subagenten als Tools“.
- `langgraph-swarm` funktioniert weiterhin, wird in der Dokumentation aber nicht
  mehr erwähnt. Handoffs werden mit `Command(goto=..., graph=Command.PARENT)`
  oder mit Middleware in einem einzelnen Agenten gebaut.
- `langchain-mcp-adapters` wurde im September 2026 archiviert. `langchain.mcp`
  (Beta, `langchain[mcp]` ≥ 1.4, basiert auf FastMCP 4 und `mcp` 2.x) ersetzt es:
  https://docs.langchain.com/oss/python/migrate/langchain-mcp-adapters
- Knoten-Timeouts in LangGraph 1.2 (`add_node(..., timeout=...)`, `TimeoutPolicy`)
  funktionieren nur für asynchrone Knoten in asynchronen Läufen; `invoke` lehnt
  einen Knoten mit Timeout ab.

## Leitfäden von Anbietern und Praktikern

- Anthropic, *Building effective agents* (Dez. 2024): https://www.anthropic.com/research/building-effective-agents
- Anthropic, *How we built our multi-agent research system* (Juni 2025): https://www.anthropic.com/engineering/multi-agent-research-system
- Anthropic, *Effective context engineering for AI agents* (Sept. 2025): https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Anthropic, *Equipping agents for the real world with Agent Skills* (Okt. 2025): https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- OpenAI, *A practical guide to building agents*: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
- OpenAI Agents SDK, Handoffs und Multi-Agent-Orchestrierung: https://openai.github.io/openai-agents-python/handoffs/, https://openai.github.io/openai-agents-python/multi_agent/
- Microsoft, *AI agent orchestration patterns* (sequenziell, nebenläufig, Group Chat, Handoff, Magentic): https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns
- Google, *Developer's guide to multi-agent patterns in ADK* (Dez. 2025): https://developers.googleblog.com/developers-guide-to-multi-agent-patterns-in-adk/
- Cognition, *Don't build multi-agents* (Juni 2025): https://cognition.com/blog/dont-build-multi-agents
- Cognition, *Multi-agents: what's actually working* (Apr. 2026): https://cognition.com/blog/multi-agents-working
- LangChain, *How and when to build multi-agent systems* (Juni 2025): https://www.langchain.com/blog/how-and-when-to-build-multi-agent-systems
- LangChain, *Benchmarking multi-agent architectures* (Juni 2025): https://www.langchain.com/blog/benchmarking-multi-agent-architectures
- LangChain, *Choosing the right multi-agent architecture* (Jan. 2026): https://www.langchain.com/blog/choosing-the-right-multi-agent-architecture
- LangChain, *Plan-and-execute agents* (Feb. 2024): https://www.langchain.com/blog/planning-agents

## Papers

- Google Research, *Towards a science of scaling agent systems* (2025/26): https://arxiv.org/abs/2512.08296
- Cemri et al., *Why do multi-agent LLM systems fail?* (MAST-Taxonomie): https://arxiv.org/abs/2503.13657
- Du et al., *Improving factuality and reasoning through multiagent debate*: https://arxiv.org/abs/2305.14325
- Shinn et al., *Reflexion*: https://arxiv.org/abs/2303.11366
- Madaan et al., *Self-Refine*: https://arxiv.org/abs/2303.17651
- Salemi et al., *LLM-based multi-agent blackboard system*: https://arxiv.org/abs/2510.01285

## In den Berichten verwendete quantitative Aussagen

| Aussage | Quelle |
|---|---|
| Ein Multi-Agent-Recherchesystem (Opus als Lead + Sonnet-Subagenten) übertraf einen einzelnen Agenten in einer internen Recherche-Evaluation um 90.2% | Anthropic, multi-agent research system |
| Agenten verbrauchen etwa 4× so viele Token wie Chat, Multi-Agent-Systeme etwa 15×; der Token-Verbrauch erklärt 80% der Varianz in der Leistung (BrowseComp) | Anthropic, multi-agent research system |
| Parallele Subagenten und Tool-Aufrufe verkürzen die Recherchezeit um bis zu 90% | Anthropic, multi-agent research system |
| Einmalige Anfrage: Subagents 4 Aufrufe, Handoffs / Skills / Router je 3 | LangChain-Dokumentation, Multi-Agent-Überblick |
| Wiederholte Anfrage: Subagents 8, Handoffs 5, Skills 5, Router 6 Aufrufe (zustandsbehaftete Patterns sparen 40-50%) | LangChain-Dokumentation, Multi-Agent-Überblick |
| Mehrere Domänen: Subagents 5 Aufrufe / etwa 9k Token, Handoffs 7+ / etwa 14k, Skills 3 / etwa 15k, Router 5 / etwa 9k | LangChain-Dokumentation, Multi-Agent-Überblick |
| Swarm schneidet etwas besser ab als Supervisor; ein einzelner Agent verschlechtert sich ab 2 Ablenkungsdomänen deutlich; Korrekturen am Supervisor brachten fast 50% Verbesserung | LangChain-Benchmark (tau-bench retail + Ablenkungsdomänen) |
| Manche Agenten kommen mit 15+ verschiedenen Tools zurecht, andere scheitern schon an weniger als 10 sich überschneidenden | OpenAI, Practical Guide |
| Zentrale Koordination dämmt die Fehlerverstärkung ein (4.4× gegenüber 17.2× bei unabhängigen Agenten); +80.9% beim parallelisierbaren Finance-Agent; −39% bis −70% beim sequenziellen PlanCraft | Skalierungsstudie von Google Research |
| Verteilung der Fehlerursachen: Systemdesign 41.8%, Fehlabstimmung zwischen Agenten 36.9%, Aufgabenverifikation 21.3% | MAST (Cemri et al.) |
| Self-Refine etwa 20% absolute Verbesserung; Reflexion 91% pass@1 auf HumanEval | Madaan et al.; Shinn et al. |
| Review-Schleifen finden etwa 2 Bugs pro PR, davon etwa 58% schwerwiegend; Reviewer arbeiten am besten ohne gemeinsamen Kontext | Cognition 2026 |
| LLMCompiler bis zu 3.6× schneller | LLMCompiler-Paper, über den LangChain-Blog |
| Blackboard-Design: 13-57% relative Verbesserung | Salemi et al. |

Die Performance-Tabellen der LangChain-Dokumentation sind illustrative
Rechenbeispiele, keine Benchmarks. Die Zahlen in unserem eigenen
[Vergleich](use-case.md#gemessener-vergleich) beruhen auf strukturellen
Aufrufzählungen mit einem deterministischen Mock und sagen nichts über die
Antwortqualität aus.
