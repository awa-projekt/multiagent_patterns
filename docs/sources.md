English | [Deutsch](de/sources.md)

# Research sources

We read the primary sources in September 2026. The LangChain and LangGraph
docs were read as raw markdown, and we verified the APIs against the installed
packages (LangGraph 1.2.12, LangChain 1.4.2, `langchain-core` 1.6.5).

## Framework documentation (LangChain / LangGraph)

- Multi-agent overview (patterns, choosing a pattern, performance comparison): https://docs.langchain.com/oss/python/langchain/multi-agent
- Subagents: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents
- Handoffs: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs, with the customer-support tutorial
- Router: https://docs.langchain.com/oss/python/langchain/multi-agent/router, with the knowledge-base tutorial
- Skills: https://docs.langchain.com/oss/python/langchain/multi-agent/skills, with the SQL-assistant tutorial
- Custom workflow: https://docs.langchain.com/oss/python/langchain/multi-agent/custom-workflow
- Migrate from langgraph-supervisor: https://docs.langchain.com/oss/python/migrate/langgraph-supervisor
- LangGraph v1 migration (`create_react_agent` → `create_agent`): https://docs.langchain.com/oss/python/migrate/langgraph-v1
- Structured output (`ToolStrategy`, `ProviderStrategy`): https://docs.langchain.com/oss/python/langchain/structured-output
- Custom middleware (`wrap_model_call`, custom state, dynamic tools): https://docs.langchain.com/oss/python/langchain/middleware/custom
- Unit testing (fake chat models): https://docs.langchain.com/oss/python/langchain/test/unit-testing
- Workflows and agents (prompt chaining, parallelization, routing, orchestrator-worker, evaluator-optimizer): https://docs.langchain.com/oss/python/langgraph/workflows-agents
- Graph API (`Send`, `Command`, `Command.PARENT`, reducers): https://docs.langchain.com/oss/python/langgraph/graph-api
- Subgraphs (communication, persistence modes): https://docs.langchain.com/oss/python/langgraph/use-subgraphs
- Repositories: langgraph-supervisor-py (archived) https://github.com/langchain-ai/langgraph-supervisor-py, langgraph-swarm-py https://github.com/langchain-ai/langgraph-swarm-py

**API status we verified:**

- `langchain.agents.create_agent` replaces `langgraph.prebuilt.create_react_agent`.
- `langgraph-supervisor` is no longer maintained. The docs recommend the
  subagents-as-tools pattern.
- `langgraph-swarm` still works, but the docs no longer mention it; handoffs
  are built with `Command(goto=..., graph=Command.PARENT)` or single-agent
  middleware.
- `langchain-mcp-adapters` was archived in September 2026. `langchain.mcp`
  (beta, `langchain[mcp]` ≥ 1.4, built on FastMCP 4 and `mcp` 2.x) replaces it:
  https://docs.langchain.com/oss/python/migrate/langchain-mcp-adapters
- LangGraph 1.2 node timeouts (`add_node(..., timeout=...)`, `TimeoutPolicy`)
  only work for async nodes in async runs; `invoke` refuses a node that has one.

## Vendor and practitioner guidance

- Anthropic, *Building effective agents* (Dec 2024): https://www.anthropic.com/research/building-effective-agents
- Anthropic, *How we built our multi-agent research system* (Jun 2025): https://www.anthropic.com/engineering/multi-agent-research-system
- Anthropic, *Effective context engineering for AI agents* (Sep 2025): https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Anthropic, *Equipping agents for the real world with Agent Skills* (Oct 2025): https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- OpenAI, *A practical guide to building agents*: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
- OpenAI Agents SDK, handoffs and multi-agent orchestration: https://openai.github.io/openai-agents-python/handoffs/, https://openai.github.io/openai-agents-python/multi_agent/
- Microsoft, *AI agent orchestration patterns* (sequential, concurrent, group chat, handoff, magentic): https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns
- Google, *Developer's guide to multi-agent patterns in ADK* (Dec 2025): https://developers.googleblog.com/developers-guide-to-multi-agent-patterns-in-adk/
- Cognition, *Don't build multi-agents* (Jun 2025): https://cognition.com/blog/dont-build-multi-agents
- Cognition, *Multi-agents: what's actually working* (Apr 2026): https://cognition.com/blog/multi-agents-working
- LangChain, *How and when to build multi-agent systems* (Jun 2025): https://www.langchain.com/blog/how-and-when-to-build-multi-agent-systems
- LangChain, *Benchmarking multi-agent architectures* (Jun 2025): https://www.langchain.com/blog/benchmarking-multi-agent-architectures
- LangChain, *Choosing the right multi-agent architecture* (Jan 2026): https://www.langchain.com/blog/choosing-the-right-multi-agent-architecture
- LangChain, *Plan-and-execute agents* (Feb 2024): https://www.langchain.com/blog/planning-agents

## Papers

- Google Research, *Towards a science of scaling agent systems* (2025/26): https://arxiv.org/abs/2512.08296
- Cemri et al., *Why do multi-agent LLM systems fail?* (MAST taxonomy): https://arxiv.org/abs/2503.13657
- Du et al., *Improving factuality and reasoning through multiagent debate*: https://arxiv.org/abs/2305.14325
- Shinn et al., *Reflexion*: https://arxiv.org/abs/2303.11366
- Madaan et al., *Self-Refine*: https://arxiv.org/abs/2303.17651
- Salemi et al., *LLM-based multi-agent blackboard system*: https://arxiv.org/abs/2510.01285

## Quantitative claims used in the reports

| Claim | Source |
|---|---|
| Multi-agent research system (Opus lead + Sonnet subagents) outperformed a single agent by 90.2% on an internal research eval | Anthropic, multi-agent research system |
| Agents use about 4× the tokens of chat; multi-agent systems about 15×; token usage explains 80% of performance variance (BrowseComp) | Anthropic, multi-agent research system |
| Parallel subagents and tool calls cut research time by up to 90% | Anthropic, multi-agent research system |
| One-shot request: subagents 4 calls, handoffs / skills / router 3 each | LangChain docs, multi-agent overview |
| Repeat request: subagents 8, handoffs 5, skills 5, router 6 calls (stateful patterns save 40-50%) | LangChain docs, multi-agent overview |
| Multi-domain: subagents 5 calls / about 9k tokens, handoffs 7+ / about 14k, skills 3 / about 15k, router 5 / about 9k | LangChain docs, multi-agent overview |
| Swarm slightly outperforms supervisor; single agent degrades sharply with 2+ distractor domains; supervisor fixes gave nearly 50% improvement | LangChain benchmark (tau-bench retail + distractor domains) |
| Some agents handle 15+ distinct tools, others struggle with fewer than 10 overlapping ones | OpenAI practical guide |
| Centralized coordination contains error amplification (4.4× vs 17.2× for independent agents); +80.9% on parallelizable Finance-Agent; −39% to −70% on sequential PlanCraft | Google Research scaling study |
| Failure split: system design 41.8%, inter-agent misalignment 36.9%, task verification 21.3% | MAST (Cemri et al.) |
| Self-Refine about 20% absolute improvement; Reflexion 91% pass@1 on HumanEval | Madaan et al.; Shinn et al. |
| Review loops find about 2 bugs per PR, about 58% severe; reviewers work best without shared context | Cognition 2026 |
| LLMCompiler up to 3.6× speedup | LLMCompiler paper, via LangChain blog |
| Blackboard design: 13-57% relative improvement | Salemi et al. |

The LangChain docs' performance tables are illustrative worked examples, not
benchmarks. The numbers in our own [comparison](use-case.md#measured-comparison)
come from structural call counts with a deterministic mock and say nothing about
answer quality.
