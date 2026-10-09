# Use case: customer-service inbox automation

**Acme Cloud** is a fictional B2B SaaS company. Its support inbox receives
billing questions, outage reports, sales inquiries, spam and sometimes legal
or GDPR matters. Every incoming e-mail must be

1. **categorized**: category, further intents, priority, sentiment, needs a human?
2. **processed**: look up the customer and facts, then act by policy (refund, ticket, lead, escalation)
3. **answered**: write a reply, send it, or escalate to a human.

We implemented this identical task with each of the eleven patterns, so the
patterns can be compared directly.

## Inbox (workflow input)

`src/email_assistant/data.py` contains six e-mails that together exercise every branch:

| ID | From | Content | Expected |
|---|---|---|---|
| E-1001 | Anna Schmidt (Contoso, gold) | Charged twice for invoice INV-2026-0815 | billing, refund 348 EUR, reply |
| E-1002 | Mark Jones (Fabrikam) | Dashboard shows HTTP 503, team blocked | technical, urgent, ticket linked to incident INC-7781, reply |
| E-1003 | Priya Natarajan (Northwind, unknown sender) | Enterprise pricing for 50 seats + demo | sales, pricing with 15% volume discount, new lead, reply |
| E-1004 | Jordan Doe (Tailspin, platinum) | Invoice bills 60 instead of 50 seats **and** CSV export crashes | **multi-intent**: billing correction ticket + technical known issue, one reply |
| E-1005 | "Prize Department" | "You have WON an iPhone, click here" | spam, ignore |
| E-1006 | Anna Schmidt | GDPR Art. 17 deletion request + threat of legal action | needs a human, escalate to the DPO, neutral acknowledgment |

## Mock data sources (tools)

`src/email_assistant/tools.py` provides LangChain `@tool`s over in-memory mock systems:

| Tool | Kind | Mocked system |
|---|---|---|
| `lookup_customer(email)` | read | CRM |
| `get_invoice(invoice_id)` | read | billing system (payments, seats) |
| `search_knowledge_base(query)` | read | policies and troubleshooting articles (KB-101 ... KB-910) |
| `check_service_status(service)` | read | status page (dashboard incident INC-7781) |
| `get_plan_pricing(plan, seats)` | read | price calculator |
| `issue_refund(invoice_id, amount, reason)` | **write** | payments (enforces: only paid invoices, only the overpaid amount, at most 500 EUR, idempotent) |
| `create_ticket(customer_id, queue, summary, priority)` | **write** | ticketing |
| `create_sales_lead(...)` | **write** | CRM leads (assigns an account executive) |
| `escalate_to_human(customer_email, team, reason, summary)` | **write** | escalation queue (DPO, team lead, finance) |

Side effects land in `data.BACKEND` (refunds, tickets, leads, escalations,
outbox), which tests inspect. Record IDs are derived from the content, so
parallel branches produce the same IDs regardless of timing.

## Contract of every workflow

```python
input  = {"email": Email}
output = {"resolution": EmailResolution, "delivery": Delivery}
```

`EmailResolution` (structured output, `schemas.py`) contains category,
priority, action (`reply` / `escalate` / `ignore`), actions taken with
reference IDs, the reply subject and body, the escalation reason and an
internal note. Every workflow ends with the deterministic **`dispatch`** node,
which sends the reply to the outbox and records escalations. The model never
sends anything itself.

```
START -> <pattern> -> dispatch -> END
```

## The mock LLM

We don't call an API. Instead, `agentpatterns.testing.ScriptedChatModel`
behaves like a real tool-calling chat model:

- `bind_tools()` works, so the model "sees" the tools of each call.
- It can return **parallel tool calls** and **structured output**, both via
  tool calling and via native JSON output. `create_agent(response_format=...)`
  and `with_structured_output()` therefore work unchanged.
  `create_mock_llm(native_structured_output=True)` declares native support like
  current Claude models do; the tests run every workflow both ways.
- It refuses what a real model cannot do: calling unbound tools, or answering
  in text when a tool call is forced.
- It reports estimated token usage, so `UsageTracker` can compare patterns.

For every call it runs a *policy*. `email_assistant/mock_llm.py` contains
**one** policy for the whole use case. Like a real model, it infers its role
from the system prompt ("You are the billing specialist of Acme Cloud ...")
and then acts: call tools, delegate, hand off, plan, or answer. The actual
"reasoning" lives in `brain.py`: keyword classification, the per-domain
procedure (gather information, act, report) and reply composition. The mock
only uses information visible in the conversation, never the databases
directly. So a pattern that fails to pass information along really does fail.

One consequence is that **every pattern yields exactly the same resolution for
every e-mail**, which the tests assert. That makes the mock ideal for testing
orchestration and counting calls. It cannot show quality differences between
patterns: real models degrade differently when they have too many tools, too
much context, or a lossy handoff. The reports discuss those differences based
on the research.

The evaluator-optimizer pattern is a deliberate exception: the mock drafter
writes a sloppy first draft (no name, no reference numbers, no signature) so
that the review loop actually iterates once.

## Running it

```bash
uv sync
uv run email-demo                          # comparison of all native workflows
uv run email-demo --impl library           # the same built with agentpatterns (+ composite)
uv run email-demo -p supervisor -e E-1004 -v   # replies + model calls per agent
uv run email-demo --mermaid router         # graph as a Mermaid diagram
uv run email-demo --digest                 # map-reduce over the inbox
uv run pytest                              # 178 tests (+11 opt-in real-model tests)
```

### Switching to a real model

Every workflow takes the chat model as a parameter. For Claude, install the
`anthropic` extra (`uv sync --extra anthropic`) and set `ANTHROPIC_API_KEY`:

```python
from langchain.chat_models import init_chat_model
from email_assistant.native import PATTERNS

model = init_chat_model("anthropic:claude-opus-5")
graph = PATTERNS["supervisor"](model)
graph.invoke({"email": email})
```

With a real model, `response_format=SomeSchema` automatically uses the
provider's native structured output (`ProviderStrategy`) when the model's
profile declares support, which every current Claude model does.

The `with_structured_output()` calls need more care. For Claude its default is
forced tool calling. Claude Opus 5.5, Claude Sonnet 5.5 and Claude Fable 5.1
reject forced `tool_choice`, and so does any Claude model with extended thinking
enabled.
`langchain-anthropic` then sends an unforced tool call and raises
`OutputParserException` when the model answers in text instead.

- The **library-based** workflows handle this: their internal structured calls
  default to `structured_output_method="auto"`, which uses native structured
  output (`method="json_schema"`) where the profile declares it.
- The **hand-written** workflows in `email_assistant/native/` call
  `model.with_structured_output(Schema)` with the provider default, which is
  fine on Claude Opus 5 and Sonnet 5. For Opus 5.5, Sonnet 5.5 or Fable 5.1,
  add `method="json_schema"` to those calls.

Before switching, run the smoke tests against the model:
`AGENTPATTERNS_TEST_MODEL=anthropic:claude-opus-5 uv run pytest -m real_model`
(see [library.md](library.md#testing)).

## Measured comparison

`uv run email-demo` (native implementations, whole inbox of 6 e-mails):

| Pattern | Correct | Model calls (total) | E-1001 (1 intent) | E-1004 (2 intents) | E-1005 (spam) | Tool calls | Est. input tokens |
|---|---|---|---|---|---|---|---|
| single_agent | 6/6 | 16 | 3 | 3 | 1 | 22 | 26,794 |
| sequential | 6/6 | 21 | 4 | 4 | 1 | 21 | 16,962 |
| router | 6/6 | 30 | 5 | 8 | 2 | 23 | 25,724 |
| parallel | 6/6 | 39 | 7 | 7 | 4 | 22 | 33,874 |
| orchestrator | 6/6 | 52 | 10 | 10 | 2 | 22 | 30,489 |
| supervisor | 6/6 | 29 | 5 | 8 | 1 | 29 | 30,579 |
| hierarchical | 6/6 | 39 | 7 | 10 | 1 | 34 | 35,431 |
| swarm | 6/6 | 24 | 4 | 7 | 1 | 28 | 33,026 |
| state_machine | 6/6 | 33 | 6 | 6 | 3 | 34 | 25,933 |
| skills | 6/6 | 21 | 4 | 4 | 1 | 28 | 27,598 |
| evaluator_optimizer | 6/6 | 32 | 6 | 6 | 2 | 22 | 42,182 |

"Tool calls" include delegation, handoff, transition and skill-loading tools.
Tokens are estimates (about 4 characters per token over prompt, tool schemas
and history); use them to compare patterns with each other, not as absolute
cost. The library-based implementations make exactly the same model calls;
their token counts differ slightly because the library's default prompts add
agent catalogs.

**What the numbers show**

- **Fixed-flow patterns cost the same for every e-mail** (sequential, parallel,
  orchestrator, state machine). Dynamic patterns pay per intent: router,
  supervisor, hierarchical and swarm need 3 extra calls for the second intent
  of E-1004.
- **Agents that batch tool calls are cheap in calls.** The single agent and
  skills handle two intents in 3-4 calls because their tool calls run in
  parallel. The price is one growing context.
- **Every coordination layer costs calls.** Supervisor adds 2 calls (delegate
  + compose) over the specialist's own 3, and hierarchical adds 2 more per level.
- **Early exits matter.** Gates (sequential, parallel, supervisor) handle spam
  in 1 call. The state machine still walks its states (3 calls) and the
  parallel fan-out pays for all four analysts.
- **Shared history is expensive.** The swarm forwards the full conversation,
  so E-1004 is the most token-intensive run (about 11k tokens) although it
  needs fewer calls than the supervisor.
- **The pipeline uses the fewest tokens.** Its single-shot LLM calls carry no
  growing tool-call history, only the facts each step needs.
