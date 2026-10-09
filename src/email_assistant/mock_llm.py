"""The mock LLM of the e-mail use case.

`create_mock_llm()` returns ONE chat model that is used for every role in
every pattern - exactly like a real deployment would pass one
`init_chat_model(...)` instance around. The model recognises its current role
from the system prompt ("You are the <role> of Acme Cloud ...") and simulates
what a capable LLM would do in that role: call tools (in parallel where it
makes sense), hand off, delegate, or return structured output.

The decisions themselves come from `brain.py` and only use what is visible in
the conversation (e-mail text, tool results, reports quoted in the prompt).

To use a real model instead:

    from langchain.chat_models import init_chat_model
    model = init_chat_model("anthropic:claude-opus-5")
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from functools import partial
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agentpatterns.testing import ModelTurn, ScriptedChatModel, ScriptError
from email_assistant import brain
from email_assistant.schemas import EmailClassification, TeamReport

_ROLE_RE = re.compile(r"You are the ([a-z ]+?) of Acme Cloud")
_EMAIL_BLOCK_RE = re.compile(r"<email>.*?</email>", re.S)
_INSTRUCTION_RE = re.compile(r"Call (\w+) with (\{.*\})")

TEAM_OF_DOMAIN = {
    "billing": "customer_care",
    "technical": "customer_care",
    "sales": "accounts",
    "relations": "accounts",
}


# ------------------------------------------------------------------ helpers
def _visible_text(turn: ModelTurn) -> str:
    return "\n".join(m.text for m in turn.messages if not isinstance(m, SystemMessage))


def _email(turn: ModelTurn) -> brain.ParsedEmail:
    email = brain.find_email(_visible_text(turn))
    if email is None:
        raise ScriptError("Mock LLM: no <email> found in the conversation.")
    return email


def _email_block(turn: ModelTurn) -> str:
    match = _EMAIL_BLOCK_RE.search(_visible_text(turn))
    return match.group(0) if match else ""


def _context(turn: ModelTurn) -> brain.Context:
    """Tool results of this conversation plus facts quoted as JSON in messages."""
    return brain.Context.from_calls([*_quoted_facts(turn), *turn.call_results()])


def _json_objects(turn: ModelTurn, *, include_tools: bool = True) -> list[Any]:
    types = (HumanMessage, ToolMessage) if include_tools else (HumanMessage,)
    objects: list[Any] = []
    for message in turn.messages:
        if isinstance(message, types):
            for obj in brain.extract_json_objects(message.text):
                objects.extend(obj if isinstance(obj, list) else [obj])
    return objects


def _quoted_facts(turn: ModelTurn) -> list[dict[str, Any]]:
    """Tool results that colleagues quoted as JSON lines (orchestrator / pipeline)."""
    facts = [o for o in _json_objects(turn, include_tools=False) if isinstance(o, dict) and "tool" in o]
    for message in turn.messages:  # results of subagents that ran as tools
        if isinstance(message, ToolMessage):
            facts += [o for o in brain.extract_json_objects(message.text) if isinstance(o, dict) and "tool" in o]
    return [f for f in facts if "result" in f]


def _reports(turn: ModelTurn) -> list[dict[str, Any]]:
    """Specialist / team reports visible in the conversation."""
    return [o for o in _json_objects(turn) if isinstance(o, dict) and "customer_reply" in o]


def _enum_values(schema: Any) -> list[str]:
    """All enum values in a JSON schema (e.g. allowed agent names)."""
    found: list[str] = []
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "enum" and isinstance(value, list):
                found += [v for v in value if isinstance(v, str)]
            else:
                found += _enum_values(value)
    elif isinstance(schema, list):
        for item in schema:
            found += _enum_values(item)
    return found


def _next_calls(
    turn: ModelTurn, email: brain.ParsedEmail, ctx: brain.Context, classification: EmailClassification
) -> list[tuple[str, dict]]:
    """Next tool calls across all domains of the e-mail (deduplicated, parallel)."""
    available = set(turn.tool_names)
    seen: set[tuple] = set()
    batch = []
    for domain in brain.domains_for(classification):
        for call in brain.pending_calls(domain, email, ctx, classification.priority, available):
            key = brain.call_key(*call)
            if key not in seen:
                seen.add(key)
                batch.append(call)
    return batch


def _task_text(domain: str, classification: EmailClassification, block: str) -> str:
    return f"Handle the {domain} part of this customer e-mail ({classification.summary})\n\n{block}"


def _delegate_call(turn: ModelTurn, keyword: str, task: str) -> tuple[str, dict[str, Any]] | None:
    """Tool call that delegates `task` to the agent/team whose name contains `keyword`."""
    candidates = [n for n in turn.tool_names if keyword in n and not n.startswith("transfer_to_") and n[0].islower()]
    if candidates:
        name = candidates[0]
        params = turn.tool_parameters(name)
        arg = (params.get("required") or list(params.get("properties", {})) or ["task"])[0]
        return name, {arg: task}
    for name in turn.tool_names:  # single dispatch tool: task(agent_name, description)
        params = turn.tool_parameters(name).get("properties", {})
        agents = [a for a in _enum_values(params) if keyword in a]
        if agents:
            enum_param = next(p for p, s in params.items() if agents[0] in _enum_values(s))
            text_param = next(p for p in params if p != enum_param)
            return name, {enum_param: agents[0], text_param: task}
    return None


def _has_feedback(turn: ModelTurn) -> bool:
    return any("feedback" in text.lower() for text in turn.human_messages[1:])


# ------------------------------------------------------------------ roles
def classifier(turn: ModelTurn) -> AIMessage:
    return turn.structured(brain.classify(_email(turn)))


def specialist(domain: str, turn: ModelTurn) -> AIMessage:
    email = _email(turn)
    classification = brain.classify(email)
    ctx = _context(turn)
    calls = brain.pending_calls(domain, email, ctx, classification.priority, set(turn.tool_names))
    if calls:
        return turn.call_many(*calls)
    if turn.tools_with_prefix("transfer_to_"):  # peer in a swarm
        return _swarm_next(turn, email, ctx, classification, me=domain)
    report = brain.report(domain, email, ctx)
    if turn.has_tool("SpecialistReport"):
        return turn.structured(report)
    return turn.say(report.model_dump_json())


def _swarm_next(
    turn: ModelTurn, email: brain.ParsedEmail, ctx: brain.Context, classification: EmailClassification, me: str | None
) -> AIMessage:
    """Hand off to the peer owning the next unfinished request, or finish."""
    for domain in brain.domains_for(classification):
        if domain == me or not brain.pending_calls(domain, email, ctx, classification.priority):
            continue
        tool = next((n for n in turn.tools_with_prefix("transfer_to_") if domain in n), None)
        if tool:
            params = turn.tool_parameters(tool).get("properties", {})
            note = (
                f"{(me or 'triage').title()} part is done. Please handle the {domain} request: {classification.summary}"
            )
            return turn.call(tool, **{p: note for p in params})
    return turn.structured(brain.resolve_from_context(email, ctx, classification))


def front_desk(turn: ModelTurn) -> AIMessage:
    email = _email(turn)
    return _swarm_next(turn, email, _context(turn), brain.classify(email), me=None)


def generalist(turn: ModelTurn, *, quick_first_draft: bool = False) -> AIMessage:
    """Single agent / responder / drafter: do everything, then resolve."""
    email = _email(turn)
    classification = brain.classify(email)
    ctx = _context(turn)
    calls = _next_calls(turn, email, ctx, classification)
    if calls:
        return turn.call_many(*calls)
    quick = quick_first_draft and not _has_feedback(turn)
    return turn.structured(brain.resolve_from_context(email, ctx, classification, quick=quick))


def router(turn: ModelTurn) -> AIMessage:
    email = _email(turn)
    classification = brain.classify(email)
    agents = _enum_values(turn.structured_tool_schema() or {})
    routes = []
    for domain in brain.domains_for(classification):
        agent = next((a for a in agents if domain in a), None)
        if agent:
            routes.append({"agent": agent, "task": _task_text(domain, classification, _email_block(turn))})
    return turn.structured({"routes": routes})


def reply_writer(turn: ModelTurn) -> AIMessage:
    email = _email(turn)
    classification = brain.classify(email)
    reports = _reports(turn)
    if reports:
        resolution = brain.compose_resolution(email, classification, reports)
    else:
        resolution = brain.resolve_from_context(email, _context(turn), classification)
    return turn.structured(resolution)


def coordinator(
    turn: ModelTurn, *, team_of: Callable[[str], str] | None = None, team_name: str | None = None
) -> AIMessage:
    """Supervisor (top level or team lead) that delegates via tools."""
    email = _email(turn)
    classification = brain.classify(email)
    reports = _reports(turn)
    delegated = any(turn.tool_calls_made())
    if classification.category != "spam" and not delegated:
        calls, seen = [], set()
        for domain in brain.domains_for(classification):
            keyword = team_of(domain) if team_of else domain
            if keyword in seen:
                continue
            call = _delegate_call(turn, keyword, _task_text(domain, classification, _email_block(turn)))
            if call:
                seen.add(keyword)
                calls.append(call)
        if calls:
            return turn.call_many(*calls)
    if team_name:  # team lead: merge member reports into a team report
        merged = TeamReport(
            team=team_name,
            findings=" ".join(r.get("findings", "") for r in reports),
            actions_taken=[a for r in reports for a in r.get("actions_taken", [])],
            customer_reply="\n\n".join(r["customer_reply"] for r in reports),
            needs_human=any(r.get("needs_human") for r in reports),
        )
        return turn.structured(merged) if turn.has_tool("TeamReport") else turn.say(merged.model_dump_json())
    return turn.structured(brain.compose_resolution(email, classification, reports))


def case_handler(turn: ModelTurn) -> AIMessage:
    """State-machine agent: behaviour depends on the step named in the prompt."""
    step = re.search(r"Current step: (\w+)", turn.system).group(1)
    email = _email(turn)
    classification = brain.classify(email)
    ctx = _context(turn)
    if step == "triage":
        if turn.has_tool("lookup_customer") and ("lookup_customer", email.sender) not in ctx.done:
            return turn.call("lookup_customer", email=email.sender)
        if turn.has_tool("record_triage"):
            return turn.call("record_triage", **classification.model_dump())
        target = "respond" if classification.category == "spam" else "resolve"
        return turn.call(f"go_to_{target}", reason=classification.summary)
    if step == "resolve":
        calls = _next_calls(turn, email, ctx, classification)
        if calls:
            return turn.call_many(*calls)
        if turn.has_tool("finish_resolution"):
            return turn.call("finish_resolution", summary="All requests handled.")
        return turn.call("go_to_respond", reason="All requests handled.")
    return turn.structured(brain.resolve_from_context(email, ctx, classification))


def skills_agent(turn: ModelTurn) -> AIMessage:
    email = _email(turn)
    classification = brain.classify(email)
    domains = brain.domains_for(classification)
    params = turn.tool_parameters("load_skill")
    param = (params.get("required") or ["skill_name"])[0]
    skills = _enum_values(params)
    loaded = {str(tc["args"].get(param)) for tc in turn.tool_calls_made("load_skill")}
    needed = [s for d in domains for s in skills if d in s and s not in loaded]
    if needed:
        return turn.call_many(*[("load_skill", {param: s}) for s in dict.fromkeys(needed)])
    return generalist(turn)


def sentiment_analyst(turn: ModelTurn) -> AIMessage:
    email = _email(turn)
    classification = brain.classify(email)
    signals = [w for w in (*brain.NEGATIVE_WORDS, "asap", "urgent") if w in email.text]
    return turn.structured(
        {"sentiment": classification.sentiment, "urgency": classification.priority, "signals": signals}
    )


def entity_extractor(turn: ModelTurn) -> AIMessage:
    email = _email(turn)
    return turn.structured(
        {
            "sender": email.sender,
            "customer_name": email.name,
            "invoice_ids": email.invoice_ids,
            "services": email.services,
            "seats": email.seats,
        }
    )


def compliance_screener(turn: ModelTurn) -> AIMessage:
    email = _email(turn)
    text = email.text
    flags = {
        "phishing": any(w in text for w in brain.SPAM_WORDS),
        "legal_threat": any(w in text for w in ("lawyer", "legal action")),
        "data_protection_request": any(w in text for w in ("gdpr", "art. 17", "deletion")),
    }
    notes = ", ".join(k for k, v in flags.items() if v) or "no findings"
    return turn.structured({**flags, "notes": notes})


def planner(turn: ModelTurn) -> AIMessage:
    """Orchestrator: plan research first, then operations, then stop."""
    email = _email(turn)
    classification = brain.classify(email)
    ctx = _context(turn)
    domains = brain.domains_for(classification)
    groups: dict[str, list[tuple[str, dict]]] = {}
    research = [c for d in domains for c in brain.info_calls(d, email) if brain.call_key(*c) not in ctx.done]
    for call in dict.fromkeys(json.dumps(c) for c in research):
        tool, args = json.loads(call)
        worker = "account_researcher" if tool in ("lookup_customer", "get_invoice") else "knowledge_researcher"
        groups.setdefault(worker, []).append((tool, args))
    if not groups:
        actions = [
            c
            for d in domains
            for c in brain.action_calls(d, email, ctx, classification.priority)
            if brain.call_key(*c) not in ctx.done
        ]
        if actions:
            groups["operations"] = actions
    workers = _enum_values(turn.structured_tool_schema() or {})
    tasks = [
        {
            "worker": next(w for w in workers if name in w),
            "instruction": "\n".join(f"Call {t} with {json.dumps(a)}" for t, a in calls),
        }
        for name, calls in groups.items()
    ]
    return turn.structured({"tasks": tasks})


def worker(turn: ModelTurn) -> AIMessage:
    """Executes 'Call <tool> with {...}' instructions, then reports raw results."""
    wanted = [(m[1], json.loads(m[2])) for text in turn.human_messages for m in _INSTRUCTION_RE.finditer(text)]
    done = brain.Context.from_calls(turn.call_results()).done
    pending = [(t, a) for t, a in wanted if brain.call_key(t, a) not in done and turn.has_tool(t)]
    if pending:
        return turn.call_many(*pending)
    return turn.say("\n".join(json.dumps(r) for r in turn.call_results()) or "Nothing to do.")


def reviewer(turn: ModelTurn) -> AIMessage:
    email = _email(turn)
    classification = brain.classify(email)
    drafts = [o for o in _json_objects(turn) if isinstance(o, dict) and "action" in o and "email_id" in o]
    issues = brain.review(email, drafts[-1] if drafts else {}, classification.sentiment)
    return turn.structured(
        {
            "passed": not issues,
            "score": max(0.0, round(1 - 0.2 * len(issues), 2)),
            "issues": issues,
            "feedback": " ".join(issues) or "The resolution meets all rules.",
        }
    )


def action_planner(turn: ModelTurn) -> AIMessage:
    email = _email(turn)
    classification = brain.classify(email)
    ctx = _context(turn)
    kinds = {
        "issue_refund": "refund",
        "create_ticket": "ticket",
        "create_sales_lead": "lead",
        "escalate_to_human": "escalation",
    }
    actions = [
        {"kind": kinds[tool], **args}
        for domain in brain.domains_for(classification)
        for tool, args in brain.action_calls(domain, email, ctx, classification.priority)
    ]
    rationale = f"{len(actions)} action(s) required by policy." if actions else "No action required."
    return turn.structured({"actions": actions, "rationale": rationale})


ROLES: dict[str, Callable[[ModelTurn], AIMessage]] = {
    "triage classifier": classifier,
    "billing specialist": partial(specialist, "billing"),
    "technical specialist": partial(specialist, "technical"),
    "sales specialist": partial(specialist, "sales"),
    "customer relations specialist": partial(specialist, "relations"),
    "customer service agent": generalist,
    "responder": generalist,
    "reply drafter": partial(generalist, quick_first_draft=True),
    "inbox router": router,
    "reply writer": reply_writer,
    "customer service supervisor": coordinator,
    "inbox manager": partial(coordinator, team_of=TEAM_OF_DOMAIN.get),
    "customer care team lead": partial(coordinator, team_name="customer_care"),
    "accounts team lead": partial(coordinator, team_name="accounts"),
    "front desk agent": front_desk,
    "case handler": case_handler,
    "service generalist": skills_agent,
    "sentiment analyst": sentiment_analyst,
    "entity extractor": entity_extractor,
    "compliance screener": compliance_screener,
    "case planner": planner,
    "account researcher": worker,
    "knowledge researcher": worker,
    "operations agent": worker,
    "quality reviewer": reviewer,
    "action planner": action_planner,
}


def email_policy(turn: ModelTurn) -> AIMessage:
    match = _ROLE_RE.search(turn.system)
    if not match or match.group(1) not in ROLES:
        raise ScriptError(f"Mock LLM has no behaviour for system prompt: {turn.system[:120]!r}")
    return ROLES[match.group(1)](turn)


def create_mock_llm(*, native_structured_output: bool = False) -> ScriptedChatModel:
    """One mock chat model for all roles of the e-mail use case.

    With `native_structured_output=True` the mock declares native structured
    output support (like current Claude / OpenAI models), so `create_agent` uses
    `ProviderStrategy` and the library uses `method="json_schema"`.
    """
    profile = {"structured_output": True} if native_structured_output else None
    return ScriptedChatModel(policy=email_policy, model_name="mock-email-llm", profile=profile)
