"""Deterministic "reasoning" used by the mock LLM.

This module is what a real LLM would contribute: understanding e-mails,
deciding which tools to call, and writing replies. It is written as plain,
testable functions so that every pattern (single agent, router, supervisor,
swarm, ...) behaves consistently while the *orchestration* differs.

Rule: these functions only use information the model can actually see - the
e-mail text in the prompt and tool results in the conversation. They never
peek into the mock databases directly.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from email_assistant.schemas import (
    CATEGORY_TO_DOMAIN,
    EmailClassification,
    EmailResolution,
    SpecialistReport,
)

STATUS_PAGE = "https://status.acme-cloud.example"
SIGNATURE = "Best regards,\nAcme Cloud Customer Service"

# ------------------------------------------------------------------ parsing
_EMAIL_RE = re.compile(
    r"Email-ID:\s*(?P<id>\S+)\s*\n"
    r"From:\s*(?P<name>.*?)\s*<(?P<sender>[^>]+)>\s*\n"
    r"Subject:\s*(?P<subject>.*?)\s*\n"
    r"Received:\s*(?P<received>.*?)\s*\n\n"
    r"(?P<body>.*?)(?:\n</email>|\Z)",
    re.S,
)
_INVOICE_RE = re.compile(r"INV-\d{4}-\d{4}")
_SEATS_RE = re.compile(r"(\d+)\s*(?:seats|users|licen[cs]es)", re.I)
_REF_RE = re.compile(r"\b(?:RF|TCK|LEAD|ESC|INC)-\d{4}\b")


@dataclass
class ParsedEmail:
    id: str
    sender: str
    name: str
    subject: str
    body: str

    @property
    def first_name(self) -> str:
        return self.name.split()[0] if self.name else "there"

    @property
    def text(self) -> str:
        return f"{self.subject}\n{self.body}".lower()

    @property
    def invoice_ids(self) -> list[str]:
        return list(dict.fromkeys(_INVOICE_RE.findall(f"{self.subject} {self.body}")))

    @property
    def seats(self) -> int | None:
        match = _SEATS_RE.search(self.body)
        return int(match.group(1)) if match else None

    @property
    def services(self) -> list[str]:
        mapping = {"dashboard": ["dashboard", "log in", "login"], "export": ["export"], "api": [" api "]}
        return [svc for svc, words in mapping.items() if any(w in f" {self.text} " for w in words)]

    @property
    def company(self) -> str:
        """Best guess of the sender's company (signature line or e-mail domain)."""
        for line in reversed(self.body.strip().splitlines()):
            if len(line) < 40 and any(s in line for s in ("GmbH", "Inc", "Ltd", "Traders", "Toys", "AG")):
                return line.strip()
        match = re.search(r"for ([A-Z][\w&]+(?: [A-Z][\w&]+)*)", self.body)
        if match:
            return match.group(1)
        return self.sender.split("@")[1].split(".")[0].title()


def find_email(text: str) -> ParsedEmail | None:
    """Find the (first) rendered e-mail inside a prompt."""
    match = _EMAIL_RE.search(text)
    if not match:
        return None
    return ParsedEmail(
        id=match["id"],
        sender=match["sender"].strip().lower(),
        name=match["name"].strip(),
        subject=match["subject"].strip(),
        body=match["body"].strip(),
    )


def extract_json_objects(text: str) -> list[Any]:
    """All top-level JSON objects/arrays embedded in a text (e.g. reports in a prompt)."""
    decoder = json.JSONDecoder()
    found, index = [], 0
    while index < len(text):
        next_obj = min((i for i in (text.find("{", index), text.find("[", index)) if i != -1), default=-1)
        if next_obj == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, next_obj)
        except json.JSONDecodeError:
            index = next_obj + 1
            continue
        found.append(obj)
        index = end
    return found


# ----------------------------------------------------------- classification
SPAM_WORDS = ["you have won", "lucky winner", "claim your prize", "click here", "lottery", "free iphone"]
LEGAL_WORDS = ["lawyer", "legal action", "gdpr", "art. 17", "deletion of all", "data protection"]
KEYWORDS = {
    "billing": ["invoice", "charged", "refund", "payment", "billing", "credit card", "bills us"],
    "technical": ["error", "crash", "bug", "can't log", "cannot log", "503", "not loading", "export"],
    "sales": ["pricing", "quote", "demo", "enterprise plan", "evaluating", "interested in", "trial"],
}
NEGATIVE_WORDS = ["unacceptable", "frustrat", "third time", "angry", "crash", "blocked", "wrong", "legal action"]
POSITIVE_WORDS = ["interested", "great", "love", "happy"]


def _first_hit(text: str, words: Iterable[str]) -> int:
    positions = [text.find(w) for w in words if w in text]
    return min(positions) if positions else len(text) + 1


def classify(email: ParsedEmail) -> EmailClassification:
    text = email.text
    if any(w in text for w in SPAM_WORDS):
        return EmailClassification(
            category="spam",
            priority="low",
            sentiment="neutral",
            needs_human=False,
            summary="Unsolicited prize / phishing e-mail.",
        )
    negative = any(w in text for w in NEGATIVE_WORDS)
    sentiment = "negative" if negative else ("positive" if any(w in text for w in POSITIVE_WORDS) else "neutral")
    legal = any(w in text for w in LEGAL_WORDS)
    scores = {cat: sum(w in text for w in words) for cat, words in KEYWORDS.items()}
    # The concern mentioned first is the primary one (customers lead with their main issue).
    hits = sorted((c for c, s in scores.items() if s), key=lambda c: _first_hit(text, KEYWORDS[c]))
    if legal:
        category, additional = "general", []
    else:
        category, additional = (hits[0], hits[1:]) if hits else ("general", [])
    urgent = legal or ("asap" in text and negative) or "blocked" in text
    priority = "urgent" if urgent else ("high" if negative else "normal")
    return EmailClassification(
        category=category,
        additional_categories=additional,
        priority=priority,
        sentiment=sentiment,
        needs_human=legal,
        summary=summarize(email, [category, *additional], legal),
    )


def summarize(email: ParsedEmail, categories: list[str], legal: bool = False) -> str:
    text, parts = email.text, []
    invoices = ", ".join(email.invoice_ids) or "an invoice"
    if "billing" in categories:
        if "twice" in text or "duplicate" in text:
            parts.append(f"duplicate charge on {invoices}, refund requested")
        elif "seats" in text:
            parts.append(f"{invoices} bills more seats than contracted")
        else:
            parts.append(f"billing question about {invoices}")
    if "technical" in categories:
        if "503" in text:
            parts.append("dashboard unavailable (HTTP 503)")
        if "export" in text:
            parts.append("CSV export crashes for a large project")
        if not ("503" in text or "export" in text):
            parts.append("technical problem")
    if "sales" in categories:
        parts.append(f"pricing and demo request for the Enterprise plan ({email.seats or '?'} seats)")
    if legal:
        parts.append("GDPR erasure request with threat of legal action")
    if not parts:
        parts.append(email.subject)
    summary = "; ".join(parts)
    return summary[0].upper() + summary[1:] + "."


def domains_for(classification: EmailClassification) -> list[str]:
    """Specialist domains that must work on an e-mail (in order)."""
    if classification.category == "spam":
        return []
    domains = [CATEGORY_TO_DOMAIN[c] for c in classification.categories if c in CATEGORY_TO_DOMAIN]
    if classification.needs_human and "relations" not in domains:
        domains.append("relations")
    return list(dict.fromkeys(domains))


# --------------------------------------------------------- tool-call memory
def call_key(tool: str, args: dict[str, Any]) -> tuple:
    """Identity of a tool call, used to decide whether it was already made."""
    keys = {
        "lookup_customer": ("email",),
        "get_invoice": ("invoice_id",),
        "search_knowledge_base": ("query",),
        "check_service_status": ("service",),
        "get_plan_pricing": ("plan", "seats"),
        "issue_refund": ("invoice_id",),
        "create_ticket": ("customer_id", "queue"),
        "create_sales_lead": ("contact_email",),
        "escalate_to_human": ("customer_email", "team"),
    }.get(tool, tuple(sorted(args)))
    return (tool, *(str(args.get(k, "")).lower() for k in keys))


@dataclass
class Context:
    """What the model knows from tool results seen so far."""

    customers: dict[str, dict] = field(default_factory=dict)
    invoices: dict[str, dict] = field(default_factory=dict)
    articles: dict[str, dict] = field(default_factory=dict)
    status: dict[str, dict] = field(default_factory=dict)
    pricing: list[dict] = field(default_factory=list)
    refunds: dict[str, dict] = field(default_factory=dict)
    tickets: dict[str, dict] = field(default_factory=dict)
    leads: dict[str, dict] = field(default_factory=dict)
    escalations: dict[str, dict] = field(default_factory=dict)
    done: set[tuple] = field(default_factory=set)

    @classmethod
    def from_calls(cls, calls: Iterable[dict[str, Any]]) -> Context:
        ctx = cls()
        for call in calls:
            ctx.add(call.get("tool", ""), call.get("args") or {}, call.get("result"))
        return ctx

    def add(self, tool: str, args: dict[str, Any], result: Any) -> None:
        self.done.add(call_key(tool, args))
        if not isinstance(result, (dict, list)):
            return
        if tool == "lookup_customer":
            self.customers[str(args.get("email", "")).lower()] = result
        elif tool == "get_invoice":
            self.invoices[str(args.get("invoice_id", "")).upper()] = result
        elif tool == "search_knowledge_base":
            for article in result if isinstance(result, list) else []:
                self.articles[article["id"]] = article
        elif tool == "check_service_status":
            self.status[str(args.get("service"))] = result
        elif tool == "get_plan_pricing":
            self.pricing.append(result)
        elif tool == "issue_refund":
            self.refunds[str(args.get("invoice_id", "")).upper()] = result
        elif tool == "create_ticket":
            self.tickets[str(args.get("queue"))] = result
        elif tool == "create_sales_lead":
            self.leads[str(args.get("contact_email", "")).lower()] = result
        elif tool == "escalate_to_human":
            self.escalations[str(args.get("team"))] = result

    def customer(self, email: ParsedEmail) -> dict:
        return self.customers.get(email.sender, {})


# ------------------------------------------------------ domain procedures
# Each domain is handled in three phases: gather information (read-only
# tools, issued in parallel), act (side-effect tools), and report.


def _kb_query(domain: str, email: ParsedEmail) -> str:
    text = email.text
    if domain == "billing":
        return "refund duplicate charge" if ("twice" in text or "duplicate" in text) else "invoice correction seats"
    if domain == "technical":
        parts = []
        if "dashboard" in email.services:
            parts.append("dashboard 503 login error")
        if "export" in email.services:
            parts.append("csv export crash large")
        return " ".join(parts) or "error troubleshooting"
    if domain == "sales":
        return "enterprise plan pricing demo"
    return "gdpr deletion lawyer legal complaint"


def info_calls(domain: str, email: ParsedEmail) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = [("lookup_customer", {"email": email.sender})]
    if domain == "billing":
        calls += [("get_invoice", {"invoice_id": i}) for i in email.invoice_ids]
    if domain == "technical":
        calls += [("check_service_status", {"service": s}) for s in email.services]
    if domain == "sales":
        plan = "enterprise" if "enterprise" in email.text else "business"
        calls.append(("get_plan_pricing", {"plan": plan, "seats": email.seats or 10}))
    calls.append(("search_knowledge_base", {"query": _kb_query(domain, email)}))
    return calls


def action_calls(
    domain: str, email: ParsedEmail, ctx: Context, priority: str = "normal"
) -> list[tuple[str, dict[str, Any]]]:
    customer = ctx.customer(email)
    customer_id = customer.get("customer_id", "UNKNOWN")
    calls: list[tuple[str, dict[str, Any]]] = []
    if domain == "billing":
        for invoice_id in email.invoice_ids:
            invoice = ctx.invoices.get(invoice_id, {})
            if not invoice.get("found"):
                continue
            overpaid = sum(p["amount"] for p in invoice.get("payments", [])) - invoice["amount"]
            if invoice["status"] == "paid" and overpaid > 0:
                calls.append(
                    ("issue_refund", {"invoice_id": invoice_id, "amount": overpaid, "reason": "duplicate charge"})
                )
            if invoice["seats_billed"] > invoice["contracted_seats"]:
                calls.append(
                    (
                        "create_ticket",
                        {
                            "customer_id": customer_id,
                            "queue": "billing",
                            "priority": "high",
                            "summary": f"Correct {invoice_id}: {invoice['seats_billed']} seats billed, "
                            f"{invoice['contracted_seats']} contracted; issue credit note.",
                        },
                    )
                )
    elif domain == "technical":
        incidents = [s["incident"] for s in ctx.status.values() if s.get("incident")]
        summary = summarize(email, ["technical"]).rstrip(".")
        if incidents:
            summary += f" - linked to incident {', '.join(incidents)}"
        if "KB-310" in ctx.articles:
            summary += " - known issue KB-310"
        calls.append(
            (
                "create_ticket",
                {"customer_id": customer_id, "queue": "technical", "summary": summary, "priority": priority},
            )
        )
    elif domain == "sales":
        if not customer.get("found"):
            calls.append(
                (
                    "create_sales_lead",
                    {
                        "company": email.company,
                        "contact_email": email.sender,
                        "seats": email.seats or 10,
                        "plan_interest": "enterprise" if "enterprise" in email.text else "business",
                        "notes": "Demo requested" if "demo" in email.text else "",
                    },
                )
            )
    elif domain == "relations":
        gdpr = any(w in email.text for w in ("gdpr", "art. 17", "deletion"))
        calls.append(
            (
                "escalate_to_human",
                {
                    "customer_email": email.sender,
                    "team": "dpo" if gdpr else "team_lead",
                    "reason": "GDPR erasure request with legal threat" if gdpr else "Complaint / legal threat",
                    "summary": summarize(email, [], legal=True),
                },
            )
        )
    return calls


def pending_calls(
    domain: str,
    email: ParsedEmail,
    ctx: Context,
    priority: str = "normal",
    available: set[str] | None = None,
) -> list[tuple[str, dict]]:
    """Next batch of tool calls for a domain ([] when the domain is finished).

    `available` restricts the calls to tools the model can currently use.
    """

    def usable(call: tuple[str, dict]) -> bool:
        return call_key(*call) not in ctx.done and (available is None or call[0] in available)

    gather = [c for c in info_calls(domain, email) if usable(c)]
    if gather:
        return gather
    return [c for c in action_calls(domain, email, ctx, priority) if usable(c)]


def report(domain: str, email: ParsedEmail, ctx: Context) -> SpecialistReport:
    """Write the specialist report once all calls of a domain are done."""
    actions: list[str] = []
    paragraphs: list[str] = []
    findings: list[str] = []
    needs_human = False
    customer = ctx.customer(email)
    if customer.get("found"):
        findings.append(f"Customer {customer['customer_id']} ({customer['company']}, {customer['plan']}).")

    if domain == "billing":
        for invoice_id in email.invoice_ids:
            invoice = ctx.invoices.get(invoice_id, {})
            if not invoice.get("found"):
                paragraphs.append(f"I could not find invoice {invoice_id} - could you double-check the number?")
                continue
            refund = ctx.refunds.get(invoice_id)
            if refund and refund.get("status") == "processed":
                actions.append(f"Refund {refund['refund_id']} of {refund['amount']:.2f} EUR issued for {invoice_id}")
                paragraphs.append(
                    f"I checked invoice {invoice_id}: the amount of {invoice['amount']:.2f} EUR was indeed charged "
                    f"twice on {invoice['payments'][0]['date']}. I have refunded the duplicate payment "
                    f"(reference {refund['refund_id']}); the money will be back on your original payment method "
                    f"within 5-7 business days."
                )
                findings.append(f"Duplicate payment on {invoice_id} refunded.")
            elif refund:
                needs_human = True
                findings.append(f"Refund for {invoice_id} rejected: {refund.get('reason')}")
                paragraphs.append(f"I have forwarded your refund request for {invoice_id} to our finance team.")
            ticket = ctx.tickets.get("billing")
            if invoice["seats_billed"] > invoice["contracted_seats"] and ticket:
                diff = invoice["seats_billed"] - invoice["contracted_seats"]
                credit = diff * invoice["unit_price"]
                actions.append(f"Billing correction ticket {ticket['ticket_id']} created for {invoice_id}")
                paragraphs.append(
                    f"You are right: invoice {invoice_id} lists {invoice['seats_billed']} seats while your contract "
                    f"covers {invoice['contracted_seats']}. I have opened a billing correction (reference "
                    f"{ticket['ticket_id']}); you will receive a corrected invoice with a credit of {credit:.2f} EUR "
                    f"within 3 business days. Please don't pay the current invoice in the meantime."
                )
                findings.append(f"{invoice_id} overbilled by {diff} seats ({credit:.2f} EUR).")
        if not email.invoice_ids:
            paragraphs.append("Could you send us the invoice number so we can look into this for you?")

    elif domain == "technical":
        ticket = ctx.tickets.get("technical")
        ref = f" (reference {ticket['ticket_id']})" if ticket else ""
        if ticket:
            actions.append(f"Technical ticket {ticket['ticket_id']} created")
        dashboard = ctx.status.get("dashboard", {})
        if "dashboard" in email.services and dashboard.get("incident"):
            paragraphs.append(
                f"The dashboard is currently affected by an incident ({dashboard['incident']}): "
                f"{dashboard['summary']} Our engineers expect a fix by {dashboard['eta']}; you can follow the "
                f"progress at {STATUS_PAGE}. I have linked your report to the incident{ref} so we can notify you "
                f"as soon as it is resolved. The API is not affected in the meantime."
            )
            findings.append(f"Active incident {dashboard['incident']} explains the 503 errors.")
        if "export" in email.services and "KB-310" in ctx.articles:
            paragraphs.append(
                "The CSV export problem is a known issue in the current release for exports with more than "
                "10,000 rows. As a workaround, please filter the export by date range so that each file stays "
                "below 10,000 rows. A permanent fix ships with version 4.2 on 15 October 2026. I have logged "
                f"your case{ref} and we will let you know when the fix is live."
            )
            findings.append("Export crash matches known issue KB-310 (fix in v4.2).")
        if not paragraphs:
            paragraphs.append(f"Our support engineers are looking into the problem{ref} and will get back to you.")

    elif domain == "sales":
        pricing = ctx.pricing[-1] if ctx.pricing else None
        lead = ctx.leads.get(email.sender)
        executive = (lead or {}).get("account_executive") or customer.get("account_manager") or "our sales team"
        if lead:
            actions.append(f"Sales lead {lead['lead_id']} created, assigned to {executive}")
        if pricing:
            discount = (
                f" (including a {int(pricing['discount'] * 100)}% volume discount)" if pricing["discount"] else ""
            )
            paragraphs.append(
                f"For {pricing['seats']} seats, Acme Cloud {pricing['plan'].title()} costs "
                f"{pricing['price_per_user']:.2f} EUR per user and month{discount}, i.e. "
                f"{pricing['monthly_total']:.2f} EUR per month with annual billing. The plan includes SSO, audit "
                "logs, a 99.9% SLA and a dedicated account executive."
            )
        demo = " to schedule your demo" if "demo" in email.text else ""
        ref = f" (reference {lead['lead_id']})" if lead else ""
        paragraphs.append(f"{executive} will contact you within one business day{demo}{ref}.")
        findings.append(f"New lead for {pricing['seats'] if pricing else '?'} seats." if lead else "Existing customer.")

    elif domain == "relations":
        needs_human = True
        for team, escalation in ctx.escalations.items():
            actions.append(f"Escalated to {team} ({escalation['escalation_id']})")
            if team == "dpo":
                paragraphs.append(
                    "Your request regarding the deletion of personal data has been forwarded to our Data Protection Officer (reference "
                    f"{escalation['escalation_id']}), who will get back to you within 72 hours."
                )
            else:
                paragraphs.append(
                    "A team lead will personally "
                    f"review your message (reference {escalation['escalation_id']}) and contact you within one "
                    "business day."
                )
        findings.append("Case requires a human (data protection / legal).")

    return SpecialistReport(
        domain=domain,
        findings=" ".join(findings) or "No findings.",
        actions_taken=actions,
        customer_reply="\n\n".join(paragraphs),
        needs_human=needs_human,
    )


# ------------------------------------------------------------- composition
def compose_body(email: ParsedEmail, sentiment: str, paragraphs: list[str]) -> str:
    opening = (
        "Thank you for your message, and I am sorry for the trouble this has caused."
        if sentiment == "negative"
        else "Thank you for contacting Acme Cloud."
    )
    body = "\n\n".join(p for p in paragraphs if p)
    return (
        f"Dear {email.first_name},\n\n{opening}\n\n{body}\n\n"
        f"If you have any further questions, simply reply to this e-mail.\n\n{SIGNATURE}"
    )


def quick_body(email: ParsedEmail) -> str:
    """A hasty first draft (used to demonstrate evaluator-optimizer loops)."""
    return "Hi,\n\nwe have looked into your request and taken care of it.\n\nAcme Cloud"


def _field(report: Any, name: str, default: Any = None) -> Any:
    return report.get(name, default) if isinstance(report, dict) else getattr(report, name, default)


def compose_resolution(
    email: ParsedEmail,
    classification: EmailClassification,
    reports: list[Any],
    *,
    quick: bool = False,
) -> EmailResolution:
    """Merge specialist (or team) reports into the final resolution."""
    if classification.category == "spam":
        return EmailResolution(
            email_id=email.id,
            category="spam",
            priority="low",
            action="ignore",
            internal_note="Spam / phishing - no reply sent.",
        )
    actions = [a for r in reports for a in _field(r, "actions_taken", [])]
    needs_human = classification.needs_human or any(_field(r, "needs_human", False) for r in reports)
    paragraphs = [_field(r, "customer_reply", "") for r in reports]
    body = quick_body(email) if quick else compose_body(email, classification.sentiment, paragraphs)
    findings = " ".join(_field(r, "findings", "") for r in reports)
    return EmailResolution(
        email_id=email.id,
        category=classification.category,
        priority=classification.priority,
        action="escalate" if needs_human else "reply",
        actions_taken=actions,
        reply_subject=f"Re: {email.subject}",
        reply_body=body,
        escalation_reason=classification.summary if needs_human else None,
        internal_note=f"{classification.summary} {findings}".strip(),
    )


def resolve_from_context(
    email: ParsedEmail, ctx: Context, classification: EmailClassification | None = None, *, quick: bool = False
) -> EmailResolution:
    """Final resolution when all domain work is visible in the context."""
    classification = classification or classify(email)
    reports = [report(d, email, ctx) for d in domains_for(classification)]
    return compose_resolution(email, classification, reports, quick=quick)


# ------------------------------------------------------------------ review
def review(email: ParsedEmail, draft: dict[str, Any], sentiment: str) -> list[str]:
    """Quality checks for a drafted resolution; returns a list of issues."""
    if draft.get("action") == "ignore":
        return []
    body = draft.get("reply_body") or ""
    issues = []
    if not body:
        return ["The resolution has no reply to the customer."]
    if email.first_name.lower() not in body.lower():
        issues.append(f"Address the customer by name ({email.first_name}).")
    refs = sorted(set(_REF_RE.findall(" ".join(draft.get("actions_taken") or []))))
    missing = [r for r in refs if r not in body]
    if missing:
        issues.append(f"Mention the reference number(s) {', '.join(missing)} so the customer can refer to them.")
    if sentiment == "negative" and not re.search(r"sorry|apolog", body, re.I):
        issues.append("Acknowledge the customer's frustration with an apology.")
    if "Best regards" not in body:
        issues.append("End with the standard signature ('Best regards, Acme Cloud Customer Service').")
    if re.search(r"\b(tier|platinum|gold|internal)\b", body, re.I):
        issues.append("Remove internal information (customer tier, internal notes).")
    if re.search(r"(have|has been) deleted", body, re.I):
        issues.append("Never confirm a data deletion yourself.")
    return issues


def to_call_list(calls: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [{"tool": t, "args": a} for t, a in calls]


REFERENCE_RE = _REF_RE
