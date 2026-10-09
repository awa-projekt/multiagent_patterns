"""LangChain tools over the mock data sources.

Read-only tools: lookup_customer, get_invoice, search_knowledge_base,
check_service_status, get_plan_pricing.
Tools with side effects: issue_refund, create_ticket, create_sales_lead,
escalate_to_human. Side-effect tools are idempotent (calling them twice with
the same key returns the existing record) and enforce business rules
themselves, so a confused agent cannot e.g. refund more than policy allows.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from langchain.tools import tool

from email_assistant.data import (
    ACCOUNT_EXECUTIVES,
    BACKEND,
    CUSTOMERS,
    ENTERPRISE_VOLUME_DISCOUNT,
    INVOICES,
    KNOWLEDGE_BASE,
    PRICING,
    REFUND_LIMIT_EUR,
    SERVICE_STATUS,
    stable_id,
)


# ---------------------------------------------------------------- read-only
@tool
def lookup_customer(email: str) -> dict[str, Any]:
    """Look up a customer in the CRM by the sender's e-mail address.

    Returns the customer record (customer_id, company, plan, seats, tier,
    account_manager) or {"found": false} for unknown senders.
    """
    record = CUSTOMERS.get(email.strip().lower())
    return {"found": True, **record} if record else {"found": False, "email": email}


@tool
def get_invoice(invoice_id: str) -> dict[str, Any]:
    """Fetch an invoice with its line items and payments (e.g. "INV-2026-0815")."""
    invoice = INVOICES.get(invoice_id.strip().upper())
    return {"found": True, **invoice} if invoice else {"found": False, "invoice_id": invoice_id}


@tool
def search_knowledge_base(query: str) -> list[dict[str, str]]:
    """Search internal policies and troubleshooting articles. Returns the 2 best matches."""
    words = set(re.findall(r"[a-z0-9.]+", query.lower()))

    def score(article: dict[str, Any]) -> int:
        haystack = " ".join([article["title"].lower(), *article["tags"]])
        return sum(1 for w in words if w in haystack)

    ranked = sorted(KNOWLEDGE_BASE, key=score, reverse=True)
    return [{"id": a["id"], "title": a["title"], "content": a["content"]} for a in ranked[:2] if score(a) > 0]


@tool
def check_service_status(service: Literal["dashboard", "api", "export", "billing"]) -> dict[str, Any]:
    """Check the public status page for a service (active incidents, ETA)."""
    return SERVICE_STATUS.get(service, {"service": service, "status": "unknown"})


@tool
def get_plan_pricing(plan: Literal["starter", "business", "enterprise"], seats: int) -> dict[str, Any]:
    """Calculate the monthly price of a plan for a number of seats (annual billing, EUR)."""
    unit = PRICING[plan]
    discount = ENTERPRISE_VOLUME_DISCOUNT if plan == "enterprise" and seats >= 50 else 0.0
    unit_price = round(unit * (1 - discount), 2)
    return {
        "plan": plan,
        "seats": seats,
        "list_price_per_user": unit,
        "discount": discount,
        "price_per_user": unit_price,
        "monthly_total": round(unit_price * seats, 2),
        "currency": "EUR",
    }


# ------------------------------------------------------------- side effects
@tool
def issue_refund(invoice_id: str, amount: float, reason: str) -> dict[str, Any]:
    """Refund money for a paid invoice (e.g. a duplicate charge).

    Policy is enforced here: only paid invoices, only up to the refundable
    amount, and at most 500 EUR without finance approval.
    """
    invoice = INVOICES.get(invoice_id.strip().upper())
    if not invoice:
        return {"status": "rejected", "reason": f"Unknown invoice {invoice_id}"}
    if invoice["status"] != "paid":
        return {"status": "rejected", "reason": "Invoice is not paid; correct the invoice instead of refunding."}
    overpaid = sum(p["amount"] for p in invoice["payments"]) - invoice["amount"]
    if amount > overpaid + 0.001:
        return {"status": "rejected", "reason": f"Refundable amount is {overpaid:.2f} EUR."}
    if amount > REFUND_LIMIT_EUR:
        return {"status": "rejected", "reason": "Amount above 500 EUR requires finance approval."}
    refund_id = stable_id("RF", invoice["invoice_id"])
    with BACKEND.lock:
        if refund_id in BACKEND.refunds:
            return {**BACKEND.refunds[refund_id], "note": "already refunded"}
        record = {
            "status": "processed",
            "refund_id": refund_id,
            "invoice_id": invoice["invoice_id"],
            "amount": amount,
            "currency": invoice["currency"],
            "reason": reason,
            "eta": "5-7 business days",
        }
        BACKEND.refunds[refund_id] = record
    return record


@tool
def create_ticket(
    customer_id: str,
    queue: Literal["billing", "technical", "sales"],
    summary: str,
    priority: Literal["low", "normal", "high", "urgent"] = "normal",
) -> dict[str, Any]:
    """Create a ticket in the ticketing system for follow-up work by a team."""
    ticket_id = stable_id("TCK", customer_id, queue)
    with BACKEND.lock:
        if ticket_id in BACKEND.tickets:
            return {**BACKEND.tickets[ticket_id], "note": "ticket already exists"}
        record = {
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "queue": queue,
            "summary": summary,
            "priority": priority,
            "status": "open",
        }
        BACKEND.tickets[ticket_id] = record
    return record


@tool
def create_sales_lead(
    company: str, contact_email: str, seats: int, plan_interest: str, notes: str = ""
) -> dict[str, Any]:
    """Register a new sales lead in the CRM and assign an account executive."""
    lead_id = stable_id("LEAD", contact_email.lower())
    executive = ACCOUNT_EXECUTIVES[int(lead_id.split("-")[1]) % len(ACCOUNT_EXECUTIVES)]
    with BACKEND.lock:
        if lead_id in BACKEND.leads:
            return {**BACKEND.leads[lead_id], "note": "lead already exists"}
        record = {
            "lead_id": lead_id,
            "company": company,
            "contact_email": contact_email,
            "seats": seats,
            "plan_interest": plan_interest,
            "notes": notes,
            "account_executive": executive,
        }
        BACKEND.leads[lead_id] = record
    return record


@tool
def escalate_to_human(
    customer_email: str,
    team: Literal["dpo", "team_lead", "finance"],
    reason: str,
    summary: str,
) -> dict[str, Any]:
    """Hand a case to a human team (data protection officer, team lead or finance)."""
    escalation_id = stable_id("ESC", customer_email.lower(), team)
    with BACKEND.lock:
        if escalation_id in BACKEND.escalations:
            return {**BACKEND.escalations[escalation_id], "note": "already escalated"}
        record = {
            "escalation_id": escalation_id,
            "customer_email": customer_email,
            "team": team,
            "reason": reason,
            "summary": summary,
            "sla": "72 hours" if team == "dpo" else "1 business day",
        }
        BACKEND.escalations[escalation_id] = record
    return record


# ---------------------------------------------------------------- toolsets
BILLING_TOOLS = [lookup_customer, get_invoice, search_knowledge_base, issue_refund, create_ticket]
TECHNICAL_TOOLS = [lookup_customer, check_service_status, search_knowledge_base, create_ticket]
SALES_TOOLS = [lookup_customer, search_knowledge_base, get_plan_pricing, create_sales_lead]
RELATIONS_TOOLS = [lookup_customer, search_knowledge_base, escalate_to_human]

READ_ONLY_TOOLS = [lookup_customer, get_invoice, search_knowledge_base, check_service_status, get_plan_pricing]
ACTION_TOOLS = [issue_refund, create_ticket, create_sales_lead, escalate_to_human]
ALL_TOOLS = READ_ONLY_TOOLS + ACTION_TOOLS

DOMAIN_TOOLS = {
    "billing": BILLING_TOOLS,
    "technical": TECHNICAL_TOOLS,
    "sales": SALES_TOOLS,
    "relations": RELATIONS_TOOLS,
}
