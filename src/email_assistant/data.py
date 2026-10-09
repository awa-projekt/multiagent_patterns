"""Mock data sources of "Acme Cloud", a fictional B2B SaaS company.

Everything here stands in for real systems (CRM, billing, knowledge base,
status page, ticketing, outbox). Read-only data is static; side effects
(refunds, tickets, leads, escalations, sent e-mails) go into an in-memory
`Backend` that tests can inspect and reset.

IDs of created records are derived from their content (not from a counter), so
parallel branches produce the same IDs regardless of execution order.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any

from email_assistant.schemas import Email

# --------------------------------------------------------------------- inbox
INBOX: list[Email] = [
    Email(
        id="E-1001",
        sender="anna.schmidt@contoso.example",
        sender_name="Anna Schmidt",
        subject="Charged twice for September",
        body=(
            "Hello,\n\nI just noticed that our credit card was charged twice for invoice "
            "INV-2026-0815 (348.00 EUR each, both on 1 September). Could you please refund "
            "the duplicate payment?\n\nThanks,\nAnna"
        ),
        received_at="2026-09-27T08:05:00+02:00",
    ),
    Email(
        id="E-1002",
        sender="mark.jones@fabrikam.example",
        sender_name="Mark Jones",
        subject="Dashboard not loading - error 503",
        body=(
            "Hi support,\n\nsince this morning I can't log into the dashboard, it just shows "
            "'503 Service Unavailable'. Our whole team is blocked, this is really frustrating. "
            "Please help asap!\n\nMark"
        ),
        received_at="2026-09-27T08:41:00+02:00",
    ),
    Email(
        id="E-1003",
        sender="procurement@northwind.example",
        sender_name="Priya Natarajan",
        subject="Enterprise plan for 50 users",
        body=(
            "Dear Acme team,\n\nwe are evaluating collaboration platforms for Northwind Traders. "
            "Could you send us pricing for the Enterprise plan for 50 seats? We would also be "
            "interested in a demo next week.\n\nKind regards,\nPriya Natarajan\nHead of Procurement"
        ),
        received_at="2026-09-27T09:12:00+02:00",
    ),
    Email(
        id="E-1004",
        sender="j.doe@tailspin.example",
        sender_name="Jordan Doe",
        subject="Wrong invoice + export crash",
        body=(
            "Hi,\n\ntwo things: invoice INV-2026-0901 bills us for 60 seats, but our contract "
            "covers 50 seats. Please correct it.\n\nAlso, the CSV export crashes every time for "
            "our large project (about 40k rows). We need that export for our monthly report.\n\n"
            "Best,\nJordan"
        ),
        received_at="2026-09-27T09:30:00+02:00",
    ),
    Email(
        id="E-1005",
        sender="winner-notice@lucky-prize.example",
        sender_name="Prize Department",
        subject="Congratulations!!! You have WON",
        body=(
            "You have been selected as the lucky winner of a brand new iPhone! "
            "Click here to claim your prize within 24 hours: http://lucky-prize.example/claim"
        ),
        received_at="2026-09-27T09:44:00+02:00",
    ),
    Email(
        id="E-1006",
        sender="anna.schmidt@contoso.example",
        sender_name="Anna Schmidt",
        subject="Formal request: deletion of our data",
        body=(
            "To whom it may concern,\n\nthis is the third time we have had problems this year and "
            "it is unacceptable. Under GDPR Art. 17 we request the deletion of all personal data "
            "of our employees stored in your system. If we do not receive a confirmation, our "
            "lawyer will take legal action.\n\nAnna Schmidt\nContoso GmbH"
        ),
        received_at="2026-09-27T10:02:00+02:00",
    ),
]

#: Expected triage per e-mail; used by tests and the comparison table.
EXPECTED: dict[str, dict[str, Any]] = {
    "E-1001": {"category": "billing", "action": "reply"},
    "E-1002": {"category": "technical", "action": "reply"},
    "E-1003": {"category": "sales", "action": "reply"},
    "E-1004": {"category": "billing", "action": "reply", "also": ["technical"]},
    "E-1005": {"category": "spam", "action": "ignore"},
    "E-1006": {"category": "general", "action": "escalate"},
}

# ----------------------------------------------------------------------- CRM
CUSTOMERS: dict[str, dict[str, Any]] = {
    "anna.schmidt@contoso.example": {
        "customer_id": "C-1001",
        "name": "Anna Schmidt",
        "company": "Contoso GmbH",
        "plan": "Business",
        "seats": 12,
        "tier": "gold",
        "account_manager": "Lena Weber",
    },
    "mark.jones@fabrikam.example": {
        "customer_id": "C-1002",
        "name": "Mark Jones",
        "company": "Fabrikam Inc.",
        "plan": "Starter",
        "seats": 8,
        "tier": "standard",
        "account_manager": None,
    },
    "j.doe@tailspin.example": {
        "customer_id": "C-1003",
        "name": "Jordan Doe",
        "company": "Tailspin Toys",
        "plan": "Enterprise",
        "seats": 50,
        "tier": "platinum",
        "account_manager": "Tom Becker",
    },
}

# ------------------------------------------------------------------- billing
INVOICES: dict[str, dict[str, Any]] = {
    "INV-2026-0815": {
        "invoice_id": "INV-2026-0815",
        "customer_id": "C-1001",
        "period": "2026-09",
        "plan": "Business",
        "seats_billed": 12,
        "contracted_seats": 12,
        "unit_price": 29.0,
        "amount": 348.0,
        "currency": "EUR",
        "status": "paid",
        "payments": [
            {"payment_id": "PAY-77120", "amount": 348.0, "date": "2026-09-01"},
            {"payment_id": "PAY-77121", "amount": 348.0, "date": "2026-09-01"},
        ],
    },
    "INV-2026-0901": {
        "invoice_id": "INV-2026-0901",
        "customer_id": "C-1003",
        "period": "2026-09",
        "plan": "Enterprise",
        "seats_billed": 60,
        "contracted_seats": 50,
        "unit_price": 45.0,
        "amount": 2700.0,
        "currency": "EUR",
        "status": "open",
        "payments": [],
    },
}

PRICING = {"starter": 12.0, "business": 29.0, "enterprise": 45.0}
ENTERPRISE_VOLUME_DISCOUNT = 0.15  # from 50 seats
REFUND_LIMIT_EUR = 500.0  # agents may refund up to this amount without finance approval

# ------------------------------------------------------------ knowledge base
KNOWLEDGE_BASE: list[dict[str, Any]] = [
    {
        "id": "KB-101",
        "title": "Refund policy",
        "tags": ["refund", "duplicate", "charge", "charged", "twice", "payment", "billing"],
        "content": (
            "Duplicate or erroneous charges are refunded in full to the original payment method "
            "within 5-7 business days. Support agents may issue refunds up to 500 EUR per invoice; "
            "larger amounts require finance approval (escalate to finance)."
        ),
    },
    {
        "id": "KB-102",
        "title": "Invoice corrections",
        "tags": ["invoice", "correction", "seats", "wrong", "credit", "billing"],
        "content": (
            "If an invoice lists more seats than contracted, open a billing correction ticket. "
            "A corrected invoice with a credit note is issued within 3 business days. Do not refund "
            "open (unpaid) invoices - correct them instead and ask the customer to wait for the new invoice."
        ),
    },
    {
        "id": "KB-205",
        "title": "Dashboard login errors (401/403/503)",
        "tags": ["dashboard", "login", "503", "error", "unavailable", "sso"],
        "content": (
            "HTTP 503 on the dashboard indicates a service disruption: check the status page. If an "
            "incident is active, link the customer's ticket to the incident and share "
            "https://status.acme-cloud.example. For 401/403 errors, ask the customer to reset their SSO session."
        ),
    },
    {
        "id": "KB-310",
        "title": "CSV export fails for large projects",
        "tags": ["export", "csv", "crash", "large", "rows", "report"],
        "content": (
            "Known issue in v4.1: CSV exports with more than 10,000 rows can fail. Workaround: filter the "
            "export by date range so that each file stays below 10,000 rows. A fix ships with v4.2 on "
            "2026-10-15."
        ),
    },
    {
        "id": "KB-400",
        "title": "Plans and pricing",
        "tags": ["pricing", "price", "plan", "enterprise", "business", "starter", "seats", "demo", "quote"],
        "content": (
            "Starter 12 EUR, Business 29 EUR, Enterprise 45 EUR per user and month (annual billing). "
            "Enterprise: 15% volume discount from 50 seats, SSO, audit log, 99.9% SLA and a dedicated "
            "account executive. New leads are assigned to an account executive who schedules demos."
        ),
    },
    {
        "id": "KB-900",
        "title": "Data protection requests (GDPR)",
        "tags": ["gdpr", "deletion", "delete", "erasure", "art. 17", "personal data", "privacy"],
        "content": (
            "Requests under GDPR (e.g. Art. 17 erasure) must be escalated to the Data Protection Officer. "
            "Send a neutral acknowledgment with the escalation reference and promise an answer within "
            "72 hours. Never confirm the deletion yourself."
        ),
    },
    {
        "id": "KB-910",
        "title": "Complaints and legal threats",
        "tags": ["lawyer", "legal", "complaint", "unacceptable", "threat"],
        "content": (
            "E-mails that mention lawyers or legal action are handled by a human team lead. Do not admit "
            "fault or promise compensation; acknowledge receipt politely and escalate."
        ),
    },
]

# --------------------------------------------------------------- status page
SERVICE_STATUS: dict[str, dict[str, Any]] = {
    "dashboard": {
        "service": "dashboard",
        "status": "degraded",
        "incident": "INC-7781",
        "summary": "Elevated 503 errors on dashboard login since 08:12 CEST; a fix is being deployed.",
        "eta": "11:00 CEST",
    },
    "api": {"service": "api", "status": "operational"},
    "export": {"service": "export", "status": "operational"},
    "billing": {"service": "billing", "status": "operational"},
}

ACCOUNT_EXECUTIVES = ["Tom Becker", "Lena Weber"]


# ------------------------------------------------------------- side effects
def stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()
    return f"{prefix}-{int(digest, 16) % 9000 + 1000}"


@dataclass
class Backend:
    """In-memory store for everything the agents *do*."""

    refunds: dict[str, dict[str, Any]] = field(default_factory=dict)
    tickets: dict[str, dict[str, Any]] = field(default_factory=dict)
    leads: dict[str, dict[str, Any]] = field(default_factory=dict)
    escalations: dict[str, dict[str, Any]] = field(default_factory=dict)
    outbox: dict[str, dict[str, Any]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reset(self) -> None:
        with self.lock:
            for store in (self.refunds, self.tickets, self.leads, self.escalations, self.outbox):
                store.clear()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self.lock:
            return {
                "refunds": dict(self.refunds),
                "tickets": dict(self.tickets),
                "leads": dict(self.leads),
                "escalations": dict(self.escalations),
                "outbox": dict(self.outbox),
            }


BACKEND = Backend()


def reset_backend() -> None:
    BACKEND.reset()
