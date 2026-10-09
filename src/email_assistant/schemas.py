"""Pydantic schemas of the e-mail use case (inputs, structured outputs, results)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["billing", "technical", "sales", "general", "spam"]
Priority = Literal["low", "normal", "high", "urgent"]
Sentiment = Literal["positive", "neutral", "negative"]
Domain = Literal["billing", "technical", "sales", "relations"]

#: Which specialist domain handles which category.
CATEGORY_TO_DOMAIN: dict[str, Domain] = {
    "billing": "billing",
    "technical": "technical",
    "sales": "sales",
    "general": "relations",
}


class Email(BaseModel):
    """An incoming customer e-mail (workflow input)."""

    id: str
    sender: str
    sender_name: str
    subject: str
    body: str
    received_at: str

    def as_prompt(self) -> str:
        """Render the e-mail the way it is shown to models (XML-tagged, so it can be quoted)."""
        return (
            "<email>\n"
            f"Email-ID: {self.id}\n"
            f"From: {self.sender_name} <{self.sender}>\n"
            f"Subject: {self.subject}\n"
            f"Received: {self.received_at}\n\n"
            f"{self.body}\n"
            "</email>"
        )


class EmailClassification(BaseModel):
    """Triage result for an incoming customer e-mail."""

    category: Category = Field(description="Primary category of the e-mail.")
    additional_categories: list[Category] = Field(
        default_factory=list, description="Further intents contained in the same e-mail."
    )
    priority: Priority
    sentiment: Sentiment
    needs_human: bool = Field(
        description="True for legal threats, data-protection requests or anything the AI must not answer alone."
    )
    summary: str = Field(description="One-sentence summary of the request.")

    @property
    def categories(self) -> list[str]:
        return [self.category, *[c for c in self.additional_categories if c != self.category]]


class SpecialistReport(BaseModel):
    """Result of a specialist that handled its part of an e-mail."""

    domain: Domain
    findings: str = Field(description="Internal summary of what was found (not shown to the customer).")
    actions_taken: list[str] = Field(default_factory=list, description="Side effects, with reference IDs.")
    customer_reply: str = Field(description="Paragraph(s) to include in the reply to the customer.")
    needs_human: bool = False


class TeamReport(BaseModel):
    """Merged result of a team (hierarchical pattern)."""

    team: str
    findings: str
    actions_taken: list[str] = Field(default_factory=list)
    customer_reply: str
    needs_human: bool = False


class EmailResolution(BaseModel):
    """Final decision on how an e-mail is handled (output of every workflow)."""

    email_id: str
    category: Category
    priority: Priority
    action: Literal["reply", "escalate", "ignore"] = Field(
        description="reply = send reply_body; escalate = hand to a human (optionally with an acknowledgment); ignore = no reply."
    )
    actions_taken: list[str] = Field(default_factory=list)
    reply_subject: str | None = None
    reply_body: str | None = None
    escalation_reason: str | None = None
    internal_note: str = Field(description="Short note for the CRM timeline.")


# ------------------------------------------------ pattern-specific outputs
class SentimentAnalysis(BaseModel):
    """Sentiment and urgency of an e-mail."""

    sentiment: Sentiment
    urgency: Priority
    signals: list[str] = Field(default_factory=list, description="Phrases signalling frustration or urgency.")


class ExtractedEntities(BaseModel):
    """Entities mentioned in an e-mail."""

    sender: str
    customer_name: str
    invoice_ids: list[str] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    seats: int | None = None


class ComplianceScreen(BaseModel):
    """Compliance flags of an e-mail."""

    phishing: bool
    legal_threat: bool
    data_protection_request: bool
    notes: str = ""


class RefundAction(BaseModel):
    kind: Literal["refund"] = "refund"
    invoice_id: str
    amount: float
    reason: str


class TicketAction(BaseModel):
    kind: Literal["ticket"] = "ticket"
    customer_id: str
    queue: Literal["billing", "technical", "sales"]
    summary: str
    priority: Priority = "normal"


class LeadAction(BaseModel):
    kind: Literal["lead"] = "lead"
    company: str
    contact_email: str
    seats: int
    plan_interest: str
    notes: str = ""


class EscalationAction(BaseModel):
    kind: Literal["escalation"] = "escalation"
    customer_email: str
    team: Literal["dpo", "team_lead", "finance"]
    reason: str
    summary: str


class ActionPlan(BaseModel):
    """Actions the workflow should execute for an e-mail (executed by code, not by the model)."""

    actions: list[RefundAction | TicketAction | LeadAction | EscalationAction] = Field(default_factory=list)
    rationale: str = ""


class QualityReview(BaseModel):
    """Verdict of a quality review."""

    passed: bool
    score: float = Field(ge=0, le=1)
    issues: list[str] = Field(default_factory=list)
    feedback: str


class Delivery(BaseModel):
    """What the deterministic dispatch step did with a resolution."""

    email_id: str
    status: Literal["sent", "escalated", "ignored"]
    outbox_id: str | None = None
    escalation_id: str | None = None
