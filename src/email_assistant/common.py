"""State schemas and the deterministic dispatch step shared by all e-mail workflows.

Every workflow - native or library-based - has the same contract:

    input:  {"email": Email}
    output: {"resolution": EmailResolution, "delivery": Delivery}

so they can be compared, swapped and embedded in bigger graphs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NotRequired, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentpatterns import agent_as_node, make_serializer
from email_assistant import schemas
from email_assistant.data import BACKEND, stable_id
from email_assistant.schemas import Delivery, Email, EmailResolution


class EmailInput(TypedDict):
    email: Email


class EmailOutput(TypedDict):
    resolution: EmailResolution
    delivery: Delivery


class EmailState(TypedDict):
    email: Email
    resolution: NotRequired[EmailResolution]
    delivery: NotRequired[Delivery]


def create_checkpointer():
    """In-memory checkpointer that may store the use case's Pydantic schemas."""
    from langgraph.checkpoint.memory import InMemorySaver

    types = [v for v in vars(schemas).values() if isinstance(v, type) and issubclass(v, schemas.BaseModel)]
    return InMemorySaver(serde=make_serializer(*types))


def email_message(email: Email) -> HumanMessage:
    """The e-mail as the user message of an agent conversation."""
    return HumanMessage(content=f"New e-mail in the support inbox:\n\n{email.as_prompt()}")


def dispatch(state: EmailState) -> dict:
    """Deterministically execute the resolution: send the reply and/or escalate.

    Irreversible side effects (sending e-mails) stay in plain code - the model
    only *proposes* the reply. This is also the natural place for a
    human-in-the-loop approval (`interrupt()`) in production.
    """
    email, resolution = state["email"], state["resolution"]
    outbox_id = escalation_id = None
    if resolution.action != "ignore" and resolution.reply_body:
        outbox_id = stable_id("MSG", email.id)
        with BACKEND.lock:
            BACKEND.outbox[outbox_id] = {
                "to": email.sender,
                "subject": resolution.reply_subject or f"Re: {email.subject}",
                "body": resolution.reply_body,
                "in_reply_to": email.id,
            }
    if resolution.action == "escalate":
        with BACKEND.lock:
            existing = [e for e in BACKEND.escalations.values() if e["customer_email"] == email.sender]
            if existing:
                escalation_id = existing[0]["escalation_id"]
            else:  # the workflow decided to escalate without doing it itself
                escalation_id = stable_id("ESC", email.sender, "team_lead")
                BACKEND.escalations[escalation_id] = {
                    "escalation_id": escalation_id,
                    "customer_email": email.sender,
                    "team": "team_lead",
                    "reason": resolution.escalation_reason or "Escalated by workflow",
                    "summary": resolution.internal_note,
                    "sla": "1 business day",
                }
    status = {"reply": "sent", "escalate": "escalated", "ignore": "ignored"}[resolution.action]
    return {"delivery": Delivery(email_id=email.id, status=status, outbox_id=outbox_id, escalation_id=escalation_id)}


def build_email_workflow(
    pattern: Runnable,
    name: str,
    *,
    extra_input: Callable[[EmailState], dict[str, Any]] | None = None,
) -> CompiledStateGraph:
    """Embed any agent-contract pattern graph into the e-mail workflow.

        START -> <pattern as subgraph node> -> dispatch -> END

    The pattern speaks `messages`/`structured_response`, the workflow speaks
    `email`/`resolution`; `agent_as_node` maps between the two schemas.
    """
    node = agent_as_node(
        pattern,
        input=lambda state: {
            "messages": [email_message(state["email"])],
            **(extra_input(state) if extra_input else {}),
        },
        output=lambda result, state: {"resolution": result["structured_response"]},
        name=name,
    )
    return (
        StateGraph(EmailState, input_schema=EmailInput, output_schema=EmailOutput)
        .add_node(name, node)
        .add_node("dispatch", dispatch)
        .add_edge(START, name)
        .add_edge(name, "dispatch")
        .add_edge("dispatch", END)
        .compile(name=f"{name}_workflow")
    )
