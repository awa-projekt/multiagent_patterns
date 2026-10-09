"""Pattern 10 - Skills (progressive disclosure), native implementation.

One agent stays in control. Its system prompt only lists skill *names and
descriptions*; the detailed playbook of a skill is loaded on demand with the
`load_skill` tool (it arrives as a tool result). Loading a skill also unlocks
the skill's tools (dynamic tool registration via `wrap_model_call`).

    agent -- load_skill("billing") --> playbook in context + billing tools unlocked
          -- get_invoice / issue_refund ... --> EmailResolution
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal, NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from email_assistant import prompts
from email_assistant.common import EmailInput, EmailOutput, EmailState, dispatch, email_message
from email_assistant.schemas import EmailResolution
from email_assistant.tools import ALL_TOOLS, DOMAIN_TOOLS, lookup_customer

SKILLS = {
    "billing": {
        "description": "Invoices, duplicate charges, refunds and invoice corrections.",
        "instructions": (
            "Billing playbook:\n1. Look up the customer and every invoice mentioned.\n2. Search the knowledge "
            "base for the applicable policy (refund policy KB-101, invoice corrections KB-102).\n3. Duplicate "
            "payment on a paid invoice: refund the overpaid amount (max 500 EUR, otherwise escalate to finance).\n"
            "4. More seats billed than contracted: open a billing ticket (priority high); never refund open invoices.\n"
            "5. Tell the customer the reference numbers and timelines (refund 5-7 business days, credit note 3 days)."
        ),
    },
    "technical": {
        "description": "Outages, error messages, bugs and known issues.",
        "instructions": (
            "Technical playbook:\n1. Check the status page for every service mentioned.\n2. Search the "
            "knowledge base for known issues and workarounds.\n3. Always open a technical ticket; link active "
            "incidents and known issues in the summary.\n4. Share the status page link, ETA and workarounds."
        ),
    },
    "sales": {
        "description": "Pricing, plans, quotes, demos and new leads.",
        "instructions": (
            "Sales playbook:\n1. Look up the sender; unknown senders are new leads.\n2. Calculate the price with "
            "get_plan_pricing (Enterprise: 15% discount from 50 seats).\n3. Create a sales lead for new "
            "prospects; the assigned account executive schedules demos.\n4. Quote monthly totals with annual billing."
        ),
    },
    "relations": {
        "description": "Complaints, legal threats and data-protection (GDPR) requests.",
        "instructions": (
            "Relations playbook:\n1. Never admit fault, promise compensation or confirm data deletion.\n2. GDPR "
            "requests: escalate to the Data Protection Officer (team 'dpo'); legal threats without GDPR: escalate "
            "to a team lead.\n3. Send only a neutral acknowledgment with the escalation reference and the SLA."
        ),
    },
}
SkillName = Literal["billing", "technical", "sales", "relations"]


def _merge_unique(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Reducer: parallel load_skill calls in one turn must not overwrite each other."""
    return list(dict.fromkeys([*(left or []), *(right or [])]))


class SkillState(AgentState):
    loaded_skills: NotRequired[Annotated[list[str], _merge_unique]]


@tool
def load_skill(skill_name: SkillName, runtime: ToolRuntime) -> Command:
    """Load a skill: returns its playbook and unlocks its tools."""
    skill = SKILLS[skill_name]
    tool_names = ", ".join(t.name for t in DOMAIN_TOOLS[skill_name])
    return Command(
        update={
            "messages": [
                ToolMessage(
                    f"Skill '{skill_name}' loaded.\n\n{skill['instructions']}\n\nUnlocked tools: {tool_names}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "loaded_skills": [skill_name],
        }
    )


@wrap_model_call
def expose_skill_tools(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    tools = {"load_skill": load_skill, "lookup_customer": lookup_customer}
    for name in request.state.get("loaded_skills", []):
        tools.update({t.name: t for t in DOMAIN_TOOLS[name]})
    return handler(request.override(tools=list(tools.values())))


def build_agent(model: BaseChatModel, *, checkpointer=None):
    catalog = "\n".join(f"- {name}: {s['description']}" for name, s in SKILLS.items())
    return create_agent(
        model,
        tools=[*ALL_TOOLS, load_skill],
        system_prompt=f"{prompts.SKILLS_AGENT}\n\nAvailable skills:\n{catalog}",
        state_schema=SkillState,
        middleware=[expose_skill_tools],
        response_format=EmailResolution,
        checkpointer=checkpointer,
        name="service_generalist",
    )


def build_graph(model: BaseChatModel):
    agent = build_agent(model)

    def handle_email(state: EmailState) -> dict:
        result = agent.invoke({"messages": [email_message(state["email"])]})
        return {"resolution": result["structured_response"]}

    return (
        StateGraph(EmailState, input_schema=EmailInput, output_schema=EmailOutput)
        .add_node("agent", handle_email)
        .add_node("dispatch", dispatch)
        .add_edge(START, "agent")
        .add_edge("agent", "dispatch")
        .add_edge("dispatch", END)
        .compile(name="skills")
    )
