"""Pattern 7 - Hierarchical teams (supervisor of supervisors), native implementation.

The subagents-as-tools pattern, nested: a top-level inbox manager delegates
to team leads, each team lead is itself a supervisor over its specialists.

    inbox_manager ──tool──> customer_care_team (lead) ──tool──> billing_specialist
                  │                                  └─tool──> technical_specialist
                  └─tool──> accounts_team (lead)     ──tool──> sales_specialist
                                                     └─tool──> relations_specialist
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from email_assistant import prompts
from email_assistant.common import EmailInput, EmailOutput, EmailState, dispatch, email_message
from email_assistant.native.supervisor import make_specialist_tool
from email_assistant.schemas import EmailResolution, TeamReport

TEAMS = {
    "customer_care": {
        "members": ["billing", "technical"],
        "description": "Customer care team: billing issues (invoices, refunds) and technical support.",
    },
    "accounts": {
        "members": ["sales", "relations"],
        "description": "Accounts team: sales inquiries and customer relations (complaints, legal, GDPR).",
    },
}


def make_team_tool(model: BaseChatModel, team: str):
    lead = create_agent(
        model,
        tools=[make_specialist_tool(model, d) for d in TEAMS[team]["members"]],
        system_prompt=prompts.TEAM_LEADS[team],
        response_format=TeamReport,
        name=f"{team}_lead",
    )

    @tool(f"{team}_team", description=TEAMS[team]["description"])
    def call_team(task: str) -> str:
        """Delegate a self-contained task (quote the e-mail) to the team."""
        result = lead.invoke({"messages": [HumanMessage(task)]})
        return result["structured_response"].model_dump_json()

    return call_team


def build_graph(model: BaseChatModel):
    manager = create_agent(
        model,
        tools=[make_team_tool(model, team) for team in TEAMS],
        system_prompt=prompts.INBOX_MANAGER,
        response_format=EmailResolution,
        name="inbox_manager",
    )

    def handle_email(state: EmailState) -> dict:
        result = manager.invoke({"messages": [email_message(state["email"])]})
        return {"resolution": result["structured_response"]}

    return (
        StateGraph(EmailState, input_schema=EmailInput, output_schema=EmailOutput)
        .add_node("inbox_manager", handle_email)
        .add_node("dispatch", dispatch)
        .add_edge(START, "inbox_manager")
        .add_edge("inbox_manager", "dispatch")
        .add_edge("dispatch", END)
        .compile(name="hierarchical")
    )
