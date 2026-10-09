"""Pattern 6 - Supervisor with subagents-as-tools, native implementation.

This is the pattern the LangChain v1 docs recommend for supervisors (it
replaces the unmaintained `langgraph-supervisor` package). The supervisor is a
normal `create_agent`; each specialist is another `create_agent` wrapped in a
`@tool`. The supervisor decides whom to call, with what task, possibly several
in parallel, and merges the results. Specialists are stateless and work in a
clean context window (context isolation).

    START -> supervisor(agent) --tool--> billing_specialist(agent)
                               --tool--> technical_specialist(agent)   (parallel tool calls)
                               --tool--> sales_specialist(agent)
                               --tool--> relations_specialist(agent)
             -> dispatch -> END
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from email_assistant import prompts
from email_assistant.common import EmailInput, EmailOutput, EmailState, dispatch, email_message
from email_assistant.schemas import EmailResolution, SpecialistReport
from email_assistant.tools import DOMAIN_TOOLS


def make_specialist_tool(model: BaseChatModel, domain: str):
    """Wrap a specialist agent as a tool the supervisor can call."""
    specialist = create_agent(
        model,
        tools=DOMAIN_TOOLS[domain],
        system_prompt=prompts.SPECIALISTS[domain],
        response_format=SpecialistReport,
        name=f"{domain}_specialist",
    )

    @tool(f"{domain}_specialist", description=prompts.SPECIALIST_DESCRIPTIONS[domain])
    def call_specialist(task: str) -> str:
        """Delegate a self-contained task (quote the e-mail) to the specialist."""
        # Input: only the task -> the subagent starts with a clean context.
        result = specialist.invoke({"messages": [HumanMessage(task)]})
        # Output: only the structured report goes back to the supervisor,
        # not the specialist's internal tool calls.
        return result["structured_response"].model_dump_json()

    return call_specialist


def build_supervisor_agent(model: BaseChatModel):
    return create_agent(
        model,
        tools=[make_specialist_tool(model, d) for d in DOMAIN_TOOLS],
        system_prompt=prompts.SUPERVISOR,
        response_format=EmailResolution,
        name="supervisor",
    )


def build_graph(model: BaseChatModel):
    supervisor = build_supervisor_agent(model)

    def handle_email(state: EmailState) -> dict:
        result = supervisor.invoke({"messages": [email_message(state["email"])]})
        return {"resolution": result["structured_response"]}

    return (
        StateGraph(EmailState, input_schema=EmailInput, output_schema=EmailOutput)
        .add_node("supervisor", handle_email)
        .add_node("dispatch", dispatch)
        .add_edge(START, "supervisor")
        .add_edge("supervisor", "dispatch")
        .add_edge("dispatch", END)
        .compile(name="supervisor")
    )
