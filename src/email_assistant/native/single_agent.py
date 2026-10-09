"""Pattern 1 - Single agent (baseline), native LangGraph/LangChain implementation.

One tool-calling agent owns the whole e-mail: it sees all tools, loops
"think -> call tools -> observe" and ends with a structured `EmailResolution`.

    START -> agent (create_agent, all 9 tools, response_format) -> dispatch -> END
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from email_assistant import prompts
from email_assistant.common import EmailInput, EmailOutput, EmailState, dispatch, email_message
from email_assistant.schemas import EmailResolution
from email_assistant.tools import ALL_TOOLS


def build_graph(model: BaseChatModel):
    agent = create_agent(
        model,
        tools=ALL_TOOLS,
        system_prompt=prompts.SINGLE_AGENT,
        # Structured output: a bare schema lets LangChain pick ProviderStrategy
        # for models with native support and ToolStrategy otherwise.
        response_format=EmailResolution,
        # Guardrail against runaway loops (default recursion limit is 1000).
        middleware=[ModelCallLimitMiddleware(run_limit=15, exit_behavior="error")],
        name="customer_service_agent",
    )

    def handle_email(state: EmailState) -> dict:
        # The agent speaks "messages", the workflow speaks "email/resolution":
        # a small node function maps between the two schemas.
        result = agent.invoke({"messages": [email_message(state["email"])]})
        return {"resolution": result["structured_response"]}

    return (
        StateGraph(EmailState, input_schema=EmailInput, output_schema=EmailOutput)
        .add_node("agent", handle_email)
        .add_node("dispatch", dispatch)
        .add_edge(START, "agent")
        .add_edge("agent", "dispatch")
        .add_edge("dispatch", END)
        .compile(name="single_agent")
    )
