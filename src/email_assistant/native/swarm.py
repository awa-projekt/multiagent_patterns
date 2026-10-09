"""Pattern 8 - Swarm / network of peers with handoffs, native implementation.

Agents are nodes of one graph and hand control to each other with handoff
tools that return `Command(goto=<agent>, graph=Command.PARENT)`. There is no
central coordinator: whoever holds control either works, hands off, or ends
the run with the final answer. `active_agent` is stored in state so that a
multi-turn conversation (with a checkpointer) resumes with the last agent.

    START -> (active_agent | front_desk) ⇄ billing_specialist ⇄ technical_specialist
                                          ⇄ sales_specialist   ⇄ relations_specialist -> END

Context engineering: the handoff forwards the full message history (like
`langgraph-swarm`), so the next agent sees what its peers did. See the docs
for the alternative (forward only the handoff pair + a summary note).
"""

from __future__ import annotations

from typing import NotRequired

from langchain.agents import AgentState, create_agent
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from email_assistant import prompts
from email_assistant.common import EmailInput, EmailOutput, EmailState, dispatch, email_message
from email_assistant.schemas import EmailResolution
from email_assistant.tools import DOMAIN_TOOLS

SPECIALISTS = [f"{d}_specialist" for d in DOMAIN_TOOLS]


class SwarmState(AgentState):
    active_agent: NotRequired[str]


def make_handoff_tool(target: str, description: str):
    @tool(f"transfer_to_{target}", description=description)
    def handoff(note: str, runtime: ToolRuntime) -> Command:
        """Transfer the case to a colleague. `note` tells them what is left to do."""
        tool_message = ToolMessage(
            content=f"Transferred to {target}. Note: {note}",
            name=f"transfer_to_{target}",
            tool_call_id=runtime.tool_call_id,  # pairs with the AIMessage tool call
        )
        return Command(
            goto=target,
            graph=Command.PARENT,  # navigate in the swarm graph, not inside the agent
            update={"messages": [*runtime.state["messages"], tool_message], "active_agent": target},
        )

    return handoff


def build_swarm(model: BaseChatModel, *, checkpointer=None):
    handoffs = {
        f"{d}_specialist": make_handoff_tool(f"{d}_specialist", f"Transfer to the {prompts.SPECIALIST_DESCRIPTIONS[d]}")
        for d in DOMAIN_TOOLS
    }
    agents = {
        "front_desk": create_agent(
            model,
            tools=list(handoffs.values()),
            system_prompt=prompts.FRONT_DESK,
            response_format=EmailResolution,
            name="front_desk",
        )
    }
    for domain, tools in DOMAIN_TOOLS.items():
        name = f"{domain}_specialist"
        peers = [h for target, h in handoffs.items() if target != name]
        agents[name] = create_agent(
            model,
            tools=[*tools, *peers],
            system_prompt=prompts.SWARM_SPECIALISTS[domain],
            response_format=EmailResolution,
            name=name,
        )

    builder = StateGraph(SwarmState)
    for name, agent in agents.items():
        # The compiled agent is added directly as a subgraph node (shared `messages` key).
        builder.add_node(name, agent, destinations=tuple(n for n in SPECIALISTS if n != name))
    builder.add_conditional_edges(START, lambda s: s.get("active_agent") or "front_desk", list(agents))
    # No outgoing edges: an agent that answers without handing off ends the run.
    return builder.compile(name="swarm", checkpointer=checkpointer)


def build_graph(model: BaseChatModel):
    swarm = build_swarm(model)

    def handle_email(state: EmailState) -> dict:
        result = swarm.invoke({"messages": [email_message(state["email"])]})
        return {"resolution": result["structured_response"]}

    return (
        StateGraph(EmailState, input_schema=EmailInput, output_schema=EmailOutput)
        .add_node("swarm", handle_email)
        .add_node("dispatch", dispatch)
        .add_edge(START, "swarm")
        .add_edge("swarm", "dispatch")
        .add_edge("dispatch", END)
        .compile(name="swarm_workflow")
    )
