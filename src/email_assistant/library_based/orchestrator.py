"""Pattern 5 - Orchestrator-workers with re-planning, built with `create_orchestrator`."""

from langchain.agents import create_agent

from agentpatterns import AgentSpec, create_orchestrator
from email_assistant import prompts
from email_assistant.common import build_email_workflow
from email_assistant.native.orchestrator import WORKER_TOOLS
from email_assistant.schemas import EmailResolution

DESCRIPTIONS = {
    "account_researcher": "CRM and billing lookups (lookup_customer, get_invoice).",
    "knowledge_researcher": "Knowledge base, status page and pricing lookups.",
    "operations": "Executes actions with side effects (refunds, tickets, leads, escalations).",
}


def build_graph(model):
    workers = [
        AgentSpec(
            name,
            DESCRIPTIONS[name],
            create_agent(model, tools=tools, system_prompt=prompts.WORKERS[name] + prompts.WORKER_SUFFIX, name=name),
        )
        for name, tools in WORKER_TOOLS.items()
    ]
    orchestrator = create_orchestrator(
        model,
        workers,
        planner_prompt=prompts.PLANNER,
        synthesizer_prompt=prompts.REPLY_WRITER,
        response_format=EmailResolution,
        max_rounds=3,
        name="case_orchestrator",
    )
    return build_email_workflow(orchestrator, "orchestrator")
