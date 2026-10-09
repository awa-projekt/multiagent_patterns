"""Pattern 1 - Single agent, built with `agentpatterns.create_single_agent`."""

from agentpatterns import create_single_agent
from email_assistant import prompts
from email_assistant.common import build_email_workflow
from email_assistant.schemas import EmailResolution
from email_assistant.tools import ALL_TOOLS


def build_graph(model):
    agent = create_single_agent(
        model,
        ALL_TOOLS,
        system_prompt=prompts.SINGLE_AGENT,
        response_format=EmailResolution,
        max_model_calls=15,
        name="customer_service_agent",
    )
    return build_email_workflow(agent, "single_agent")
