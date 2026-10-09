"""Pattern 8 - Swarm with handoffs, built with `create_swarm` + `SwarmAgent`."""

from agentpatterns import SwarmAgent, create_swarm
from email_assistant import prompts
from email_assistant.common import build_email_workflow
from email_assistant.schemas import EmailResolution
from email_assistant.tools import DOMAIN_TOOLS


def build_swarm(model, **kwargs):
    specialists = [
        SwarmAgent(
            f"{domain}_specialist",
            prompts.SPECIALIST_DESCRIPTIONS[domain],
            prompts.SWARM_SPECIALISTS[domain],
            tools=tools,
        )
        for domain, tools in DOMAIN_TOOLS.items()
    ]
    front_desk = SwarmAgent(
        "front_desk",
        "Reads new e-mails and passes them to the right specialist.",
        prompts.FRONT_DESK,
        handoffs=[s.name for s in specialists],
    )
    return create_swarm(
        model,
        [front_desk, *specialists],
        default_agent="front_desk",
        response_format=EmailResolution,
        name="service_swarm",
        **kwargs,
    )


def build_graph(model):
    return build_email_workflow(build_swarm(model), "swarm")
