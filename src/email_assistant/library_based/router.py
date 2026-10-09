"""Pattern 3 - Router, built with `agentpatterns.create_router`."""

from agentpatterns import create_router
from email_assistant import prompts
from email_assistant.common import build_email_workflow
from email_assistant.library_based.specialists import specialist_specs
from email_assistant.schemas import EmailResolution


def build_graph(model):
    router = create_router(
        model,
        specialist_specs(model),
        system_prompt=prompts.ROUTER,
        synthesizer_prompt=prompts.REPLY_WRITER,
        response_format=EmailResolution,
        name="inbox_router",
    )
    return build_email_workflow(router, "router")
