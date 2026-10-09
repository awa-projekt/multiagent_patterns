"""Pattern 6 - Supervisor with subagents as tools, built with `create_supervisor`."""

from agentpatterns import create_supervisor
from email_assistant import prompts
from email_assistant.common import build_email_workflow
from email_assistant.library_based.specialists import specialist_specs
from email_assistant.schemas import EmailResolution


def build_graph(model, *, delegation="tool_per_agent"):
    supervisor = create_supervisor(
        model,
        specialist_specs(model),
        system_prompt=prompts.SUPERVISOR,
        response_format=EmailResolution,
        delegation=delegation,
        name="supervisor",
    )
    return build_email_workflow(supervisor, "supervisor")
