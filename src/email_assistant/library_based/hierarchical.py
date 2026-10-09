"""Pattern 7 - Hierarchical teams, built with `create_hierarchy` + `Team`."""

from agentpatterns import Team, create_hierarchy
from email_assistant import prompts
from email_assistant.common import build_email_workflow
from email_assistant.library_based.specialists import specialist_specs
from email_assistant.schemas import EmailResolution, TeamReport


def build_graph(model):
    hierarchy = create_hierarchy(
        model,
        teams=[
            Team(
                "customer_care_team",
                "Customer care team: billing issues and technical support.",
                system_prompt=prompts.TEAM_LEADS["customer_care"],
                members=specialist_specs(model, ["billing", "technical"]),
                response_format=TeamReport,
            ),
            Team(
                "accounts_team",
                "Accounts team: sales and customer relations (complaints, legal, GDPR).",
                system_prompt=prompts.TEAM_LEADS["accounts"],
                members=specialist_specs(model, ["sales", "relations"]),
                response_format=TeamReport,
            ),
        ],
        system_prompt=prompts.INBOX_MANAGER,
        response_format=EmailResolution,
        name="inbox_manager",
    )
    return build_email_workflow(hierarchy, "hierarchical")
