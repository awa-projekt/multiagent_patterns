"""Pattern 11 - Evaluator-optimizer, built with `create_evaluator_optimizer`."""

from langchain.agents import create_agent

from agentpatterns import create_evaluator_optimizer
from email_assistant import prompts
from email_assistant.common import build_email_workflow
from email_assistant.schemas import EmailResolution, QualityReview
from email_assistant.tools import ALL_TOOLS


def escalate(draft: EmailResolution, review: QualityReview) -> EmailResolution:
    """Never send a draft that failed review: hand it to a human instead."""
    return draft.model_copy(
        update={"action": "escalate", "escalation_reason": "Draft failed review: " + "; ".join(review.issues)}
    )


def build_graph(model):
    drafter = create_agent(
        model, tools=ALL_TOOLS, system_prompt=prompts.DRAFTER, response_format=EmailResolution, name="reply_drafter"
    )
    loop = create_evaluator_optimizer(
        drafter,
        model,
        evaluator_prompt=prompts.REVIEWER,
        evaluation_schema=QualityReview,
        max_iterations=3,
        on_max_iterations=escalate,
        name="reflection_loop",
    )
    return build_email_workflow(loop, "evaluator_optimizer")
