"""Pattern 2 - Sequential pipeline, built with `agentpatterns.create_pipeline`."""

import re

from agentpatterns import create_pipeline, function_step, llm_step
from email_assistant import prompts
from email_assistant.common import build_email_workflow
from email_assistant.schemas import ActionPlan, EmailClassification, EmailResolution, QualityReview
from email_assistant.tools import ALL_TOOLS

TOOLS = {t.name: t for t in ALL_TOOLS}
ACTION_TOOL = {
    "refund": "issue_refund",
    "ticket": "create_ticket",
    "lead": "create_sales_lead",
    "escalation": "escalate_to_human",
}


def enrich(state) -> list[dict]:
    """Deterministic retrieval based on the e-mail and its classification."""
    email, classification = state["context"]["email"], state["outputs"]["classify"]
    text = f"{email.subject} {email.body}".lower()
    calls = [("lookup_customer", {"email": email.sender})]
    calls += [("get_invoice", {"invoice_id": i}) for i in dict.fromkeys(re.findall(r"INV-\d{4}-\d{4}", email.body))]
    calls += [("check_service_status", {"service": s}) for s in ("dashboard", "export", "api") if s in text]
    if "sales" in classification.categories:
        seats = re.search(r"(\d+)\s*(?:seats|users)", text)
        plan = "enterprise" if "enterprise" in text else "business"
        calls.append(("get_plan_pricing", {"plan": plan, "seats": int(seats.group(1)) if seats else 10}))
    calls.append(("search_knowledge_base", {"query": f"{email.subject} {classification.summary}"}))
    return [{"tool": t, "args": a, "result": TOOLS[t].invoke(a)} for t, a in calls]


def execute_actions(state) -> list[dict]:
    """Code executes what the model proposed; tools enforce policy."""
    executed = []
    for action in state["outputs"]["plan_actions"].actions:
        tool, args = ACTION_TOOL[action.kind], action.model_dump(exclude={"kind"})
        executed.append({"tool": tool, "args": args, "result": TOOLS[tool].invoke(args)})
    return executed


def ignore_spam(state) -> EmailResolution:
    return EmailResolution(
        email_id=state["context"]["email"].id,
        category="spam",
        priority="low",
        action="ignore",
        internal_note="Spam / phishing - no reply sent.",
    )


def escalate_failed_review(state) -> EmailResolution:
    issues = "; ".join(state["outputs"]["review"].issues)
    return state["outputs"]["write_reply"].model_copy(
        update={"action": "escalate", "escalation_reason": f"Quality review failed: {issues}"}
    )


def build_graph(model):
    pipeline = create_pipeline(
        [
            llm_step(
                "classify",
                model,
                system_prompt=prompts.CLASSIFIER,
                output_schema=EmailClassification,
                gate=lambda s: s["outputs"]["classify"].category != "spam",
                on_gate_fail=ignore_spam,
            ),
            function_step("enrich", enrich),
            llm_step("plan_actions", model, system_prompt=prompts.ACTION_PLANNER, output_schema=ActionPlan),
            function_step("execute_actions", execute_actions),
            llm_step("write_reply", model, system_prompt=prompts.REPLY_WRITER, output_schema=EmailResolution),
            llm_step(
                "review",
                model,
                system_prompt=prompts.REVIEWER,
                output_schema=QualityReview,
                gate=lambda s: s["outputs"]["review"].passed,
                on_gate_fail=escalate_failed_review,
            ),
        ],
        output="write_reply",
        name="email_pipeline",
    )
    return build_email_workflow(pipeline, "sequential", extra_input=lambda s: {"context": {"email": s["email"]}})
