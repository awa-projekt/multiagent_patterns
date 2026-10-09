"""Pattern 4 - Parallelization, built with `create_parallel` - composed into a `create_pipeline`.

pipeline: triage (= create_parallel over 4 analysts) -[spam]-> ignore
                                                     -[else]-> respond (agent)
"""

from langchain.agents import create_agent

from agentpatterns import AgentSpec, agent_step, create_parallel, create_pipeline
from email_assistant import prompts
from email_assistant.common import build_email_workflow
from email_assistant.schemas import (
    ComplianceScreen,
    EmailClassification,
    EmailResolution,
    ExtractedEntities,
    SentimentAnalysis,
)
from email_assistant.tools import ALL_TOOLS

PRIORITY_ORDER = ["low", "normal", "high", "urgent"]


def aggregate(results, state) -> dict:
    """Fan-in in plain code: merge the four structured analyses into a triage report."""
    intent, sentiment = results["intent"].structured, results["sentiment"].structured
    entities, compliance = results["entities"].structured, results["compliance"].structured
    return {
        "category": "spam" if compliance.phishing else intent.category,
        "additional_categories": intent.additional_categories,
        "priority": max(intent.priority, sentiment.urgency, key=PRIORITY_ORDER.index),
        "sentiment": sentiment.sentiment,
        "needs_human": intent.needs_human or compliance.legal_threat or compliance.data_protection_request,
        "summary": intent.summary,
        "signals": sentiment.signals,
        "entities": entities.model_dump(),
        "compliance": compliance.model_dump(),
    }


def build_graph(model):
    def analyst(name, schema, system_prompt):
        # An agent without tools + response_format = one structured LLM call.
        return AgentSpec(
            name,
            system_prompt.split(".")[0],
            create_agent(model, tools=[], system_prompt=system_prompt, response_format=schema, name=name),
        )

    triage = create_parallel(
        [
            analyst("intent", EmailClassification, prompts.CLASSIFIER),
            analyst("sentiment", SentimentAnalysis, prompts.SENTIMENT_ANALYST),
            analyst("entities", ExtractedEntities, prompts.ENTITY_EXTRACTOR),
            analyst("compliance", ComplianceScreen, prompts.COMPLIANCE_SCREENER),
        ],
        aggregator=aggregate,
        name="parallel_triage",
    )
    responder = create_agent(
        model, tools=ALL_TOOLS, system_prompt=prompts.RESPONDER, response_format=EmailResolution, name="responder"
    )

    def ignore_spam(state):
        email = state["context"]["email"]
        return EmailResolution(
            email_id=email.id,
            category="spam",
            priority="low",
            action="ignore",
            internal_note="Spam / phishing - no reply sent.",
        )

    pipeline = create_pipeline(
        [
            agent_step(
                "triage", triage, gate=lambda s: s["outputs"]["triage"]["category"] != "spam", on_gate_fail=ignore_spam
            ),
            agent_step("respond", responder),
        ],
        name="parallel_email_flow",
    )
    return build_email_workflow(pipeline, "parallel", extra_input=lambda s: {"context": {"email": s["email"]}})
