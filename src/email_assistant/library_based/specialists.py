"""Specialist agents shared by several library-based workflows."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from agentpatterns import AgentSpec
from email_assistant import prompts
from email_assistant.schemas import SpecialistReport
from email_assistant.tools import DOMAIN_TOOLS


def specialist_specs(model: BaseChatModel, domains: list[str] | None = None) -> list[AgentSpec]:
    """One `AgentSpec` per domain: a tool-calling agent returning a `SpecialistReport`."""
    return [
        AgentSpec(
            name=f"{domain}_specialist",
            description=prompts.SPECIALIST_DESCRIPTIONS[domain],
            agent=create_agent(
                model,
                tools=DOMAIN_TOOLS[domain],
                system_prompt=prompts.SPECIALISTS[domain],
                response_format=SpecialistReport,
                name=f"{domain}_specialist",
            ),
        )
        for domain in (domains or list(DOMAIN_TOOLS))
    ]
