"""Hierarchical teams - supervisors of supervisors.

Describe the organisation as a tree of `Team`s; leaves are `AgentSpec`s.
Every team becomes a supervisor (see `supervisor.py`) that is exposed to its
parent as a tool, so the top-level agent only sees a handful of team tools
and each team lead only sees its own members.

    hierarchy = create_hierarchy(
        model,
        teams=[
            Team("support", "Billing and technical issues", system_prompt="You lead support...",
                 members=[billing_spec, tech_spec]),
            Team("growth", "Sales and partnerships", system_prompt="You lead growth...",
                 members=[sales_spec, Team("partners", "...", system_prompt="...", members=[...])]),
        ],
        system_prompt="You are the head of customer operations...",
        response_format=Answer,
    )

Tip: only the outermost graph needs a checkpointer; interrupts raised deep in
the tree propagate to the top.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from agentpatterns.core import AgentSpec
from agentpatterns.supervisor import create_supervisor

__all__ = ["Team", "build_team", "create_hierarchy"]

# Supervisor settings that apply to every team lead, not only the top level.
TEAM_KWARGS = {"delegation", "input_mode", "max_model_calls", "max_tool_calls", "on_limit", "max_calls_per_agent"}


@dataclass
class Team:
    """A team lead (supervisor) with members (agents or sub-teams)."""

    name: str
    description: str
    system_prompt: str
    members: Sequence[AgentSpec | Team] = field(default_factory=list)
    response_format: Any = None
    """Structured report the team lead returns to its parent (optional)."""
    model: BaseChatModel | None = None
    """Model for this team lead (defaults to the hierarchy's model)."""


def _names(members: Sequence[AgentSpec | Team]) -> Iterator[str]:
    for member in members:
        yield member.name
        if isinstance(member, Team):
            yield from _names(member.members)


def _for_members(supervisor_kwargs: dict[str, Any], members: Sequence[AgentSpec]) -> dict[str, Any]:
    """A `max_calls_per_agent` mapping applies to the members of each level."""
    limits = supervisor_kwargs.get("max_calls_per_agent")
    if not isinstance(limits, Mapping):
        return supervisor_kwargs
    names = {m.name for m in members}
    return {**supervisor_kwargs, "max_calls_per_agent": {k: v for k, v in limits.items() if k in names}}


def build_team(model: BaseChatModel, team: Team, **supervisor_kwargs: Any) -> AgentSpec:
    """Recursively turn a `Team` into an `AgentSpec` whose agent is a supervisor."""
    members = [build_team(model, m, **supervisor_kwargs) if isinstance(m, Team) else m for m in team.members]
    lead = create_supervisor(
        team.model or model,
        members,
        system_prompt=team.system_prompt,
        response_format=team.response_format,
        name=f"{team.name}_lead",
        **_for_members(supervisor_kwargs, members),
    )
    return AgentSpec(team.name, team.description, lead)


def create_hierarchy(
    model: BaseChatModel,
    teams: Sequence[Team | AgentSpec],
    *,
    system_prompt: str,
    response_format: Any = None,
    checkpointer: Checkpointer = None,
    name: str = "hierarchy",
    **supervisor_kwargs: Any,
) -> CompiledStateGraph:
    """Build the top-level supervisor over teams (and optionally individual agents).

    `delegation`, `input_mode` and the limits (`max_model_calls`, `max_tool_calls`,
    `on_limit`, `max_calls_per_agent`) apply to every supervisor in the tree; a
    `max_calls_per_agent` mapping may name members of any level. Everything else
    (`middleware`, ...) goes to the top level only, like `checkpointer`.
    """
    limits = supervisor_kwargs.get("max_calls_per_agent")
    if isinstance(limits, Mapping) and (unknown := set(limits) - set(_names(teams))):
        raise ValueError(f"max_calls_per_agent names unknown members: {sorted(unknown)}")
    team_kwargs = {k: v for k, v in supervisor_kwargs.items() if k in TEAM_KWARGS}
    members = [build_team(model, t, **team_kwargs) if isinstance(t, Team) else t for t in teams]
    return create_supervisor(
        model,
        members,
        system_prompt=system_prompt,
        response_format=response_format,
        checkpointer=checkpointer,
        name=name,
        **_for_members(supervisor_kwargs, members),
    )
