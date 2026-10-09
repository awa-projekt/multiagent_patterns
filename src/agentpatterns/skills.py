"""Skills - progressive disclosure of prompts and tools for a single agent.

The agent only sees a catalog of skill *names and descriptions*. Calling
`load_skill(name)` returns the skill's full instructions as a tool result and
unlocks the skill's tools for the rest of the conversation. Context stays
small until a capability is actually needed, and teams can own skills
independently.

    agent = create_skills_agent(
        model,
        skills=[Skill("sql", "Write SQL for our warehouse", instructions=SQL_GUIDE, tools=[run_query]),
                Skill("legal", "Review contracts", instructions=LEGAL_GUIDE)],
        system_prompt="You are a data assistant...",
    )

`SkillsMiddleware` can be added to any existing `create_agent`, too.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer, Command

from agentpatterns.core import literal_of
from agentpatterns.limits import OnLimit, loop_limits

__all__ = ["Skill", "SkillsMiddleware", "SkillsState", "create_skills_agent"]


@dataclass
class Skill:
    name: str
    description: str
    """Shown in the catalog - the only thing the model sees before loading."""
    instructions: str
    """Full playbook, returned by `load_skill`."""
    tools: Sequence[BaseTool] = ()
    """Tools unlocked when the skill is loaded."""


def _merge_unique(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Reducer: several `load_skill` calls in one turn must not overwrite each other."""
    return list(dict.fromkeys([*(left or []), *(right or [])]))


class SkillsState(AgentState):
    loaded_skills: NotRequired[Annotated[list[str], _merge_unique]]


class SkillsMiddleware(AgentMiddleware):
    """Adds `load_skill`, a skill catalog in the system prompt, and hides tools of unloaded skills."""

    state_schema = SkillsState

    def __init__(
        self, skills: Sequence[Skill], *, always_available: Sequence[str] = (), show_catalog: bool = True
    ) -> None:
        """`always_available`: tool names that stay visible even if they also belong to a skill."""
        super().__init__()
        self.skills = {s.name: s for s in skills}
        self.always_available = set(always_available)
        self.show_catalog = show_catalog
        self.load_tool = self._make_load_tool()
        registered: dict[str, BaseTool] = {self.load_tool.name: self.load_tool}
        for skill in skills:
            for tool in skill.tools:
                registered.setdefault(tool.name, tool)
        self.tools = list(registered.values())

    def catalog(self) -> str:
        return "\n".join(f"- {s.name}: {s.description}" for s in self.skills.values())

    def _make_load_tool(self) -> BaseTool:
        skills = self.skills

        def load_skill(skill_name: str, runtime: ToolRuntime[Any, Any]) -> Command:
            """Load a skill: returns its instructions and unlocks its tools.

            Args:
                skill_name: Name of the skill to load.
            """
            skill = skills[skill_name]
            unlocked = ", ".join(t.name for t in skill.tools) or "none"
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            f"Skill '{skill.name}' loaded.\n\n{skill.instructions}\n\nUnlocked tools: {unlocked}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                    "loaded_skills": [skill.name],
                }
            )

        load_skill.__annotations__ = {
            "skill_name": literal_of(list(skills)),
            "runtime": ToolRuntime[Any, Any],
            "return": Command,
        }
        return StructuredTool.from_function(
            func=load_skill,
            name="load_skill",
            description="Load a skill before working on a matching request.",
            parse_docstring=True,
        )

    def _configure(self, request: ModelRequest) -> ModelRequest:
        loaded = set(request.state.get("loaded_skills") or [])
        visible = {t.name for name in loaded for t in self.skills[name].tools}
        hidden = {t.name for s in self.skills.values() for t in s.tools} - visible - self.always_available
        tools = [t for t in request.tools if getattr(t, "name", None) not in hidden]
        overrides: dict[str, Any] = {"tools": tools}
        if self.show_catalog:
            base = request.system_message.text if request.system_message else ""
            overrides["system_message"] = SystemMessage(
                f"{base}\n\nAvailable skills (load with load_skill):\n{self.catalog()}".strip()
            )
        return request.override(**overrides)

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        return handler(self._configure(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        return await handler(self._configure(request))


def create_skills_agent(
    model: BaseChatModel | str,
    skills: Sequence[Skill],
    *,
    system_prompt: str,
    tools: Sequence[BaseTool] = (),
    response_format: Any = None,
    max_model_calls: int | None = 25,
    max_tool_calls: int | None = None,
    on_limit: OnLimit = "error",
    middleware: Sequence[AgentMiddleware] = (),
    checkpointer: Checkpointer = None,
    name: str = "skills_agent",
    **agent_kwargs: Any,
) -> CompiledStateGraph:
    """Create a single agent with on-demand skills.

    `tools` are always available; skill tools appear after their skill is loaded.
    `max_model_calls` / `max_tool_calls` / `on_limit` cap the loop (see `agentpatterns.limits`).
    `checkpointer` checkpoints the agent; only the outermost graph needs one.
    """
    guards = loop_limits(max_model_calls=max_model_calls, max_tool_calls=max_tool_calls, on_limit=on_limit)
    return create_agent(
        model,
        tools=list(tools),
        system_prompt=system_prompt,
        middleware=[*guards, SkillsMiddleware(skills, always_available=[t.name for t in tools]), *middleware],
        response_format=response_format,
        checkpointer=checkpointer,
        name=name,
        **agent_kwargs,
    )
