"""agentpatterns - reusable multi-agent patterns for LangGraph / LangChain v1.

Every factory returns a compiled LangGraph graph that follows the *agent
contract* (`{"messages"}` in, `{"messages", "structured_response"}` out), so
patterns can be nested, swapped, embedded as subgraphs or exposed as tools.

| Pattern                | Factory                                                      |
|------------------------|--------------------------------------------------------------|
| Single agent           | `create_single_agent`                                        |
| Sequential pipeline    | `create_pipeline` + `llm_step` / `agent_step` / `function_step` |
| Router                 | `create_router`                                              |
| Parallelization        | `create_parallel`, `create_voting`, `create_map_reduce`      |
| Orchestrator-workers   | `create_orchestrator`                                        |
| Supervisor (subagents) | `create_supervisor`                                          |
| Hierarchical teams     | `create_hierarchy` + `Team`                                  |
| Swarm (handoffs)       | `create_swarm` + `SwarmAgent`                                |
| State machine          | `create_state_machine_agent` + `Step` / `StateMachineMiddleware` |
| Skills                 | `create_skills_agent` + `Skill` / `SkillsMiddleware`         |
| Evaluator-optimizer    | `create_evaluator_optimizer`                                 |

Limits: `max_model_calls` / `max_tool_calls` / `on_limit` on every agent factory
(`loop_limits` for your own `create_agent`), per-participant caps
(`max_calls_per_agent`, `SwarmAgent.max_activations`, `Step.max_visits`) and
`RunBudget` for a whole run - see `agentpatterns.limits`.

Messaging: `MessagingMiddleware` lets the agents of a run message each other,
with a `Mailbox` per run - see `agentpatterns.messaging`.
"""

from agentpatterns.core import (
    AgentResult,
    AgentSpec,
    PatternInput,
    PatternOutput,
    PatternState,
    StructuredOutputMethod,
    agent_as_node,
    agent_as_tool,
    final_text,
    invoke_agent,
    make_serializer,
    results_block,
    structured_llm,
)
from agentpatterns.evaluator_optimizer import Evaluation, create_evaluator_optimizer
from agentpatterns.hierarchical import Team, build_team, create_hierarchy
from agentpatterns.limits import BudgetExceededError, ModelCallLimitExceededError, OnLimit, RunBudget, loop_limits
from agentpatterns.messaging import AgentMessage, Mailbox, MessagingMiddleware
from agentpatterns.orchestrator import OrchestratorInput, create_orchestrator
from agentpatterns.parallel import create_map_reduce, create_parallel, create_voting
from agentpatterns.router import RouteTask, create_router
from agentpatterns.sequential import Step as PipelineStep
from agentpatterns.sequential import agent_step, create_pipeline, function_step, llm_step
from agentpatterns.single_agent import create_single_agent
from agentpatterns.skills import Skill, SkillsMiddleware, create_skills_agent
from agentpatterns.state_machine import (
    StateMachineMiddleware,
    StateMachineState,
    Step,
    create_state_machine_agent,
    create_transition_tool,
    transition,
)
from agentpatterns.supervisor import DelegationLimitMiddleware, create_supervisor, create_task_tool
from agentpatterns.swarm import SwarmAgent, SwarmState, create_handoff_tool, create_swarm

__all__ = [
    "AgentMessage",
    "AgentResult",
    "AgentSpec",
    "BudgetExceededError",
    "DelegationLimitMiddleware",
    "Evaluation",
    "Mailbox",
    "MessagingMiddleware",
    "ModelCallLimitExceededError",
    "OnLimit",
    "OrchestratorInput",
    "PatternInput",
    "PatternOutput",
    "PatternState",
    "PipelineStep",
    "RouteTask",
    "RunBudget",
    "Skill",
    "SkillsMiddleware",
    "StateMachineMiddleware",
    "StateMachineState",
    "Step",
    "StructuredOutputMethod",
    "SwarmAgent",
    "SwarmState",
    "Team",
    "agent_as_node",
    "agent_as_tool",
    "agent_step",
    "build_team",
    "create_evaluator_optimizer",
    "create_handoff_tool",
    "create_hierarchy",
    "create_map_reduce",
    "create_orchestrator",
    "create_parallel",
    "create_pipeline",
    "create_router",
    "create_single_agent",
    "create_skills_agent",
    "create_state_machine_agent",
    "create_supervisor",
    "create_swarm",
    "create_task_tool",
    "create_transition_tool",
    "create_voting",
    "final_text",
    "function_step",
    "invoke_agent",
    "llm_step",
    "loop_limits",
    "make_serializer",
    "results_block",
    "structured_llm",
    "transition",
]
