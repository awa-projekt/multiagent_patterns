"""Native (hand-written) LangGraph implementations of every pattern for the e-mail use case.

Each module exposes `build_graph(model) -> CompiledStateGraph` with the shared
contract `{"email": Email} -> {"resolution": EmailResolution, "delivery": Delivery}`.
"""

from email_assistant.native import (
    evaluator_optimizer,
    hierarchical,
    orchestrator,
    parallel,
    router,
    sequential,
    single_agent,
    skills,
    state_machine,
    supervisor,
    swarm,
)

PATTERNS = {
    "single_agent": single_agent.build_graph,
    "sequential": sequential.build_graph,
    "router": router.build_graph,
    "parallel": parallel.build_graph,
    "orchestrator": orchestrator.build_graph,
    "supervisor": supervisor.build_graph,
    "hierarchical": hierarchical.build_graph,
    "swarm": swarm.build_graph,
    "state_machine": state_machine.build_graph,
    "skills": skills.build_graph,
    "evaluator_optimizer": evaluator_optimizer.build_graph,
}

__all__ = ["PATTERNS"]
