"""The e-mail use case rebuilt with the `agentpatterns` library (same contract as `native`)."""

from email_assistant.library_based import (
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
