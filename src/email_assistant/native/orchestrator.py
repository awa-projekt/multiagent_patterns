"""Pattern 5 - Orchestrator-workers (with re-planning), native implementation.

An orchestrator LLM decomposes the case into tasks *at runtime* (structured
output), `Send` spawns one worker per task in parallel, and the results flow
back to the orchestrator which may plan another round (plan-and-execute).
When the plan is empty, a synthesizer writes the final resolution.

    START -> plan(LLM) --Send--> account_researcher   --\\
                   ^   --Send--> knowledge_researcher ---+--> plan (next round)
                   |   --Send--> operations           --/
                   +-- (empty plan or max rounds) --> synthesize(LLM) -> dispatch -> END
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from email_assistant import prompts
from email_assistant.common import EmailInput, EmailOutput, EmailState, dispatch
from email_assistant.schemas import EmailResolution
from email_assistant.tools import (
    ACTION_TOOLS,
    check_service_status,
    get_invoice,
    get_plan_pricing,
    lookup_customer,
    search_knowledge_base,
)

MAX_ROUNDS = 3
WorkerName = Literal["account_researcher", "knowledge_researcher", "operations"]
WORKER_TOOLS = {
    "account_researcher": [lookup_customer, get_invoice],
    "knowledge_researcher": [search_knowledge_base, check_service_status, get_plan_pricing],
    "operations": ACTION_TOOLS,
}


class PlannedTask(BaseModel):
    worker: WorkerName
    instruction: str = Field(description="Explicit instructions for the worker.")


class Plan(BaseModel):
    """Tasks for the next round. An empty list means the case is fully handled."""

    tasks: list[PlannedTask] = Field(default_factory=list)


class OrchestratorState(EmailState):
    round: NotRequired[int]
    plan: NotRequired[Plan]
    results: Annotated[list[dict[str, Any]], operator.add]


class WorkerInput(TypedDict):
    instruction: str


def _results_block(results: list[dict[str, Any]]) -> str:
    if not results:
        return "(no results yet)"
    return "\n\n".join(f"[{r['worker']}] {r['instruction']}\n-> {r['output']}" for r in results)


def build_graph(model: BaseChatModel):
    planner = model.with_structured_output(Plan)
    writer = model.with_structured_output(EmailResolution)

    def plan(state: OrchestratorState) -> dict:
        prompt = f"{state['email'].as_prompt()}\n\nResults so far:\n{_results_block(state.get('results', []))}"
        new_plan = planner.invoke([SystemMessage(prompts.PLANNER), HumanMessage(prompt)])
        return {"plan": new_plan, "round": state.get("round", 0) + 1}

    def assign_workers(state: OrchestratorState) -> list[Send] | str:
        if not state["plan"].tasks or state["round"] > MAX_ROUNDS:
            return "synthesize"
        return [Send(t.worker, {"instruction": t.instruction}) for t in state["plan"].tasks]

    def make_worker(name: str):
        agent = create_agent(
            model, tools=WORKER_TOOLS[name], system_prompt=prompts.WORKERS[name] + prompts.WORKER_SUFFIX, name=name
        )

        def run(state: WorkerInput) -> dict:
            result = agent.invoke({"messages": [HumanMessage(state["instruction"])]})
            output = result["messages"][-1].text
            return {"results": [{"worker": name, "instruction": state["instruction"], "output": output}]}

        return run

    def synthesize(state: OrchestratorState) -> dict:
        prompt = f"{state['email'].as_prompt()}\n\nWork results:\n{_results_block(state.get('results', []))}"
        return {"resolution": writer.invoke([SystemMessage(prompts.REPLY_WRITER), HumanMessage(prompt)])}

    builder = StateGraph(OrchestratorState, input_schema=EmailInput, output_schema=EmailOutput)
    builder.add_node("plan", plan)
    for name in WORKER_TOOLS:
        builder.add_node(name, make_worker(name))
        builder.add_edge(name, "plan")  # results flow back for re-planning
    builder.add_node("synthesize", synthesize)
    builder.add_node("dispatch", dispatch)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", assign_workers, [*WORKER_TOOLS, "synthesize"])
    builder.add_edge("synthesize", "dispatch")
    builder.add_edge("dispatch", END)
    return builder.compile(name="orchestrator_workers")
