"""Pattern 11 - Evaluator-optimizer (reflection loop), native implementation.

A generator (the drafter agent) produces a resolution, an evaluator (LLM with
structured output) checks it against explicit criteria and either accepts it
or returns feedback. The loop repeats until the draft passes or the iteration
cap is hit - then a human takes over instead of sending a bad reply.

    START -> draft(agent) -> evaluate(LLM) -[passed]-> dispatch -> END
                 ^                         -[failed]-> add_feedback --+
                 +----------------------------------------------------+
                                           -[max iterations]-> escalate_to_human -> dispatch

The drafter keeps its conversation across iterations (`messages`), so a
revision does not repeat the tool calls of the first attempt.
"""

from __future__ import annotations

from typing import Annotated, Literal, NotRequired

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from email_assistant import prompts
from email_assistant.common import EmailInput, EmailOutput, EmailState, dispatch, email_message
from email_assistant.schemas import EmailResolution, QualityReview
from email_assistant.tools import ALL_TOOLS

MAX_ITERATIONS = 3


class ReflectionState(EmailState):
    messages: Annotated[list[AnyMessage], add_messages]
    review: NotRequired[QualityReview]
    iterations: NotRequired[int]


def build_graph(model: BaseChatModel):
    drafter = create_agent(
        model, tools=ALL_TOOLS, system_prompt=prompts.DRAFTER, response_format=EmailResolution, name="reply_drafter"
    )
    evaluator = model.with_structured_output(QualityReview)

    def draft(state: ReflectionState) -> dict:
        history = state.get("messages") or [email_message(state["email"])]
        result = drafter.invoke({"messages": history})
        return {
            "messages": result["messages"],
            "resolution": result["structured_response"],
            "iterations": state.get("iterations", 0) + 1,
        }

    def evaluate(state: ReflectionState) -> dict:
        prompt = f"{state['email'].as_prompt()}\n\nProposed resolution:\n{state['resolution'].model_dump_json()}"
        return {"review": evaluator.invoke([SystemMessage(prompts.REVIEWER), HumanMessage(prompt)])}

    def route(state: ReflectionState) -> Literal["dispatch", "add_feedback", "escalate_to_human"]:
        if state["review"].passed:
            return "dispatch"
        if state["iterations"] >= MAX_ITERATIONS:
            return "escalate_to_human"
        return "add_feedback"

    def add_feedback(state: ReflectionState) -> dict:
        review = state["review"]
        issues = "\n".join(f"- {i}" for i in review.issues)
        return {
            "messages": [
                HumanMessage(
                    f"Reviewer feedback (score {review.score:.2f}): {review.feedback}\n{issues}\n"
                    "Please revise the resolution accordingly."
                )
            ]
        }

    def escalate_to_human(state: ReflectionState) -> dict:
        resolution = state["resolution"].model_copy(
            update={
                "action": "escalate",
                "escalation_reason": f"Draft failed review {MAX_ITERATIONS}x: " + "; ".join(state["review"].issues),
            }
        )
        return {"resolution": resolution}

    builder = StateGraph(ReflectionState, input_schema=EmailInput, output_schema=EmailOutput)
    builder.add_node("draft", draft)
    builder.add_node("evaluate", evaluate)
    builder.add_node("add_feedback", add_feedback)
    builder.add_node("escalate_to_human", escalate_to_human)
    builder.add_node("dispatch", dispatch)
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "evaluate")
    builder.add_conditional_edges("evaluate", route)
    builder.add_edge("add_feedback", "draft")
    builder.add_edge("escalate_to_human", "dispatch")
    builder.add_edge("dispatch", END)
    return builder.compile(name="evaluator_optimizer")
