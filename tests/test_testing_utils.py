"""The scripted mock model behaves like a real tool-calling chat model."""

from __future__ import annotations

import pytest
from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel

from agentpatterns.testing import ScriptedChatModel, ScriptError, UsageTracker


@tool
def get_weather(city: str) -> dict:
    """Weather for a city."""
    return {"city": city, "temp": 21}


class Forecast(BaseModel):
    """Structured forecast."""

    city: str
    temp: int


def weather_policy(turn):
    if not turn.called("get_weather"):
        return turn.call_many(("get_weather", {"city": "Berlin"}), ("get_weather", {"city": "Rome"}))
    first = turn.json_results("get_weather")[0]
    return turn.structured(Forecast(**first))


def test_agent_with_parallel_tool_calls_and_structured_output():
    agent = create_agent(ScriptedChatModel(policy=weather_policy), tools=[get_weather], response_format=Forecast)
    tracker = UsageTracker()
    result = agent.invoke({"messages": [("user", "weather?")]}, config={"callbacks": [tracker]})
    assert result["structured_response"] == Forecast(city="Berlin", temp=21)
    summary = tracker.summary()
    assert summary["model_calls"] == 2
    assert summary["tools"] == {"get_weather": 2}
    assert summary["input_tokens"] > 0


def test_with_structured_output_uses_the_single_bound_tool():
    model = ScriptedChatModel(policy=lambda t: t.structured({"city": "Oslo", "temp": 3}))
    assert model.with_structured_output(Forecast).invoke("hi") == Forecast(city="Oslo", temp=3)


def test_calling_an_unbound_tool_is_an_error():
    model = ScriptedChatModel(policy=lambda t: t.call("does_not_exist"))
    with pytest.raises(ScriptError, match="only"):
        create_agent(model, tools=[get_weather]).invoke({"messages": [("user", "x")]})


def test_forced_tool_choice_rejects_plain_text():
    model = ScriptedChatModel(policy=lambda t: t.say("plain text"))
    with pytest.raises(ScriptError, match="forces a tool call"):
        model.with_structured_output(Forecast).invoke("hi")


def test_native_structured_output_returns_json_text():
    seen = []

    def policy(turn):
        seen.append((turn.response_format["json_schema"]["name"], turn.tool_names))
        return turn.structured({"city": "Oslo", "temp": 3})

    model = ScriptedChatModel(policy=policy)
    assert model.with_structured_output(Forecast, method="json_schema").invoke("hi") == Forecast(city="Oslo", temp=3)
    assert seen == [("Forecast", [])]  # schema sent as response format, no tool bound


def test_profile_with_native_support_makes_create_agent_use_provider_strategy():
    seen = []

    def policy(turn):
        seen.append(turn.response_format is not None)
        return weather_policy(turn)

    model = ScriptedChatModel(policy=policy, profile={"structured_output": True})
    agent = create_agent(model, tools=[get_weather], response_format=Forecast)
    assert agent.invoke({"messages": [("user", "weather?")]})["structured_response"] == Forecast(city="Berlin", temp=21)
    assert seen == [True, True]
