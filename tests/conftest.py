from __future__ import annotations

import re
from collections.abc import Callable

import pytest
from langchain_core.messages import AIMessage

from agentpatterns.testing import ModelTurn, ScriptedChatModel
from email_assistant.data import reset_backend
from email_assistant.mock_llm import create_mock_llm


@pytest.fixture(autouse=True)
def _clean_backend():
    reset_backend()
    yield
    reset_backend()


@pytest.fixture(scope="session")
def email_model() -> ScriptedChatModel:
    return create_mock_llm()


@pytest.fixture(scope="session")
def native_email_model() -> ScriptedChatModel:
    """Mock declaring native structured output support (the path current Claude models take)."""
    return create_mock_llm(native_structured_output=True)


def role_model(roles: dict[str, Callable[[ModelTurn], AIMessage | str]], **model_kwargs) -> ScriptedChatModel:
    """Scripted model that dispatches on a `role=<name>` marker in the system prompt."""

    def policy(turn: ModelTurn):
        match = re.search(r"role=(\w+)", turn.system)
        if not match or match.group(1) not in roles:
            raise AssertionError(f"unexpected system prompt: {turn.system!r}")
        return roles[match.group(1)](turn)

    return ScriptedChatModel(policy=policy, **model_kwargs)
