import pytest

from promptforge.agent import Agent, AgentStep, ToolRegistry


def test_agent_returns_final_answer_directly():
    def chat_fn(messages, tool_specs):
        return AgentStep(kind="final", content="42")

    agent = Agent(chat_fn=chat_fn)
    assert agent.run("what is the answer?") == "42"


def test_agent_calls_tool_then_finishes():
    tools = ToolRegistry()

    @tools.register("add", "Add two numbers", {"a": "int", "b": "int"})
    def add(a: int, b: int) -> int:
        return a + b

    calls = {"n": 0}

    def chat_fn(messages, tool_specs):
        calls["n"] += 1
        if calls["n"] == 1:
            return AgentStep(kind="tool_call", tool_name="add", tool_args={"a": 2, "b": 3})
        return AgentStep(kind="final", content="The answer is 5")

    agent = Agent(chat_fn=chat_fn, tools=tools)
    result = agent.run("add 2 and 3")
    assert result == "The answer is 5"
    assert calls["n"] == 2

    # tool result should have been recorded in memory
    tool_messages = [m for m in agent.memory.messages if m.role == "tool"]
    assert any(m.content == "5" for m in tool_messages)


def test_agent_raises_after_max_steps():
    def chat_fn(messages, tool_specs):
        return AgentStep(kind="tool_call", tool_name="noop", tool_args={})

    tools = ToolRegistry()

    @tools.register("noop", "does nothing")
    def noop():
        return "done"

    agent = Agent(chat_fn=chat_fn, tools=tools, max_steps=2)
    with pytest.raises(RuntimeError):
        agent.run("loop forever")


def test_unknown_tool_raises_keyerror():
    tools = ToolRegistry()
    with pytest.raises(KeyError):
        tools.call("does_not_exist")
