from promptforge.memory import ConversationMemory


def test_add_and_retrieve_messages():
    mem = ConversationMemory(token_budget=1000)
    mem.add_system("You are helpful.")
    mem.add_user("Hi")
    mem.add_assistant("Hello!")
    roles = [m.role for m in mem.messages]
    assert roles == ["system", "user", "assistant"]


def test_system_message_preserved_under_budget_pressure():
    mem = ConversationMemory(token_budget=5)
    mem.add_system("system prompt " * 20)  # big system message
    mem.add_user("hi")
    assert mem.messages[0].role == "system"


def test_oldest_non_system_dropped_first():
    mem = ConversationMemory(token_budget=8)
    mem.add_system("sys")
    mem.add_user("first message here")
    mem.add_user("second message here")
    mem.add_user("third message here")
    roles_and_content = [(m.role, m.content) for m in mem.messages]
    # oldest user message should have been evicted first
    assert not any(c == "first message here" for _, c in roles_and_content)


def test_clear_keep_system():
    mem = ConversationMemory(token_budget=1000)
    mem.add_system("sys")
    mem.add_user("hi")
    mem.clear(keep_system=True)
    assert [m.role for m in mem.messages] == ["system"]


def test_clear_all():
    mem = ConversationMemory(token_budget=1000)
    mem.add_system("sys")
    mem.add_user("hi")
    mem.clear(keep_system=False)
    assert mem.messages == []


def test_as_dicts_format():
    mem = ConversationMemory(token_budget=1000)
    mem.add_user("hi")
    assert mem.as_dicts() == [{"role": "user", "content": "hi"}]
