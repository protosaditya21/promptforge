# promptforge

A lightweight, **provider-agnostic** toolkit for prompt engineering and
LLM agents. Works as a Python library and as a CLI. No hard dependency
on any specific LLM SDK (Anthropic, OpenAI, local models — bring your
own `chat_fn`).

## Features

- **`PromptTemplate`** — `{{variable}}` prompt templates with validation
  (catches missing/typo'd variables before they hit the API).
- **`count_tokens` / `fits_budget`** — token counting, using `tiktoken`
  if installed, with a sane heuristic fallback otherwise.
- **`with_retry`** — exponential backoff decorator for flaky API calls.
- **`ConversationMemory`** — rolling chat history that stays under a
  token budget, always preserving the system prompt.
- **`Agent` / `ToolRegistry`** — a minimal tool-calling loop you can
  drive with any model backend.
- **CLI** — `promptforge render`, `promptforge tokens`, `promptforge chat`.

## Install

```bash
pip install -e .
# optional accurate tokenizer:
pip install -e ".[tiktoken]"
```

## Library usage

### Prompt templates

```python
from promptforge import PromptTemplate

t = PromptTemplate(
    template="Summarize the following {{doc_type}} in {{n}} sentences:\n\n{{text}}",
    name="summarize",
)
prompt = t.render(doc_type="article", n="3", text="...")
```

### Token budgeting

```python
from promptforge import count_tokens, fits_budget

count_tokens("hello world")        # -> int
fits_budget(prompt, budget=8000)   # -> bool
```

### Retrying flaky calls

```python
from promptforge import with_retry

@with_retry(max_attempts=5, exceptions=(TimeoutError, ConnectionError))
def call_llm(prompt: str) -> str:
    ...
```

### Conversation memory

```python
from promptforge import ConversationMemory

mem = ConversationMemory(token_budget=4000)
mem.add_system("You are a helpful assistant.")
mem.add_user("Hi!")
mem.add_assistant("Hello, how can I help?")
mem.as_dicts()  # -> list[{"role": ..., "content": ...}], ready for any chat API
```

### Agent loop with tools

`Agent` is deliberately decoupled from any SDK: you write a `chat_fn`
that maps `(messages, tool_specs) -> AgentStep`. Here's a sketch wiring
it to the Anthropic API:

```python
import anthropic
from promptforge import Agent, AgentStep, ToolRegistry

client = anthropic.Anthropic()
tools = ToolRegistry()

@tools.register("get_weather", "Get current weather for a city",
                 {"city": {"type": "string"}})
def get_weather(city: str) -> str:
    return f"It's sunny in {city}."

def chat_fn(messages, tool_specs):
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=messages,
        tools=[{"name": t["name"], "description": t["description"],
                "input_schema": {"type": "object", "properties": t["parameters"]}}
               for t in tool_specs],
    )
    block = response.content[0]
    if block.type == "tool_use":
        return AgentStep(kind="tool_call", tool_name=block.name, tool_args=block.input)
    return AgentStep(kind="final", content=block.text)

agent = Agent(chat_fn=chat_fn, tools=tools)
print(agent.run("What's the weather in Paris?"))
```

## CLI usage

```bash
# Render a template file
promptforge render prompt.txt --var name=Ada --var topic=orbital-mechanics

# Count tokens
promptforge tokens "how many tokens is this?"
promptforge tokens --file prompt.txt

# Interactive chat REPL (offline echo mode by default; wire in your own
# chat_fn for a real provider, see Agent docs above)
promptforge chat --system "You are a pirate." --budget 4000
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
