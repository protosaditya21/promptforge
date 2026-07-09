"""Command line interface for promptforge."""
from __future__ import annotations

import os
from typing import Optional

import click

from .agent import Agent, AgentStep
from .memory import ConversationMemory
from .templates import ExtraVariableError, MissingVariableError, PromptTemplate
from .tokens import count_tokens


@click.group()
@click.version_option(package_name="promptforge")
def main() -> None:
    """promptforge: a toolkit for prompt engineering and lightweight LLM agents."""


@main.command()
@click.argument("template_path", type=click.Path(exists=True))
@click.option("--var", "variables", multiple=True, help="key=value pair, repeatable")
@click.option("--strict/--no-strict", default=True, help="Error on unused variables")
def render(template_path: str, variables: tuple, strict: bool) -> None:
    """Render a prompt template file with the given variables."""
    kwargs = {}
    for item in variables:
        if "=" not in item:
            raise click.BadParameter(f"Expected key=value, got '{item}'")
        key, _, value = item.partition("=")
        kwargs[key] = value

    template = PromptTemplate.from_file(template_path, name=os.path.basename(template_path))
    try:
        result = template.render(strict=strict, **kwargs)
    except (MissingVariableError, ExtraVariableError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(result)


@main.command()
@click.argument("text", required=False)
@click.option("--file", "file_path", type=click.Path(exists=True), help="Read text from a file")
@click.option("--encoding", default="cl100k_base", help="Tokenizer encoding name")
def tokens(text: Optional[str], file_path: Optional[str], encoding: str) -> None:
    """Count tokens in TEXT, or in the file given by --file."""
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    if not text:
        raise click.UsageError("Provide TEXT or --file")
    click.echo(str(count_tokens(text, encoding)))


def _echo_chat_fn(messages: list, tool_specs: list) -> AgentStep:
    """Default offline chat function: echoes the last user message.

    Lets the CLI run with zero configuration. Wire up a real provider by
    writing your own chat_fn -- see the README for an example.
    """
    last_user = next((m for m in reversed(messages) if m["role"] == "user"), None)
    content = f"(echo) {last_user['content']}" if last_user else "(echo) ..."
    return AgentStep(kind="final", content=content)


@main.command()
@click.option("--system", default="You are a helpful assistant.", help="System prompt")
@click.option("--budget", default=4000, help="Token budget for conversation memory")
def chat(system: str, budget: int) -> None:
    """Start an interactive REPL chat session.

    Uses an offline echo responder by default so the CLI works without
    any API key configured. See the README for wiring up a real
    provider's chat_fn.
    """
    memory = ConversationMemory(token_budget=budget)
    memory.add_system(system)
    agent = Agent(chat_fn=_echo_chat_fn, memory=memory)

    click.echo("promptforge chat (offline echo mode). Type 'exit' to quit.")
    while True:
        try:
            user_input = click.prompt(">")
        except (EOFError, KeyboardInterrupt):
            click.echo("\nbye")
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            click.echo("bye")
            break
        reply = agent.run(user_input)
        click.echo(reply)


if __name__ == "__main__":
    main()
