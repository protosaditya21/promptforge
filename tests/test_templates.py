import pytest

from promptforge.templates import (
    ExtraVariableError,
    MissingVariableError,
    PromptTemplate,
)


def test_render_basic():
    t = PromptTemplate(template="Hello {{name}}, welcome to {{place}}.")
    assert t.render(name="Ada", place="Cloud") == "Hello Ada, welcome to Cloud."


def test_variables_detected():
    t = PromptTemplate(template="{{a}} and {{b}} and {{a}}")
    assert t.variables() == {"a", "b"}


def test_missing_variable_raises():
    t = PromptTemplate(template="Hello {{name}}", name="greet")
    with pytest.raises(MissingVariableError):
        t.render()


def test_extra_variable_strict_raises():
    t = PromptTemplate(template="Hello {{name}}", name="greet")
    with pytest.raises(ExtraVariableError):
        t.render(name="Ada", extra="oops")


def test_extra_variable_non_strict_ok():
    t = PromptTemplate(template="Hello {{name}}")
    assert t.render(strict=False, name="Ada", extra="oops") == "Hello Ada"


def test_from_file(tmp_path):
    p = tmp_path / "prompt.txt"
    p.write_text("Summarize: {{text}}", encoding="utf-8")
    t = PromptTemplate.from_file(str(p), name="summary")
    assert t.render(text="hello world") == "Summarize: hello world"
