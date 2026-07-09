from click.testing import CliRunner

from promptforge.cli import main


def test_render_command(tmp_path):
    template_file = tmp_path / "greet.txt"
    template_file.write_text("Hello {{name}}!", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(main, ["render", str(template_file), "--var", "name=World"])
    assert result.exit_code == 0
    assert result.output.strip() == "Hello World!"


def test_render_command_missing_variable_errors(tmp_path):
    template_file = tmp_path / "greet.txt"
    template_file.write_text("Hello {{name}}!", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(main, ["render", str(template_file)])
    assert result.exit_code != 0


def test_tokens_command():
    runner = CliRunner()
    result = runner.invoke(main, ["tokens", "hello world"])
    assert result.exit_code == 0
    assert int(result.output.strip()) > 0


def test_tokens_command_requires_input():
    runner = CliRunner()
    result = runner.invoke(main, ["tokens"])
    assert result.exit_code != 0


def test_chat_command_echo_mode():
    runner = CliRunner()
    result = runner.invoke(main, ["chat"], input="hello\nexit\n")
    assert result.exit_code == 0
    assert "(echo) hello" in result.output
