from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from repolore import __version__
from repolore.cli import app
from repolore.paths import resolve_paths

runner = CliRunner()
EXAMPLE_FILE = Path(__file__).parent.parent / "evals" / "questions.example.yaml"


def test_version_prints_version_and_default_paths() -> None:
    result = runner.invoke(app, ["version"])

    defaults = resolve_paths()
    assert result.exit_code == 0
    assert f"repolore {__version__}" in result.output
    assert f"config: {defaults.config_file}" in result.output
    assert f"data:   {defaults.data_dir}" in result.output


def test_data_dir_option_overrides_data_path(tmp_path: Path) -> None:
    result = runner.invoke(app, ["--data-dir", str(tmp_path), "version"])

    assert result.exit_code == 0
    assert f"data:   {tmp_path.resolve()}" in result.output


def test_no_command_shows_help() -> None:
    result = runner.invoke(app, [])

    assert "version" in result.output


def test_questions_summarizes_a_valid_file() -> None:
    result = runner.invoke(app, ["questions", "--file", str(EXAMPLE_FILE)])

    assert result.exit_code == 0
    assert "total:     3 (1 held out)" in result.output
    assert "factual       1" in result.output
    assert "commit-only   0" in result.output


def test_questions_defaults_to_the_config_dir_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "evals" / "questions.yaml"
    monkeypatch.setattr("repolore.cli.resolve_paths", lambda _: replace(resolve_paths(), questions_file=missing))

    result = runner.invoke(app, ["questions"])

    assert result.exit_code == 1
    assert f"No question file at {missing}" in result.output


@pytest.mark.parametrize("content", ["questions: [", "questions:\n  - id: Not-Kebab\n"])
def test_questions_reports_an_invalid_file(tmp_path: Path, content: str) -> None:
    path = tmp_path / "questions.yaml"
    path.write_text(content, encoding="utf-8")

    result = runner.invoke(app, ["questions", "--file", str(path)])

    assert result.exit_code == 1
    assert f"{path} is invalid" in result.output
