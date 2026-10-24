import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from repolore import __version__
from repolore.cli import app
from repolore.paths import resolve_paths
from repolore.sources.github import GitHubClient, GitHubError

runner = CliRunner()
EXAMPLE_FILE = Path(__file__).parent.parent / "evals" / "questions.example.yaml"
FIXTURES = Path(__file__).parent / "fixtures"


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


def ingest_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: str | None) -> Path:
    config_file = tmp_path / "repolore.toml"
    if config is not None:
        config_file.write_text(config, encoding="utf-8")
    monkeypatch.setattr(
        "repolore.cli.resolve_paths", lambda _: replace(resolve_paths(tmp_path), config_file=config_file)
    )
    return tmp_path


def test_offline_ingest_prints_counts_per_repo_and_kind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = ingest_setup(tmp_path, monkeypatch, 'repos = ["ryckakas/crashcause"]')
    shutil.copytree(FIXTURES / "crashcause", data_dir / "raw" / "ryckakas" / "crashcause")

    result = runner.invoke(app, ["ingest", "--offline"])

    assert result.exit_code == 0
    lines = result.output.splitlines()
    assert lines[0].split() == ["repo", "pr", "review", "issue", "commit", "doc"]
    assert lines[1].split() == ["ryckakas/crashcause", "2", "0", "2", "1", "2"]
    assert lines[2].split() == ["total", "2", "0", "2", "1", "2"]


def test_ingest_without_config_explains_how_to_create_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ingest_setup(tmp_path, monkeypatch, None)

    result = runner.invoke(app, ["ingest", "--offline"])

    assert result.exit_code == 1
    assert 'repos = ["owner/name"]' in result.output


@pytest.mark.parametrize("config", ["repos = [", "repos = []"])
def test_ingest_with_invalid_config_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: str) -> None:
    ingest_setup(tmp_path, monkeypatch, config)

    result = runner.invoke(app, ["ingest", "--offline"])

    assert result.exit_code == 1
    assert "is invalid" in result.output


def test_offline_ingest_without_cache_says_to_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ingest_setup(tmp_path, monkeypatch, 'repos = ["ryckakas/crashcause"]')

    result = runner.invoke(app, ["ingest", "--offline"])

    assert result.exit_code == 1
    assert "without --offline" in result.output


def test_ingest_without_token_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_token() -> str:
        raise GitHubError("No GitHub token")

    ingest_setup(tmp_path, monkeypatch, 'repos = ["ryckakas/crashcause"]')
    monkeypatch.setattr("repolore.cli.resolve_token", no_token)

    result = runner.invoke(app, ["ingest"])

    assert result.exit_code == 1
    assert "No GitHub token" in result.output


def test_online_ingest_closes_the_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[bool] = []
    ingest_setup(tmp_path, monkeypatch, 'repos = ["ryckakas/crashcause"]')
    monkeypatch.setattr("repolore.cli.resolve_token", lambda: "token")
    monkeypatch.setattr("repolore.cli.ingest_repos", lambda *_: [])
    monkeypatch.setattr(GitHubClient, "close", lambda _: closed.append(True))

    result = runner.invoke(app, ["ingest"])

    assert result.exit_code == 0
    assert closed == [True]
