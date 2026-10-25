import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from repolore.config import Config, load_config


def write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "repolore.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_settings_are_loaded(tmp_path: Path) -> None:
    config = load_config(
        write(tmp_path, 'include = ["octo/*"]\nexclude = ["octo/old-*"]\nlocal_roots = ["D:/projects"]')
    )

    assert config == Config(include=["octo/*"], exclude=["octo/old-*"], local_roots=[Path("D:/projects")])


@pytest.mark.parametrize("content", [None, ""])
def test_missing_or_empty_config_selects_everything(tmp_path: Path, content: str | None) -> None:
    path = write(tmp_path, content) if content is not None else tmp_path / "repolore.toml"

    config = load_config(path)

    assert config == Config()
    assert config.selects("anyone/anything")


@pytest.mark.parametrize(
    ("config", "selected"),
    [
        (Config(include=["octo/*"]), ["octo/repo", "Octo/Old-Demo"]),
        (Config(exclude=["*/old-*"]), ["octo/repo", "other/repo"]),
        (Config(include=["octo/*"], exclude=["octo/old-*"]), ["octo/repo"]),
        (Config(include=["octo/repo"]), ["octo/repo"]),
    ],
)
def test_include_and_exclude_globs_ignore_case(config: Config, selected: list[str]) -> None:
    repos = ["octo/repo", "Octo/Old-Demo", "other/repo"]

    assert [repo for repo in repos if config.selects(repo)] == selected


@pytest.mark.parametrize("content", ['repos = ["octo/repo"]', 'include = "octo/*"', "local_roots = [1]"])
def test_invalid_config_is_rejected(tmp_path: Path, content: str) -> None:
    with pytest.raises(ValidationError):
        load_config(write(tmp_path, content))


def test_broken_toml_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(tomllib.TOMLDecodeError):
        load_config(write(tmp_path, "include = ["))
