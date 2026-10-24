import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from repolore.config import load_config


def write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "repolore.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_repos_are_loaded(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, 'repos = ["octo/repo", "octo/other.repo-2"]'))

    assert config.repos == ["octo/repo", "octo/other.repo-2"]


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "repolore.toml")


@pytest.mark.parametrize(
    "content",
    ["", "repos = []", 'repos = ["octo"]', 'repos = ["octo/repo/extra"]', 'repos = ["octo/repo"]\nrepo = "x"'],
)
def test_invalid_config_is_rejected(tmp_path: Path, content: str) -> None:
    with pytest.raises(ValidationError):
        load_config(write(tmp_path, content))


def test_broken_toml_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(tomllib.TOMLDecodeError):
        load_config(write(tmp_path, "repos = ["))
