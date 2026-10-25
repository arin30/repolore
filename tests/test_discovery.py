from pathlib import Path

import pytest

from repolore import discovery as discovery_module
from repolore.config import Config
from repolore.discovery import RepoSource, Scope, discover
from repolore.sources.github import GitHubClient, ViewerRepo

CLIENT = GitHubClient.connect("unused")


@pytest.fixture
def found(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    viewer_repos = [
        ViewerRepo("octocat/repo", "octocat", is_fork=False),
        ViewerRepo("octocat/fork", "octocat", is_fork=True),
        ViewerRepo("octo-org/service", "octo-org", is_fork=False),
    ]
    monkeypatch.setattr(discovery_module, "viewer_repositories", lambda _: ("OctoCat", viewer_repos))
    monkeypatch.setattr(discovery_module, "involved_repositories", lambda *_: ["octo-org/service", "other/project"])
    checkouts = {"OctoCat/Repo": tmp_path / "repo", "someone/clone": tmp_path / "clone"}
    monkeypatch.setattr(discovery_module, "find_plan_checkouts", lambda _: checkouts)
    return tmp_path


def test_repos_are_scoped_by_how_the_viewer_relates_to_them(found: Path) -> None:
    result = discover(CLIENT, Config())

    assert result.login == "OctoCat"
    assert result.included == [
        RepoSource("octo-org/service", Scope.MEMBER),
        RepoSource("octocat/repo", Scope.OWNED, found / "repo"),
        RepoSource("other/project", Scope.CONTRIBUTED),
        RepoSource("someone/clone", Scope.LOCAL, found / "clone"),
    ]
    assert result.excluded == []


def test_config_globs_split_included_from_excluded(found: Path) -> None:
    result = discover(CLIENT, Config(include=["octo*/*"], exclude=["octo-org/*"]))

    assert [source.name for source in result.included] == ["octocat/repo"]
    assert [source.name for source in result.excluded] == ["octo-org/service", "other/project", "someone/clone"]


def test_nothing_found_is_an_empty_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discovery_module, "viewer_repositories", lambda _: ("octocat", []))
    monkeypatch.setattr(discovery_module, "involved_repositories", lambda *_: [])

    result = discover(CLIENT, Config())

    assert (result.included, result.excluded) == ([], [])
