import logging
import subprocess
from pathlib import Path

import pytest

from repolore.sources.plans import PlanFile, find_plan_checkouts, github_repo, load_plans, read_plans, save_plans


def checkout(root: Path, name: str, remote: str | None = None, plans: bool = True) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(directory)], check=True)
    if remote:
        subprocess.run(["git", "-C", str(directory), "remote", "add", "origin", remote], check=True)
    if plans:
        (directory / ".plans" / "feature").mkdir(parents=True)
        (directory / ".plans" / "feature" / "plan.md").write_text("# Plan\n\nShip it. ✓", encoding="utf-8")
    return directory


def test_checkouts_with_plans_are_keyed_by_their_github_repo(tmp_path: Path) -> None:
    repo = checkout(tmp_path, "repo", "https://github.com/octo/repo.git")
    checkout(tmp_path, "no-plans", "https://github.com/octo/no-plans.git", plans=False)
    (tmp_path / "loose-file.md").write_text("not a checkout", encoding="utf-8")

    assert find_plan_checkouts([tmp_path]) == {"octo/repo": repo}


def test_no_roots_means_no_checkouts() -> None:
    assert find_plan_checkouts([]) == {}


def test_missing_root_is_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert find_plan_checkouts([tmp_path / "missing"]) == {}

    assert "doesn't exist" in caplog.text


@pytest.mark.parametrize(
    "remote",
    [
        "https://github.com/octo/my.repo",
        "https://github.com/octo/my.repo.git",
        "https://github.com/octo/my.repo/",
        "git@github.com:octo/my.repo.git",
        "ssh://git@github.com/octo/my.repo.git",
    ],
)
def test_github_remote_forms_are_recognized(tmp_path: Path, remote: str) -> None:
    assert github_repo(checkout(tmp_path, "repo", remote)) == "octo/my.repo"


@pytest.mark.parametrize("remote", ["https://github.com/../..", "git@github.com:octo/..", "https://github.com/./x"])
def test_remote_that_would_leave_the_cache_dir_is_skipped(tmp_path: Path, remote: str) -> None:
    assert github_repo(checkout(tmp_path, "repo", remote)) is None


def test_root_under_the_home_directory_is_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    repo = checkout(tmp_path / "projects", "repo", "https://github.com/octo/repo.git")

    assert find_plan_checkouts([Path("~/projects")]) == {"octo/repo": repo}


def test_checkout_without_origin_is_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert github_repo(checkout(tmp_path, "repo")) is None

    assert "has no origin remote" in caplog.text


def test_checkout_hosted_elsewhere_is_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert github_repo(checkout(tmp_path, "repo", "https://gitlab.com/octo/repo.git")) is None

    assert "isn't a GitHub checkout" in caplog.text


def test_plans_are_read_with_paths_and_file_urls(tmp_path: Path) -> None:
    repo = checkout(tmp_path, "repo")
    (repo / ".plans" / "feature" / "resources").mkdir()
    (repo / ".plans" / "feature" / "resources" / "probe.md").write_text("Probe.", encoding="utf-8")
    (repo / ".plans" / "feature" / "resources" / "probe.json").write_text("{}", encoding="utf-8")

    plans = read_plans(repo)

    assert [plan.path for plan in plans] == [".plans/feature/plan.md", ".plans/feature/resources/probe.md"]
    assert plans[0].url == (repo / ".plans" / "feature" / "plan.md").resolve().as_uri()
    assert plans[0].text == "# Plan\n\nShip it. ✓"


def test_plans_round_trip_through_the_cache(tmp_path: Path) -> None:
    plans = [PlanFile(path=".plans/a/plan.md", url="file:///a/plan.md", text="Grüße")]

    save_plans(tmp_path / "octo" / "repo", plans)

    assert load_plans(tmp_path / "octo" / "repo") == plans
    assert load_plans(tmp_path / "octo" / "other") == []
