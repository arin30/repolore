import base64
import json
import logging
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from repolore.sources import github
from repolore.sources.github import GitHubClient, GitHubError, RawRepo, fetch_repo, resolve_token

Json = dict[str, Any]
INFO = {"nameWithOwner": "octo/repo", "url": "https://github.com/octo/repo", "isPrivate": False}


def page(nodes: list[Json], after: str | None = None) -> Json:
    return {"pageInfo": {"hasNextPage": after is not None, "endCursor": after}, "nodes": nodes}


def pull(number: int, files: int = 0) -> Json:
    connection = {"totalCount": 0, "nodes": []}
    return {
        "number": number,
        "files": {"totalCount": files, "nodes": []},
        "commits": connection,
        "comments": connection,
        "reviews": connection,
        "reviewThreads": connection,
    }


def tree_entry(path: str, sha: str, mode: str = "100644", kind: str = "blob") -> Json:
    return {"path": path, "sha": sha, "mode": mode, "type": kind}


class FakeGitHub:
    def __init__(self, branch: str | None = "main", pulls: list[Json] | None = None, truncated: bool = False) -> None:
        self.info = {**INFO, "defaultBranchRef": {"name": branch} if branch else None}
        self.pulls = pulls or [pull(1)]
        self.truncated = truncated
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/graphql":
            body = json.loads(request.content)
            return httpx.Response(200, json={"data": self._graphql(body["operationName"], body["variables"])})
        if "/git/trees/" in request.url.path:
            return httpx.Response(200, json=self._tree())
        sha = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, json={"content": base64.b64encode(f"# {sha} ✓".encode()).decode()})

    def _graphql(self, operation: str, variables: Json) -> Json:
        if operation == "RepoInfo":
            return {"repository": self.info}
        connections = {
            "RepoPullRequests": {"pullRequests": self._pull_page(variables["after"])},
            "RepoIssues": {"issues": page([{"number": 7}])},
            "DefaultBranchHistory": {"defaultBranchRef": {"target": {"history": page([{"oid": "abc"}])}}},
        }
        return {"repository": connections[operation]}

    def _pull_page(self, after: str | None) -> Json:
        if after is None and len(self.pulls) > 1:
            return page(self.pulls[:1], after="cursor-1")
        return page(self.pulls[1:] if after else self.pulls)

    def _tree(self) -> Json:
        entries = [
            tree_entry("README.md", "readme"),
            tree_entry("CLAUDE.md", "link", mode="120000"),
            tree_entry("docs", "dir", mode="040000", kind="tree"),
            tree_entry("docs/Design.MD", "design"),
            tree_entry("node_modules/pkg/README.md", "dependency"),
            tree_entry("src/main.go", "code"),
        ]
        return {"truncated": self.truncated, "tree": entries}


def client_for(handler: Callable[[httpx.Request], httpx.Response]) -> GitHubClient:
    return GitHubClient(httpx.Client(base_url=github.API_URL, transport=httpx.MockTransport(handler)))


def test_fetch_repo_collects_every_source() -> None:
    fake = FakeGitHub()

    raw = fetch_repo(client_for(fake.handle), "octo/repo")

    assert raw.info["nameWithOwner"] == "octo/repo"
    assert [p["number"] for p in raw.pulls] == [1]
    assert raw.issues == [{"number": 7}]
    assert raw.commits == [{"oid": "abc"}]
    assert raw.docs == [{"path": "README.md", "text": "# readme ✓"}, {"path": "docs/Design.MD", "text": "# design ✓"}]


def test_pull_requests_are_paginated() -> None:
    fake = FakeGitHub(pulls=[pull(1), pull(2)])

    raw = fetch_repo(client_for(fake.handle), "octo/repo")

    assert [p["number"] for p in raw.pulls] == [1, 2]
    afters = [json.loads(r.content)["variables"].get("after") for r in fake.requests if r.url.path == "/graphql"]
    assert "cursor-1" in afters


def test_repo_without_a_default_branch_has_no_commits_or_docs() -> None:
    fake = FakeGitHub(branch=None)

    raw = fetch_repo(client_for(fake.handle), "octo/repo")

    assert raw.commits == []
    assert raw.docs == []
    assert all(request.url.path == "/graphql" for request in fake.requests)


def test_truncated_inner_connection_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeGitHub(pulls=[pull(4, files=120)])

    with caplog.at_level(logging.WARNING):
        fetch_repo(client_for(fake.handle), "octo/repo")

    assert "octo/repo#4: only 0 of 120 files fetched" in caplog.text


def test_truncated_tree_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeGitHub(truncated=True)

    with caplog.at_level(logging.WARNING):
        fetch_repo(client_for(fake.handle), "octo/repo")

    assert "file tree is truncated" in caplog.text


def test_missing_repository_is_an_error() -> None:
    client = client_for(lambda _: httpx.Response(200, json={"data": {"repository": None}}))

    with pytest.raises(GitHubError, match="octo/repo: not found"):
        fetch_repo(client, "octo/repo")


def test_graphql_errors_are_raised() -> None:
    errors = [{"message": "Bad credentials"}, {"message": "Try again"}]
    client = client_for(lambda _: httpx.Response(200, json={"errors": errors}))

    with pytest.raises(GitHubError, match="RepoInfo: Bad credentials; Try again"):
        client.query("RepoInfo", owner="octo", name="repo")


def test_http_errors_are_raised() -> None:
    client = client_for(lambda _: httpx.Response(401, json={"message": "Bad credentials"}))

    with pytest.raises(GitHubError, match="HTTP 401"):
        client.get("/repos/octo/repo/git/trees/main")


def test_connect_authenticates_every_request() -> None:
    client = GitHubClient.connect("secret")

    assert client._http.headers["Authorization"] == "Bearer secret"
    client.close()


def test_raw_repo_round_trips_through_the_cache(tmp_path: Path) -> None:
    raw = RawRepo(info=INFO, pulls=[{"title": "Grüße"}], issues=[], commits=[], docs=[])

    raw.save(tmp_path / "octo" / "repo")

    assert RawRepo.load(tmp_path / "octo" / "repo") == raw
    assert "Grüße" in (tmp_path / "octo" / "repo" / "pulls.json").read_text(encoding="utf-8")


def test_token_comes_from_the_environment_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(github.TOKEN_ENV, "from-env")
    monkeypatch.setattr(subprocess, "run", pytest.fail)

    assert resolve_token() == "from-env"


def test_token_falls_back_to_gh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(github.TOKEN_ENV, raising=False)
    monkeypatch.setattr(subprocess, "run", lambda *_, **__: subprocess.CompletedProcess(["gh"], 0, stdout="from-gh\n"))

    assert resolve_token() == "from-gh"


@pytest.mark.parametrize("failure", [FileNotFoundError("gh"), subprocess.CalledProcessError(1, ["gh"])])
def test_missing_token_is_an_error(monkeypatch: pytest.MonkeyPatch, failure: Exception) -> None:
    def fail(*_: Any, **__: Any) -> None:
        raise failure

    monkeypatch.delenv(github.TOKEN_ENV, raising=False)
    monkeypatch.setattr(subprocess, "run", fail)

    with pytest.raises(GitHubError, match="No GitHub token"):
        resolve_token()
