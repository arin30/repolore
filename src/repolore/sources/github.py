import base64
import json
import logging
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, fields
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any, Self

import httpx

API_URL = "https://api.github.com"
TOKEN_ENV = "REPOLORE_GITHUB_TOKEN"
QUERIES = files("repolore.sources").joinpath("github.graphql").read_text(encoding="utf-8")
INNER_CONNECTIONS = ("files", "commits", "comments", "reviews", "reviewThreads")
SKIPPED_DIRS = frozenset({"node_modules", "vendor"})
SYMLINK_MODE = "120000"

Json = dict[str, Any]
log = logging.getLogger(__name__)


class GitHubError(Exception):
    pass


def resolve_token() -> str:
    if token := os.environ.get(TOKEN_ENV):
        return token
    try:
        result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise GitHubError(f"No GitHub token: set {TOKEN_ENV}, or log in with `gh auth login`.") from error
    return result.stdout.strip()


class GitHubClient:
    def __init__(self, http: httpx.Client) -> None:
        self._http = http

    @classmethod
    def connect(cls, token: str) -> Self:
        headers = {"Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2022-11-28"}
        return cls(httpx.Client(base_url=API_URL, headers=headers, timeout=60))

    def close(self) -> None:
        self._http.close()

    def query(self, operation: str, **variables: Any) -> Json:
        body = {"query": QUERIES, "operationName": operation, "variables": variables}
        payload = self._send(self._http.post("/graphql", json=body))
        if errors := payload.get("errors"):
            raise GitHubError(f"{operation}: {'; '.join(error['message'] for error in errors)}")
        return payload["data"]

    def get(self, path: str, **params: Any) -> Json:
        return self._send(self._http.get(path, params=params))

    @staticmethod
    def _send(response: httpx.Response) -> Json:
        if response.is_error:
            raise GitHubError(f"{response.request.method} {response.request.url.path}: HTTP {response.status_code}")
        return response.json()


@dataclass(frozen=True)
class RawRepo:
    info: Json
    pulls: list[Json]
    issues: list[Json]
    commits: list[Json]
    docs: list[Json]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for field in fields(self):
            content = json.dumps(getattr(self, field.name), ensure_ascii=False, indent=1)
            (directory / f"{field.name}.json").write_text(content, encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> Self:
        def read(name: str) -> Any:
            return json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))

        return cls(**{field.name: read(field.name) for field in fields(cls)})


def fetch_repo(client: GitHubClient, repo: str) -> RawRepo:
    owner, name = repo.split("/")
    info = client.query("RepoInfo", owner=owner, name=name)["repository"]
    if info is None:
        raise GitHubError(f"{repo}: not found, or the token can't read it")
    branch = (info["defaultBranchRef"] or {}).get("name")
    pulls = _paginate(client, "RepoPullRequests", owner, name, lambda data: data["repository"]["pullRequests"])
    _warn_about_truncation(repo, pulls)
    return RawRepo(
        info=info,
        pulls=pulls,
        issues=_paginate(client, "RepoIssues", owner, name, lambda data: data["repository"]["issues"]),
        commits=_fetch_commits(client, owner, name) if branch else [],
        docs=_fetch_docs(client, repo, branch) if branch else [],
    )


def _paginate(
    client: GitHubClient, operation: str, owner: str, name: str, connection: Callable[[Json], Json]
) -> list[Json]:
    nodes: list[Json] = []
    after = None
    while True:
        page = connection(client.query(operation, owner=owner, name=name, after=after))
        nodes += page["nodes"]
        if not page["pageInfo"]["hasNextPage"]:
            return nodes
        after = page["pageInfo"]["endCursor"]


def _fetch_commits(client: GitHubClient, owner: str, name: str) -> list[Json]:
    return _paginate(
        client,
        "DefaultBranchHistory",
        owner,
        name,
        lambda data: data["repository"]["defaultBranchRef"]["target"]["history"],
    )


def _warn_about_truncation(repo: str, pulls: list[Json]) -> None:
    for pull in pulls:
        for key in INNER_CONNECTIONS:
            connection = pull[key]
            if connection["totalCount"] > len(connection["nodes"]):
                log.warning(
                    "%s#%s: only %d of %d %s fetched",
                    repo,
                    pull["number"],
                    len(connection["nodes"]),
                    connection["totalCount"],
                    key,
                )


def _fetch_docs(client: GitHubClient, repo: str, branch: str) -> list[Json]:
    tree = client.get(f"/repos/{repo}/git/trees/{branch}", recursive=1)
    if tree["truncated"]:
        log.warning("%s: the file tree is truncated, so some markdown files are missing", repo)
    return [
        {"path": entry["path"], "text": _read_blob(client, repo, entry["sha"])}
        for entry in tree["tree"]
        if _is_markdown_file(entry)
    ]


def _is_markdown_file(entry: Json) -> bool:
    path = PurePosixPath(entry["path"])
    return (
        entry["type"] == "blob"
        and entry["mode"] != SYMLINK_MODE
        and path.suffix.lower() == ".md"
        and not SKIPPED_DIRS.intersection(path.parts)
    )


def _read_blob(client: GitHubClient, repo: str, sha: str) -> str:
    blob = client.get(f"/repos/{repo}/git/blobs/{sha}")
    return base64.b64decode(blob["content"]).decode("utf-8", errors="replace")
