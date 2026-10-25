from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from repolore.config import Config
from repolore.sources.github import GitHubClient, involved_repositories, viewer_repositories
from repolore.sources.plans import find_plan_checkouts


class Scope(StrEnum):
    OWNED = "owned"
    MEMBER = "member"
    CONTRIBUTED = "contributed"
    LOCAL = "local"


@dataclass(frozen=True)
class RepoSource:
    """A repo to ingest: its whole history (owned, member), only the viewer's part (contributed) or only local plans."""

    name: str
    scope: Scope
    checkout: Path | None = None


@dataclass(frozen=True)
class Discovery:
    login: str
    included: list[RepoSource]
    excluded: list[RepoSource]


def discover(client: GitHubClient, config: Config) -> Discovery:
    login, viewer_repos = viewer_repositories(client)
    found: dict[str, RepoSource] = {}
    for repo in viewer_repos:
        if not repo.is_fork:
            scope = Scope.OWNED if repo.owner.lower() == login.lower() else Scope.MEMBER
            found[repo.name.lower()] = RepoSource(repo.name, scope)
    for name in involved_repositories(client, login):
        found.setdefault(name.lower(), RepoSource(name, Scope.CONTRIBUTED))
    for name, checkout in find_plan_checkouts(config.local_roots).items():
        known = found.get(name.lower())
        found[name.lower()] = replace(known, checkout=checkout) if known else RepoSource(name, Scope.LOCAL, checkout)
    sources = sorted(found.values(), key=lambda source: source.name.lower())
    return Discovery(
        login=login,
        included=[source for source in sources if config.selects(source.name)],
        excluded=[source for source in sources if not config.selects(source.name)],
    )
