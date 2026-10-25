import tomllib
from fnmatch import fnmatchcase
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    include: list[str] = []
    exclude: list[str] = []
    local_roots: list[Path] = []

    def selects(self, repo: str) -> bool:
        """Whether a discovered repo is ingested: matches an include glob (or none are set) and no exclude glob."""
        included = not self.include or _matches_any(repo, self.include)
        return included and not _matches_any(repo, self.exclude)


def _matches_any(repo: str, patterns: list[str]) -> bool:
    return any(fnmatchcase(repo.lower(), pattern.lower()) for pattern in patterns)


def load_config(path: Path) -> Config:
    """The config file, or the defaults (every discovered repo, no local plans) when there is none."""
    if not path.exists():
        return Config()
    return Config.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
