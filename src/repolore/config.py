import tomllib
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

RepoName = Annotated[str, StringConstraints(pattern=r"^[\w.-]+/[\w.-]+$")]


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repos: list[RepoName] = Field(min_length=1)


def load_config(path: Path) -> Config:
    return Config.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
