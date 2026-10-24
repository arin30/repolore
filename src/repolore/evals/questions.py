from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

GitHubUrl = Annotated[str, StringConstraints(pattern=r"^https://github\.com/[^/\s]+/[^/\s]+(/\S*)?$")]
QuestionId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")]
NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class QuestionKind(StrEnum):
    FACTUAL = "factual"
    RATIONALE = "rationale"
    CROSS_REPO = "cross-repo"
    COMMIT_ONLY = "commit-only"
    UNANSWERABLE = "unanswerable"


class ExpectedSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    url: GitHubUrl
    section: NonBlank | None = None


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: QuestionId
    question: NonBlank
    kind: QuestionKind
    expected: list[ExpectedSource] = Field(default_factory=list)
    answer: NonBlank
    holdout: bool = False

    @model_validator(mode="after")
    def _only_unanswerable_questions_lack_sources(self) -> Self:
        unanswerable = self.kind is QuestionKind.UNANSWERABLE
        if unanswerable and self.expected:
            raise ValueError("an unanswerable question cannot have expected sources")
        if not unanswerable and not self.expected:
            raise ValueError(f"a {self.kind} question needs at least one expected source")
        return self


class QuestionSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    questions: list[Question]

    @model_validator(mode="after")
    def _ids_are_unique(self) -> Self:
        counts = Counter(q.id for q in self.questions)
        duplicates = sorted(question_id for question_id, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate question ids: {', '.join(duplicates)}")
        return self


def load_questions(path: Path) -> QuestionSet:
    return QuestionSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
