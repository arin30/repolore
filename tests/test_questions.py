from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from repolore.evals.questions import Question, QuestionKind, QuestionSet, load_questions

EXAMPLE_FILE = Path(__file__).parent.parent / "evals" / "questions.example.yaml"
PR_URL = "https://github.com/octo-org/example-repo/pull/12"


def question(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "q-1",
        "question": "Why?",
        "kind": "rationale",
        "expected": [{"url": PR_URL}],
        "answer": "Because.",
    } | overrides


def test_example_file_loads() -> None:
    question_set = load_questions(EXAMPLE_FILE)

    assert {q.kind for q in question_set.questions} >= {QuestionKind.FACTUAL, QuestionKind.UNANSWERABLE}
    assert any(q.holdout for q in question_set.questions)


def test_question_defaults_to_tuning_set_without_section() -> None:
    parsed = Question.model_validate(question())

    assert parsed.holdout is False
    assert parsed.expected[0].section is None


def test_empty_question_list_loads() -> None:
    assert QuestionSet.model_validate({"questions": []}).questions == []


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "questions.yaml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_questions(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_questions(tmp_path / "missing.yaml")


def test_answerable_question_needs_a_source() -> None:
    with pytest.raises(ValidationError, match="needs at least one expected source"):
        Question.model_validate(question(expected=[]))


def test_unanswerable_question_cannot_have_sources() -> None:
    with pytest.raises(ValidationError, match="cannot have expected sources"):
        Question.model_validate(question(kind="unanswerable"))


def test_unanswerable_question_without_sources_is_valid() -> None:
    parsed = Question.model_validate(question(kind="unanswerable", expected=[]))

    assert parsed.kind is QuestionKind.UNANSWERABLE


@pytest.mark.parametrize(
    "url",
    [
        "https://gitlab.com/octo-org/example-repo/pull/12",
        "http://github.com/octo-org/example-repo/pull/12",
        "https://github.com/octo-org",
        "github.com/octo-org/example-repo",
    ],
)
def test_non_github_urls_are_rejected(url: str) -> None:
    with pytest.raises(ValidationError):
        Question.model_validate(question(expected=[{"url": url}]))


@pytest.mark.parametrize("question_id", ["Q-1", "q_1", "q-", "-q", ""])
def test_ids_must_be_kebab_case(question_id: str) -> None:
    with pytest.raises(ValidationError):
        Question.model_validate(question(id=question_id))


@pytest.mark.parametrize("field", ["question", "answer"])
def test_blank_text_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        Question.model_validate(question(**{field: "   "}))


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Question.model_validate(question(expected_url=PR_URL))


def test_duplicate_ids_are_reported() -> None:
    questions = [question(id="a"), question(id="b"), question(id="a")]

    with pytest.raises(ValidationError, match="duplicate question ids: a"):
        QuestionSet.model_validate({"questions": questions})
