import json
from pathlib import Path

from repolore.evals.harness import evaluate
from repolore.evals.questions import Question
from repolore.evals.runs import previous_run, run_record, save_run
from repolore.store import Hit

URL = "https://github.com/octo/repo/pull/1"


def results():
    q = Question.model_validate(
        {"id": "q", "question": "q", "kind": "factual", "expected": [{"url": URL}], "answer": "a"}
    )
    return evaluate([q], lambda _: [Hit("c", URL, "octo/repo", "pr", "t", None, None, None, "h", "t", 0.75)])


def record(timestamp: str, questions: str = "abc:tuning") -> dict:
    return {**run_record({"questions": questions}, results()), "timestamp": timestamp}


def test_run_record_keeps_config_metrics_and_per_question_ranks() -> None:
    run = run_record({"index": "default", "questions": "abc:tuning"}, results())

    assert run["config"] == {"index": "default", "questions": "abc:tuning"}
    assert run["metrics"]["recall@5"] == 1.0
    assert run["questions"] == [{"id": "q", "ranks": [1], "top_score": 0.75}]
    assert run["search_ms"] >= 0
    assert run["timestamp"].endswith("+00:00")


def test_saved_runs_are_named_by_time(tmp_path: Path) -> None:
    path = save_run(tmp_path / "runs", record("2026-10-24T18:29:34+00:00"))

    assert path.name == "2026-10-24T182934Z.json"
    assert json.loads(path.read_text(encoding="utf-8"))["questions"][0]["id"] == "q"


def test_previous_run_is_the_latest_on_the_same_questions(tmp_path: Path) -> None:
    save_run(tmp_path, record("2026-10-24T10:00:00+00:00"))
    save_run(tmp_path, record("2026-10-24T11:00:00+00:00"))
    save_run(tmp_path, record("2026-10-24T12:00:00+00:00", questions="abc:holdout"))

    previous = previous_run(tmp_path, "abc:tuning")

    assert previous is not None
    assert previous["timestamp"] == "2026-10-24T11:00:00+00:00"


def test_no_previous_run(tmp_path: Path) -> None:
    assert previous_run(tmp_path / "missing", "abc:tuning") is None
