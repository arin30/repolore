import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from repolore.evals.harness import QuestionResult, median_search_ms, summarize

Run = dict[str, Any]


def run_record(config: dict[str, Any], results: Sequence[QuestionResult]) -> Run:
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": config,
        "metrics": summarize(results),
        "search_ms": median_search_ms(results),
        "questions": [
            {"id": result.question.id, "ranks": result.ranks, "top_score": result.top_score} for result in results
        ],
    }


def previous_run(directory: Path, questions: str) -> Run | None:
    """The latest run scored on the same question set, so metric changes compare like with like."""
    runs = (json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"), reverse=True))
    return next((run for run in runs if run["config"]["questions"] == questions), None)


def save_run(directory: Path, run: Run) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run['timestamp'].replace(':', '').replace('+0000', 'Z')}.json"
    path.write_text(json.dumps(run, indent=1), encoding="utf-8")
    return path
