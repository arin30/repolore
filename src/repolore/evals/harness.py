import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from statistics import fmean, median

from repolore.evals.metrics import first_ranks, ndcg_at, recall_at, reciprocal_rank
from repolore.evals.questions import ExpectedSource, Question, QuestionKind
from repolore.store import Hit

SEARCH_DEPTH = 10
METRICS = ("recall@5", "recall@10", "mrr", "ndcg@10")


@dataclass(frozen=True)
class QuestionResult:
    question: Question
    ranks: list[int | None]
    hits: list[Hit]
    seconds: float = 0.0

    @property
    def answerable(self) -> bool:
        return self.question.kind is not QuestionKind.UNANSWERABLE

    @property
    def top_score(self) -> float | None:
        return self.hits[0].score if self.hits else None

    @property
    def missed(self) -> bool:
        return self.answerable and recall_at(self.ranks, 5) < 1


def evaluate(
    questions: Iterable[Question],
    search: Callable[[str], list[Hit]],
    clock: Callable[[], float] = time.perf_counter,
) -> list[QuestionResult]:
    results = []
    for question in questions:
        start = clock()
        hits = search(question.question)
        seconds = clock() - start
        ranking = [_matching_source(hit, question.expected) for hit in hits]
        results.append(QuestionResult(question, first_ranks(ranking, len(question.expected)), hits, seconds))
    return results


def _matching_source(hit: Hit, expected: Sequence[ExpectedSource]) -> int | None:
    return next(
        (
            index
            for index, source in enumerate(expected)
            if source.url == hit.url and source.section in (None, hit.section)
        ),
        None,
    )


def summarize(results: Sequence[QuestionResult]) -> dict[str, float]:
    answerable = [result for result in results if result.answerable]
    if not answerable:
        return {}
    return {
        "recall@5": fmean(recall_at(result.ranks, 5) for result in answerable),
        "recall@10": fmean(recall_at(result.ranks, 10) for result in answerable),
        "mrr": fmean(reciprocal_rank(result.ranks) for result in answerable),
        "ndcg@10": fmean(ndcg_at(result.ranks, 10) for result in answerable),
    }


def recall_by_kind(results: Sequence[QuestionResult]) -> dict[QuestionKind, tuple[float, int]]:
    by_kind: dict[QuestionKind, list[float]] = {}
    for result in results:
        if result.answerable:
            by_kind.setdefault(result.question.kind, []).append(recall_at(result.ranks, 5))
    return {kind: (fmean(scores), len(scores)) for kind, scores in by_kind.items()}


def median_search_ms(results: Sequence[QuestionResult]) -> float | None:
    # The median, because the first search also pays for loading the model.
    return median(result.seconds for result in results) * 1000 if results else None


def mean_top_score(results: Sequence[QuestionResult], answerable: bool) -> float | None:
    scores = [
        result.top_score for result in results if result.answerable is answerable and result.top_score is not None
    ]
    return fmean(scores) if scores else None
