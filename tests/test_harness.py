import pytest

from repolore.evals.harness import evaluate, mean_top_score, recall_by_kind, summarize
from repolore.evals.questions import Question, QuestionKind
from repolore.store import Hit

PR = "https://github.com/octo/repo/pull/1"
ISSUE = "https://github.com/octo/repo/issues/2"
DOC = "https://github.com/octo/repo/blob/main/README.md"


def hit(url: str, score: float = 0.5, section: str | None = None) -> Hit:
    return Hit(
        chunk_id=f"{url}::0", url=url, repo="octo/repo", kind="pr", section=section, header="h", text="t", score=score
    )


def question(qid: str, kind: str = "factual", sources: list[dict[str, str]] | None = None) -> Question:
    expected = [{"url": PR}] if sources is None else sources
    return Question.model_validate({"id": qid, "question": qid, "kind": kind, "expected": expected, "answer": "a"})


def search_returning(results: dict[str, list[Hit]]):
    return lambda text: results[text]


def test_sources_are_ranked_by_their_first_chunk() -> None:
    q = question("two", sources=[{"url": PR}, {"url": ISSUE}])

    [result] = evaluate([q], search_returning({"two": [hit(DOC), hit(ISSUE), hit(PR), hit(ISSUE)]}))

    assert result.ranks == [3, 2]
    assert not result.missed


def test_section_must_match_when_the_question_names_one() -> None:
    q = question("section", sources=[{"url": PR, "section": "Change"}])
    hits = [hit(PR, section="Problem"), hit(PR, section="Change")]

    [result] = evaluate([q], search_returning({"section": hits}))

    assert result.ranks == [2]


def test_source_outside_the_top_five_is_a_miss() -> None:
    [result] = evaluate([question("late")], search_returning({"late": [hit(DOC)] * 5 + [hit(PR)]}))

    assert result.ranks == [6]
    assert result.missed


def test_summary_scores_only_answerable_questions() -> None:
    questions = [question("found"), question("lost"), question("none", kind="unanswerable", sources=[])]
    results = evaluate(
        questions,
        search_returning({"found": [hit(PR, 0.9)], "lost": [hit(DOC, 0.6)], "none": [hit(DOC, 0.3)]}),
    )

    assert summarize(results) == {"recall@5": 0.5, "recall@10": 0.5, "mrr": 0.5, "ndcg@10": 0.5}
    assert not results[2].missed
    assert mean_top_score(results, answerable=True) == pytest.approx(0.75)
    assert mean_top_score(results, answerable=False) == pytest.approx(0.3)


def test_summary_of_only_unanswerable_questions_is_empty() -> None:
    results = evaluate([question("none", kind="unanswerable", sources=[])], search_returning({"none": []}))

    assert summarize(results) == {}
    assert results[0].top_score is None
    assert mean_top_score(results, answerable=False) is None


def test_recall_is_broken_down_by_kind() -> None:
    questions = [question("a"), question("b", kind="rationale"), question("c", kind="rationale")]
    results = evaluate(questions, search_returning({"a": [hit(PR)], "b": [hit(PR)], "c": []}))

    assert recall_by_kind(results) == {QuestionKind.FACTUAL: (1.0, 1), QuestionKind.RATIONALE: (0.5, 2)}
