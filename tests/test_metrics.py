import ir_measures
import pytest
from ir_measures import RR, R, nDCG

from repolore.evals.metrics import first_ranks, ndcg_at, recall_at, reciprocal_rank


def test_first_ranks_keep_only_each_sources_first_chunk() -> None:
    assert first_ranks([None, 1, 1, None, 0, 1], expected=3) == [5, 2, None]


@pytest.mark.parametrize(
    ("ranks", "k", "recall"),
    [([1], 5, 1.0), ([5], 5, 1.0), ([6], 5, 0.0), ([None], 5, 0.0), ([2, None], 5, 0.5), ([1, 7], 10, 1.0)],
)
def test_recall_counts_sources_found_within_k(ranks: list[int | None], k: int, recall: float) -> None:
    assert recall_at(ranks, k) == recall


@pytest.mark.parametrize(("ranks", "rr"), [([1], 1.0), ([4, 2], 0.5), ([None, 3], 1 / 3), ([None], 0.0)])
def test_reciprocal_rank_uses_the_best_ranked_source(ranks: list[int | None], rr: float) -> None:
    assert reciprocal_rank(ranks) == pytest.approx(rr)


def test_ndcg_is_one_for_a_perfect_ranking_and_zero_for_none() -> None:
    assert ndcg_at([1, 2], 10) == pytest.approx(1.0)
    assert ndcg_at([None, None], 10) == 0.0
    assert ndcg_at([11], 10) == 0.0


RUNS = {
    "one-source-first": ["a", "x", "y"],
    "one-source-late": ["x", "y", "z", "a"],
    "two-sources": ["x", "b", "y", "a"],
    "beyond-cutoff": ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "a"],
    "missing": ["x", "y"],
}
QRELS = {
    "one-source-first": ["a"],
    "one-source-late": ["a"],
    "two-sources": ["a", "b"],
    "beyond-cutoff": ["a"],
    "missing": ["a", "b", "c"],
}


def test_metrics_agree_with_ir_measures() -> None:
    qrels = [ir_measures.Qrel(qid, doc, 1) for qid, docs in QRELS.items() for doc in docs]
    run = [ir_measures.ScoredDoc(qid, doc, -rank) for qid, docs in RUNS.items() for rank, doc in enumerate(docs)]
    reference = list(ir_measures.iter_calc([R @ 5, R @ 10, RR, nDCG @ 10], qrels, run))

    assert len(reference) == 4 * len(RUNS)

    for metric in reference:
        expected = QRELS[metric.query_id]
        ranking = [expected.index(doc) if doc in expected else None for doc in RUNS[metric.query_id]]
        ranks = first_ranks(ranking, len(expected))
        ours = {
            "R@5": recall_at(ranks, 5),
            "R@10": recall_at(ranks, 10),
            "RR": reciprocal_rank(ranks),
            "nDCG@10": ndcg_at(ranks, 10),
        }[str(metric.measure)]
        assert ours == pytest.approx(metric.value), (metric.query_id, metric.measure)
