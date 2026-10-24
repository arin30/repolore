import math
from collections.abc import Sequence

Ranking = Sequence[int | None]
"""For each retrieved chunk, in rank order: the index of the expected source it matches, or None."""


def first_ranks(ranking: Ranking, expected: int) -> list[int | None]:
    ranks: list[int | None] = [None] * expected
    for rank, source in enumerate(ranking, start=1):
        if source is not None and ranks[source] is None:
            ranks[source] = rank
    return ranks


def recall_at(ranks: Sequence[int | None], k: int) -> float:
    return sum(rank is not None and rank <= k for rank in ranks) / len(ranks)


def reciprocal_rank(ranks: Sequence[int | None]) -> float:
    found = [rank for rank in ranks if rank is not None]
    return 1 / min(found) if found else 0.0


def ndcg_at(ranks: Sequence[int | None], k: int) -> float:
    # Binary relevance, and only the first chunk of each source counts, so a source split into many chunks
    # can't inflate the score.
    gain = sum(1 / math.log2(rank + 1) for rank in ranks if rank is not None and rank <= k)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(ranks), k) + 1))
    return gain / ideal
