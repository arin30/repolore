import sys
import types
from typing import Any, ClassVar

import pytest

from repolore import reranking
from repolore.reranking import CrossEncoderReranker
from repolore.store import Hit


class StubCrossEncoder:
    loads: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, name: str, **kwargs: Any) -> None:
        StubCrossEncoder.loads.append({"name": name, **kwargs})

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(len(passage)) for _, passage in pairs]


@pytest.fixture(autouse=True)
def stub_cross_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    StubCrossEncoder.loads = []
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=StubCrossEncoder))


def hit(text: str) -> Hit:
    return Hit(
        text, f"https://github.com/octo/repo/pull/{len(text)}", "octo/repo", "pr", "t", None, None, None, "h", text, 0.1
    )


def test_hits_are_reordered_by_the_cross_encoder_score() -> None:
    reranked = CrossEncoderReranker().rerank("q", [hit("a"), hit("abc"), hit("ab")])

    assert [h.text for h in reranked] == ["abc", "ab", "a"]
    assert reranked[0].score == len("h\n\nabc")


def test_model_is_loaded_once_and_pinned() -> None:
    reranker = CrossEncoderReranker()

    reranker.rerank("q", [hit("a")])
    reranker.rerank("q", [hit("b")])

    assert StubCrossEncoder.loads == [
        {"name": reranking.MODEL, "revision": reranking.REVISION, "max_length": reranking.MAX_LENGTH, "device": "cpu"}
    ]


def test_no_candidates_needs_no_model() -> None:
    assert CrossEncoderReranker().rerank("q", []) == []
    assert StubCrossEncoder.loads == []
