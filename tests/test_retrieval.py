import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest
from conftest import FakeEmbedder

from repolore.chunking import Chunk
from repolore.retrieval import search
from repolore.store import Filters, StoreError, connect, write_index


@pytest.fixture
def index(tmp_path: Path, corpus: list[Chunk], fake_embedder: FakeEmbedder) -> Iterator[sqlite3.Connection]:
    with closing(connect(tmp_path / "test.db")) as db:
        vectors = fake_embedder.embed_documents([chunk.embedded_text for chunk in corpus])
        write_index(db, corpus, vectors, fake_embedder.model_id, fake_embedder.dim)
        yield db


def test_search_embeds_the_query_and_returns_the_nearest_chunks(
    index: sqlite3.Connection, fake_embedder: FakeEmbedder
) -> None:
    hits = search(index, fake_embedder, "solar battery over Modbus", 1, Filters())

    assert [hit.url for hit in hits] == ["https://github.com/octo/repo/issues/3"]


def test_search_refuses_an_index_built_with_another_model(
    index: sqlite3.Connection, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(FakeEmbedder, "model_id", "other-model")

    with pytest.raises(StoreError, match="built with fake-bag-of-words, not other-model"):
        search(index, fake_embedder, "anything", 5, Filters())
