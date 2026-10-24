from contextlib import closing
from pathlib import Path

import pytest
from conftest import DocumentFactory, FakeEmbedder

from repolore.chunking import Chunk, chunk_document
from repolore.indexing import BATCH_SIZE, build_index
from repolore.store import EmbeddingCache, connect, index_model


def build(tmp_path: Path, chunks: list[Chunk], embedder: FakeEmbedder, name: str = "test") -> int:
    with closing(connect(tmp_path / f"{name}.db")) as index, closing(connect(tmp_path / "cache.db")) as cache:
        stats = build_index(chunks, embedder, index, EmbeddingCache(cache))
    assert stats.chunks == len(chunks)
    return stats.embedded


def test_first_build_embeds_every_chunk(tmp_path: Path, corpus: list[Chunk], fake_embedder: FakeEmbedder) -> None:
    assert build(tmp_path, corpus, fake_embedder) == len(corpus)


def test_second_build_embeds_nothing(tmp_path: Path, corpus: list[Chunk], fake_embedder: FakeEmbedder) -> None:
    build(tmp_path, corpus, fake_embedder)

    assert build(tmp_path, corpus, fake_embedder) == 0


def test_only_changed_chunks_are_embedded_again(
    tmp_path: Path, corpus: list[Chunk], fake_embedder: FakeEmbedder, make_document: DocumentFactory
) -> None:
    build(tmp_path, corpus, fake_embedder)
    changed = chunk_document(make_document(1, "Retry with linear backoff instead."))

    assert build(tmp_path, [*changed, *corpus[1:]], fake_embedder) == 1


def test_the_cache_is_shared_between_indexes(tmp_path: Path, corpus: list[Chunk], fake_embedder: FakeEmbedder) -> None:
    build(tmp_path, corpus, fake_embedder, name="baseline")

    assert build(tmp_path, corpus, fake_embedder, name="experiment") == 0


def test_many_chunks_are_embedded_in_batches(
    tmp_path: Path, fake_embedder: FakeEmbedder, make_document: DocumentFactory
) -> None:
    chunks = [chunk for n in range(1, 2 * BATCH_SIZE + 2) for chunk in chunk_document(make_document(n, f"Text {n}."))]

    assert build(tmp_path, chunks, fake_embedder) == len(chunks)
    assert len(fake_embedder.embedded) == len(chunks)


def test_empty_corpus_builds_an_empty_index(tmp_path: Path, fake_embedder: FakeEmbedder) -> None:
    build(tmp_path, [], fake_embedder)

    with closing(connect(tmp_path / "test.db")) as index:
        assert index_model(index) == ("fake-bag-of-words", 64)
        assert index.execute("select count(*) from chunks").fetchone() == (0,)


def test_progress_is_logged(
    tmp_path: Path, corpus: list[Chunk], fake_embedder: FakeEmbedder, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO"):
        build(tmp_path, corpus, fake_embedder)

    assert f"embedded {len(corpus)} of {len(corpus)} chunks" in caplog.text
