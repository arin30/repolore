import sys
import types
from typing import Any, ClassVar

import numpy as np
import pytest

from repolore.embedding import local
from repolore.embedding.local import LocalEmbedder, UnknownModelError, embedder_for


class StubModel:
    loads: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, name: str, **kwargs: Any) -> None:
        StubModel.loads.append({"name": name, **kwargs})

    def encode_document(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        assert kwargs == {"batch_size": local.BATCH_SIZE, "normalize_embeddings": True}
        return [[1.0, 0.0]] * len(texts)

    def encode_query(self, text: str, **kwargs: Any) -> list[float]:
        assert kwargs == {"normalize_embeddings": True}
        return [0.0, 1.0]


@pytest.fixture
def stub_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    StubModel.loads = []
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=StubModel))


@pytest.mark.usefixtures("stub_sentence_transformers")
def test_model_is_loaded_once_pinned_and_in_float32() -> None:
    import torch

    embedder = LocalEmbedder()

    embedder.embed_query("a")
    embedder.embed_documents(["b", "c"])

    assert StubModel.loads == [
        {
            "name": "voyageai/voyage-4-nano",
            "revision": local.VOYAGE_NANO.revision,
            "trust_remote_code": True,
            "truncate_dim": 1024,
            "prompts": None,
            "device": "cpu",
            "model_kwargs": {"dtype": torch.float32},
        }
    ]


@pytest.mark.usefixtures("stub_sentence_transformers")
def test_vectors_are_float32_arrays() -> None:
    embedder = LocalEmbedder()

    documents = embedder.embed_documents(["b", "c"])
    query = embedder.embed_query("a")

    assert documents.dtype == np.float32
    assert documents.shape == (2, 2)
    assert query.tolist() == [0.0, 1.0]


@pytest.mark.usefixtures("stub_sentence_transformers")
def test_model_id_names_the_pinned_revision() -> None:
    assert LocalEmbedder().model_id == f"voyageai/voyage-4-nano@{local.VOYAGE_NANO.revision[:12]}"
    assert LocalEmbedder().dim == 1024


@pytest.mark.live
def test_real_model_embeds_unit_vectors() -> None:
    embedder = LocalEmbedder()

    assert float(np.linalg.norm(embedder.embed_query("why was the release delayed?"))) == pytest.approx(1.0)


@pytest.mark.usefixtures("stub_sentence_transformers")
def test_bge_small_gets_its_query_instruction_and_no_remote_code() -> None:
    LocalEmbedder(local.BGE_SMALL).embed_query("a")

    load = StubModel.loads[0]
    assert load["name"] == "BAAI/bge-small-en-v1.5"
    assert load["trust_remote_code"] is False
    assert load["truncate_dim"] == 384
    assert load["prompts"] == {"query": "Represent this sentence for searching relevant passages: "}


def test_embedder_for_matches_the_index_model_and_reuses_it() -> None:
    embedder = embedder_for(local.BGE_SMALL.model_id)

    assert embedder.model_id == "BAAI/bge-small-en-v1.5@5c38ec7c405e"
    assert embedder.dim == 384
    assert embedder_for(local.BGE_SMALL.model_id) is embedder


def test_embedder_for_an_unknown_model_says_to_rebuild() -> None:
    with pytest.raises(UnknownModelError, match="rebuild it"):
        embedder_for("someone/else@123")
