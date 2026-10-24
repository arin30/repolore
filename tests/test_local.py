import sys
import types
from typing import Any, ClassVar

import numpy as np
import pytest

from repolore.embedding import local
from repolore.embedding.local import LocalEmbedder


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
            "revision": local.REVISION,
            "trust_remote_code": True,
            "truncate_dim": 1024,
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
    assert LocalEmbedder().model_id == f"voyageai/voyage-4-nano@{local.REVISION[:12]}"
    assert LocalEmbedder().dim == 1024


@pytest.mark.live
def test_real_model_embeds_unit_vectors() -> None:
    embedder = LocalEmbedder()

    assert float(np.linalg.norm(embedder.embed_query("why was the release delayed?"))) == pytest.approx(1.0)
