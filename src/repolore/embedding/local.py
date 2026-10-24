from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from repolore.embedding import Vectors

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

MODEL = "voyageai/voyage-4-nano"
# The model runs its own code (trust_remote_code), so it is pinned to a reviewed commit.
REVISION = "5fb34e870f49f19e79c94d5e118f5ee730133279"
DIM = 1024
BATCH_SIZE = 16


class LocalEmbedder:
    """voyage-4-nano on the CPU, in the same embedding space as Voyage's hosted voyage-4 models."""

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None

    @property
    def model_id(self) -> str:
        return f"{MODEL}@{REVISION[:12]}"

    @property
    def dim(self) -> int:
        return DIM

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        vectors = self._loaded().encode_document(list(texts), batch_size=BATCH_SIZE, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> Vectors:
        return np.asarray(self._loaded().encode_query(text, normalize_embeddings=True), dtype=np.float32)

    def _loaded(self) -> "SentenceTransformer":
        if self._model is None:
            # Imported on first use: loading torch takes seconds, and most commands never embed anything.
            import torch
            from sentence_transformers import SentenceTransformer

            # The checkpoint is BF16, which CPUs without native BF16 math run about 9x slower than float32.
            self._model = SentenceTransformer(
                MODEL,
                revision=REVISION,
                trust_remote_code=True,
                truncate_dim=DIM,
                device="cpu",
                model_kwargs={"dtype": torch.float32},
            )
        return self._model
