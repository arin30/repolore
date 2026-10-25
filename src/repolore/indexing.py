import logging
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import batched

import numpy as np

from repolore.chunking import Chunk
from repolore.embedding import Embedder, Vectors
from repolore.store import EmbeddingCache, cache_key, write_index

BATCH_SIZE = 32
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildStats:
    chunks: int
    embedded: int


def build_index(
    chunks: Sequence[Chunk],
    embedder: Embedder,
    db: sqlite3.Connection,
    cache: EmbeddingCache,
    settings: Mapping[str, str] | None = None,
) -> BuildStats:
    keys = [cache_key(embedder.model_id, chunk.embedded_text) for chunk in chunks]
    vectors = cache.get(set(keys))
    missing = {key: chunk.embedded_text for key, chunk in zip(keys, chunks, strict=True) if key not in vectors}
    # Similar lengths per batch mean less padding, which matters on a CPU.
    pending = sorted(missing.items(), key=lambda item: len(item[1]))
    done = 0
    for batch in batched(pending, BATCH_SIZE):
        batch_keys = [key for key, _ in batch]
        embedded = embedder.embed_documents([text for _, text in batch])
        cache.put(batch_keys, embedded)
        vectors.update(zip(batch_keys, embedded, strict=True))
        done += len(batch)
        log.info("embedded %d of %d chunks", done, len(pending))
    write_index(db, chunks, _stack(vectors, keys, embedder.dim), embedder.model_id, embedder.dim, settings)
    return BuildStats(chunks=len(chunks), embedded=len(pending))


def _stack(vectors: dict[str, Vectors], keys: list[str], dim: int) -> Vectors:
    return np.stack([vectors[key] for key in keys]) if keys else np.empty((0, dim), dtype=np.float32)
