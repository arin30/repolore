import sqlite3

from repolore.embedding import Embedder
from repolore.store import Filters, Hit, StoreError, index_model, nearest


def search(db: sqlite3.Connection, embedder: Embedder, query: str, k: int, filters: Filters) -> list[Hit]:
    model_id, _ = index_model(db)
    if model_id != embedder.model_id:
        raise StoreError(f"The index was built with {model_id}, not {embedder.model_id}; rebuild it.")
    return nearest(db, embedder.embed_query(query), k, filters)
