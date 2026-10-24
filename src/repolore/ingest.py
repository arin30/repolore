import logging
from pathlib import Path

from repolore.documents import Document, documents_from
from repolore.sources.github import GitHubClient, RawRepo, fetch_repo

log = logging.getLogger(__name__)


def ingest(repos: list[str], raw_dir: Path, client: GitHubClient | None) -> list[Document]:
    documents: list[Document] = []
    for repo in repos:
        cache = raw_dir / repo
        if client:
            log.info("fetching %s", repo)
            fetch_repo(client, repo).save(cache)
        documents += documents_from(RawRepo.load(cache))
    return documents
