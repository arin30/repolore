import shutil
from pathlib import Path

import pytest

from repolore import ingest as ingest_module
from repolore.documents import DocumentKind
from repolore.ingest import ingest
from repolore.sources.github import GitHubClient, RawRepo

FIXTURE = Path(__file__).parent / "fixtures" / "crashcause"
REPO = "ryckakas/crashcause"


def test_offline_ingest_reads_the_raw_cache(tmp_path: Path) -> None:
    shutil.copytree(FIXTURE, tmp_path / REPO)

    documents = ingest([REPO], tmp_path, client=None)

    assert {document.kind for document in documents} == {
        DocumentKind.PR,
        DocumentKind.ISSUE,
        DocumentKind.COMMIT,
        DocumentKind.DOC,
    }


def test_online_ingest_refreshes_the_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fetched: list[str] = []

    def fake_fetch(_: GitHubClient, repo: str) -> RawRepo:
        fetched.append(repo)
        return RawRepo.load(FIXTURE)

    monkeypatch.setattr(ingest_module, "fetch_repo", fake_fetch)

    documents = ingest([REPO], tmp_path, client=GitHubClient.connect("unused"))

    assert fetched == [REPO]
    assert RawRepo.load(tmp_path / REPO) == RawRepo.load(FIXTURE)
    assert len(documents) == 7


def test_offline_ingest_without_a_cache_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ingest([REPO], tmp_path, client=None)
