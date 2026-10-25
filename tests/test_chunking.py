from datetime import UTC, datetime

import pytest

from repolore.chunking import Section, chunk_document, estimate_tokens, split_markdown
from repolore.documents import Document, DocumentKind, Visibility

FENCE = "```"
LIMIT = 400


def document(kind: DocumentKind = DocumentKind.PR, body: str = "Body.", **changes: object) -> Document:
    fields: dict[str, object] = {
        "url": "https://github.com/octo/repo/pull/17",
        "repo": "octo/repo",
        "kind": kind,
        "visibility": Visibility.PUBLIC,
        "title": "Add the thing",
        "body": body,
        "author": "octocat",
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    return Document.model_validate(fields | changes)


def paragraph(tokens: int, word: str = "word") -> str:
    return " ".join([word] * (tokens * 4 // (len(word) + 1)))


def test_sections_follow_headings() -> None:
    text = "Intro line.\n\n## Problem\n\nIt breaks.\n\n## Change\n\nFix it.\n\nAnd test it."

    assert split_markdown(text) == [
        Section(None, "Intro line."),
        Section("Problem", "## Problem\n\nIt breaks."),
        Section("Change", "## Change\n\nFix it.\n\nAnd test it."),
    ]


@pytest.mark.parametrize("text", ["", "   \n\n  "])
def test_empty_text_has_no_sections(text: str) -> None:
    assert split_markdown(text) == []


def test_one_heading_is_one_section() -> None:
    assert split_markdown("# Title\n\nSome text.") == [Section("Title", "# Title\n\nSome text.")]


def test_heading_without_content_is_skipped() -> None:
    sections = split_markdown("## Change\n\n### Parser\n\nRewritten.")

    assert [section.heading for section in sections] == ["Parser"]


def test_oversized_section_is_split_between_blocks() -> None:
    blocks = [paragraph(150, word=f"w{n}") for n in range(5)]

    sections = split_markdown("## Notes\n\n" + "\n\n".join(blocks), LIMIT)

    assert len(sections) > 1
    assert {section.heading for section in sections} == {"Notes"}
    assert all(estimate_tokens(section.text) <= LIMIT for section in sections)
    assert "\n\n".join(section.text for section in sections).endswith(blocks[-1])


def test_section_exactly_at_the_limit_stays_whole() -> None:
    text = "a" * (LIMIT * 4)

    assert split_markdown(text, LIMIT) == [Section(None, text)]


def test_code_fence_is_never_split_or_read_as_headings() -> None:
    fence = f"{FENCE}sh\n# not a heading\n" + "\n".join(f"echo {n}" * 20 for n in range(40)) + f"\n{FENCE}"

    sections = split_markdown(f"## Usage\n\nRun:\n\n{fence}\n\nDone.", LIMIT)

    assert [section.heading for section in sections] == ["Usage", "Usage", "Usage"]
    assert sections[1].text == fence
    assert estimate_tokens(fence) > LIMIT


def test_sections_stay_whole_by_default() -> None:
    blocks = [paragraph(300, word=f"w{n}") for n in range(3)]

    assert len(split_markdown("## Notes\n\n" + "\n\n".join(blocks))) == 1


def test_table_stays_one_block() -> None:
    table = "| a | b |\n|---|---|\n| 1 | 2 |"

    assert split_markdown(f"## Data\n\n{table}") == [Section("Data", f"## Data\n\n{table}")]


def test_front_matter_is_kept_as_yaml_not_a_heading() -> None:
    sections = split_markdown("---\nname: review\ndescription: Reviews code\n---\n\n# Review\n\nSteps.")

    assert sections[0] == Section(None, f"{FENCE}yaml\nname: review\ndescription: Reviews code\n{FENCE}")
    assert sections[1].heading == "Review"


def test_windows_line_endings_are_normalized() -> None:
    assert split_markdown("## A\r\n\r\nText.") == [Section("A", "## A\n\nText.")]


def test_pull_request_chunks_carry_a_header_per_section() -> None:
    chunks = chunk_document(document(body="## Problem\n\nSlow.\n\n## Change\n\nFaster."))

    assert [chunk.header for chunk in chunks] == [
        "octo/repo · PR #17 · Add the thing · Problem",
        "octo/repo · PR #17 · Add the thing · Change",
    ]
    assert [chunk.id for chunk in chunks] == [
        "https://github.com/octo/repo/pull/17::0",
        "https://github.com/octo/repo/pull/17::1",
    ]
    assert chunks[0].embedded_text == "octo/repo · PR #17 · Add the thing · Problem\n\n## Problem\n\nSlow."
    assert chunks[0].tokens == estimate_tokens(chunks[0].embedded_text)


@pytest.mark.parametrize(
    ("kind", "url", "title", "header"),
    [
        (DocumentKind.ISSUE, "https://github.com/octo/repo/issues/5", "Crash", "octo/repo · issue #5 · Crash"),
        (
            DocumentKind.COMMIT,
            "https://github.com/octo/repo/commit/0123456789abcdef",
            "Fix",
            "octo/repo · commit 0123456 · Fix",
        ),
        (DocumentKind.DOC, "https://github.com/octo/repo/blob/main/docs/a.md", "docs/a.md", "octo/repo · docs/a.md"),
        (
            DocumentKind.REVIEW,
            "https://github.com/octo/repo/pull/1#discussion_r9",
            "PR #1 · a.go:3",
            "octo/repo · PR #1 · a.go:3",
        ),
    ],
)
def test_header_names_the_source(kind: DocumentKind, url: str, title: str, header: str) -> None:
    chunks = chunk_document(document(kind=kind, url=url, title=title, body="Plain text."))

    assert chunks[0].header == header


def test_document_without_body_is_one_chunk_of_its_title() -> None:
    chunks = chunk_document(document(kind=DocumentKind.COMMIT, url="https://github.com/octo/repo/commit/abc", body=""))

    assert len(chunks) == 1
    assert chunks[0].text == "Add the thing"
    assert chunks[0].section is None


def test_estimate_rounds_up() -> None:
    assert [estimate_tokens(text) for text in ["", "a", "abcd", "abcde"]] == [0, 1, 1, 2]


def test_max_tokens_controls_where_long_sections_split() -> None:
    body = "## Notes\n\n" + "\n\n".join(paragraph(60, word=f"w{n}") for n in range(4))

    assert len(chunk_document(document(body=body))) == 1
    assert len(chunk_document(document(body=body), max_tokens=100)) == 4
