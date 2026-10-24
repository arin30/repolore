import re
from collections.abc import Iterator
from dataclasses import dataclass

from markdown_it import MarkdownIt
from pydantic import BaseModel, ConfigDict

from repolore.documents import Document, DocumentKind

MAX_TOKENS = 400
CHARS_PER_TOKEN = 4
FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
MARKDOWN = MarkdownIt("commonmark").enable("table")
SHORT_SHA = 7


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    document: Document
    section: str | None
    header: str
    text: str

    @property
    def embedded_text(self) -> str:
        return f"{self.header}\n\n{self.text}"

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.embedded_text)


@dataclass(frozen=True)
class Section:
    heading: str | None
    text: str


@dataclass(frozen=True)
class _Block:
    text: str
    heading: str | None


def estimate_tokens(text: str) -> int:
    # A stand-in until the embedding model's own tokenizer is wired in; ~4 characters per token for English prose.
    return -(-len(text) // CHARS_PER_TOKEN)


def chunk_document(document: Document) -> list[Chunk]:
    sections = split_markdown(document.body) or [Section(heading=None, text=document.title)]
    return [
        Chunk(
            id=f"{document.url}::{ordinal}",
            document=document,
            section=section.heading,
            header=_header(document, section.heading),
            text=section.text,
        )
        for ordinal, section in enumerate(sections)
    ]


def split_markdown(text: str, max_tokens: int = MAX_TOKENS) -> list[Section]:
    sections: list[Section] = []
    heading: str | None = None
    blocks: list[str] = []
    for block in _blocks(_fence_front_matter(text.replace("\r\n", "\n"))):
        if block.heading is not None:
            sections += _pack(heading, blocks, max_tokens)
            heading, blocks = block.heading, [block.text]
        else:
            blocks.append(block.text)
    return sections + _pack(heading, blocks, max_tokens)


def _fence_front_matter(text: str) -> str:
    return FRONT_MATTER.sub(lambda match: f"```yaml\n{match.group(1)}\n```\n", text, count=1)


def _blocks(text: str) -> Iterator[_Block]:
    lines = text.split("\n")
    tokens = MARKDOWN.parse(text)
    for index, token in enumerate(tokens):
        if token.level != 0 or token.map is None or token.nesting < 0:
            continue
        start, end = token.map
        heading = tokens[index + 1].content.strip() if token.type == "heading_open" else None
        yield _Block(text="\n".join(lines[start:end]).strip(), heading=heading)


def _pack(heading: str | None, blocks: list[str], max_tokens: int) -> list[Section]:
    content = blocks[1:] if heading is not None else blocks
    if not any(content):
        return []
    sections: list[Section] = []
    current: list[str] = []
    for block in blocks:
        if current and estimate_tokens("\n\n".join([*current, block])) > max_tokens:
            sections.append(Section(heading, "\n\n".join(current)))
            current = []
        current.append(block)
    return [*sections, Section(heading, "\n\n".join(current))]


def _header(document: Document, section: str | None) -> str:
    parts = [document.repo, _reference(document), document.title, section]
    return " · ".join(part for part in parts if part)


def _reference(document: Document) -> str | None:
    last_segment = document.url.rsplit("/", 1)[-1]
    match document.kind:
        case DocumentKind.PR:
            return f"PR #{last_segment}"
        case DocumentKind.ISSUE:
            return f"issue #{last_segment}"
        case DocumentKind.COMMIT:
            return f"commit {last_segment[:SHORT_SHA]}"
        case _:
            return None
