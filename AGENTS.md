# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in
this repository. `CLAUDE.md` is a symlink to this file.

## What this is

`repolore` is a hand-built RAG pipeline over one person's engineering history on GitHub: PR bodies,
review threads, issues, PR-less commits, repo markdown (`README`, `AGENTS.md`, `docs/**`) and local
`.plans/` documents. It is searched from a Typer CLI (`search`, `eval`) and served read-only
to Claude Code through a stdio MCP server. Claude Code writes the answers; repolore itself calls no LLM.

It is a **learning project first**: every stage (fetch, chunk, embed, retrieve, generate,
evaluate) is written by hand so it stays visible. No LangChain, LlamaIndex, RAGAS or other
RAG/eval frameworks. Prefer the approach that exposes how a stage works over one that hides it.

It runs locally for a single user. Other people use it by running their own copy (`uvx` + `claude
mcp add`) with their own GitHub token. No model API key is needed: embeddings run locally by default, and
Voyage's hosted API is an option. There is no hosted service and no GitHub App.

## Commands

Python 3.12 (pinned in `.python-version`), managed with uv. Code that opens files always passes
`encoding="utf-8"`, because the Windows default cp1252 codec breaks on GitHub content. Don't lean on
`PYTHONUTF8=1` instead: `uvx` users won't have it set, and CI's Windows job runs without it.

```sh
uv sync                     # install the project and dev tools from uv.lock
uv run repolore --help      # the CLI
uv run repolore ingest      # fetch the repos in repolore.toml into the raw cache; --offline rebuilds from it
uv run repolore inspect     # chunk the cached documents: counts, token histogram, random samples
uv run ruff format          # format
uv run ruff check           # lint
uv run pyright              # type-check (standard mode)
uv run bonsai-lint          # cognitive complexity, threshold in bonsai-lint.toml
uv run pytest               # offline tests; `live` tests are deselected
uv run pytest -m live       # tests that call real GitHub / Voyage APIs, run by hand
uv run pytest --cov         # tests with the coverage floor CI enforces (90%, branch coverage)
uv audit --locked --preview-features audit-command   # known advisories in uv.lock
```

Run format, lint, type-check and tests before considering a change done. CI
(`.github/workflows/ci.yml`) runs the same gates plus a few that only run there. Each job is a
required status check on `main`, except `audit`:

- `lint`, `typecheck`: ruff check, ruff format --check, bonsai-lint (cognitive complexity over 15 fails), pyright.
- `test (ubuntu-latest)`, `test (windows-latest)`: pytest with the coverage floor.
- `build`: builds the wheel and runs the installed `repolore` from it.
- `typos`: spelling.
- `zizmor`, `actionlint`: workflow security and correctness, and shellcheck over `.githooks/`.
- `audit`: `uv audit`. It runs on every PR but isn't required.

CodeQL (Python and Actions) runs through GitHub's default setup, outside `ci.yml`. A ruleset rule
blocks merging on an error-level alert or a high or critical security alert.

Every action is pinned to a commit SHA with its version in a comment, and Dependabot moves both. A
renamed or added job also needs `.github/rulesets/protect-main.json` updated and re-applied (see
`.github/rulesets/README.md`).

## Workflow

`main` only changes through pull requests. A GitHub ruleset blocks direct pushes, force-pushes and
deletion. The versioned hooks in `.githooks/` block direct commits and pushes to `main` locally;
enable them once per clone with `git config core.hooksPath .githooks`. Work happens on
`feat/…`, `fix/…`, `chore/…` or `docs/…` branches.

## Architecture

The plan is in `.plans/initial-build/plan.md`, which is local and gitignored. Its decision log
(D1–D25) is the reasoning behind everything below. The package grows into this layout
(`src/repolore/`), one module per pipeline stage:

- `paths.py` resolves the per-user directories through `platformdirs`. Config is
  `%APPDATA%\repolore\repolore.toml` (roaming). Data is `%LOCALAPPDATA%\repolore` (local),
  holding the raw API cache and the DB files, and `--data-dir` overrides it. **The repo never
  contains corpus data.** It contains private code, so it stays out of git.
- `config.py` loads `repolore.toml` (for now just `repos = ["owner/name", …]`).
- `sources/` fetches GitHub data with a sync `httpx` client: GraphQL for PRs, issues and commits
  (`github.graphql`), REST for the markdown tree and blobs. `models.py` types the responses. Raw
  responses are cached as `<data-dir>/raw/<owner>/<repo>/*.json`, so chunking and embedding never
  need the network. Local `.plans/` come later.
- `documents.py` normalizes into `Document`s. Bots are filtered by GraphQL `author.__typename`,
  not by a login list. Deleted accounts become `ghost`.
- `chunking.py` splits each document's markdown at headings (markdown-it-py, top-level blocks only, so
  a code fence or table is never cut), packs a long section into pieces of up to 400 estimated tokens, and
  prefixes each chunk with a header like `owner/repo · PR #17 · <title> · <section>`.
- `embedding/` (an `Embedder` protocol; local `voyage-4-nano` by default, Voyage's API
  optional), `store.py` (one SQLite file with sqlite-vec + FTS5; one DB file per pipeline config),
  `retrieval.py`, `evals/`, `mcp_server.py`, `cli.py`.
- `evals/questions.py` loads the eval question set. The real set lives in the config dir
  (`repolore questions` prints where) and is never committed, because it names private repos;
  `evals/questions.example.yaml` shows the format.
- `cli.py` is Typer wiring only. Logic lives in the stage modules.

Only the MCP server is async. The rest of the pipeline is synchronous on purpose.

## Comments

No comments, strictly, on hand-written code. Names and small functions carry the meaning; if a
comment feels needed, that's usually a sign the code should be restructured or renamed instead.

The one exception is a **"why" comment**, only when absolutely necessary: a non-obvious invariant, a
workaround for a specific upstream bug or quirk, or a constraint that isn't visible at the call
site. Never a "what" comment restating what the code already says.

## Testing conventions

Tests live in `tests/`, one file per module (`test_<module>.py`), using plain pytest functions and
fixtures. **Tests never touch the network by default.** HTTP is faked with `httpx.MockTransport`
fed from saved JSON samples, and embeddings come from a deterministic fake `Embedder`. Fixtures in
`tests/fixtures/` come from **public repos only**, because this repo is public; never copy data from a
private repo into it. Anything that
calls a real API is marked `@pytest.mark.live`, which is deselected unless run with `-m live`.
Retrieval quality is measured by the eval harness (`repolore eval`), not by unit tests.
