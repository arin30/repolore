# repolore

RAG over your own engineering history on GitHub: PRs, issues, commits and docs, searchable from a CLI and from
Claude Code over MCP.

## Usage

```sh
uv sync
uv run repolore ingest     # fetch the repos listed in repolore.toml (see `uv run repolore version` for its path)
uv run repolore build      # embed and index them locally; the first build takes a while on a CPU
uv run repolore search "why did we drop the retry middleware?"
```

`repolore.toml` lists the repos: `repos = ["owner/name", …]`. The GitHub token comes from
`REPOLORE_GITHUB_TOKEN`, or from `gh auth token` when that's unset. Nothing leaves your machine except the GitHub
API calls: embeddings are computed locally.

To search from Claude Code, register the MCP server once:

```sh
claude mcp add --scope user --transport stdio repolore -- uv run --directory /path/to/repolore --no-sync repolore-mcp
```

## Development

`main` only changes through PRs. After cloning, enable the hooks that block direct commits and pushes to it:

```sh
git config core.hooksPath .githooks
```
