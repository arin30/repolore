# repolore

RAG over your own engineering history on GitHub: PRs, issues, commits and docs, searchable from a CLI and from
Claude Code over MCP.

## Usage

```sh
uv sync
uv run repolore repos      # list the repos it finds for your account, and why each is in
uv run repolore ingest     # fetch them; later runs fetch only what changed
uv run repolore build      # embed and index them locally; the first build takes a while on a CPU
uv run repolore search "why did we drop the retry middleware?"
```

Repos are discovered from your GitHub account: the ones you own or can read as a collaborator or organization
member, plus your own PRs and issues in other people's repos. `repolore.toml` (`uv run repolore version` prints its
path) is optional and narrows that set:

```toml
include = ["owner/*"]           # only these (globs over owner/name); all when unset
exclude = ["owner/old-*"]       # never these
local_roots = ["~/projects"]    # checkouts in here add their local .plans/ documents
```

The GitHub token comes from
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
