# repolore

RAG over your own engineering history on GitHub: PRs, issues, commits and docs, searchable from a CLI and from
Claude Code over MCP.

## Development

`main` only changes through PRs. After cloning, enable the hooks that block direct commits and pushes to it:

```sh
git config core.hooksPath .githooks
```
