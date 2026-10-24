import logging
import tomllib
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError

from repolore import __version__
from repolore.config import Config, load_config
from repolore.documents import Document, DocumentKind
from repolore.evals.questions import QuestionKind, load_questions
from repolore.ingest import ingest as ingest_repos
from repolore.paths import AppPaths, resolve_paths
from repolore.sources.github import GitHubClient, GitHubError, resolve_token

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main(
    ctx: typer.Context,
    data_dir: Annotated[Path | None, typer.Option(help="Override the data directory.")] = None,
) -> None:
    """RAG over your own engineering history."""
    ctx.obj = resolve_paths(data_dir)


@app.command()
def version(ctx: typer.Context) -> None:
    """Print the version and the resolved config and data paths."""
    paths: AppPaths = ctx.obj
    typer.echo(f"repolore {__version__}")
    typer.echo(f"config: {paths.config_file}")
    typer.echo(f"data:   {paths.data_dir}")


@app.command()
def questions(
    ctx: typer.Context,
    file: Annotated[
        Path | None, typer.Option(help="Question file to check instead of the one in the config dir.")
    ] = None,
) -> None:
    """Validate the eval question set and print how it is made up."""
    paths: AppPaths = ctx.obj
    path = file or paths.questions_file
    try:
        question_set = load_questions(path)
    except FileNotFoundError:
        typer.echo(f"No question file at {path}. Start from evals/questions.example.yaml.", err=True)
        raise typer.Exit(1) from None
    except (yaml.YAMLError, ValidationError) as error:
        typer.echo(f"{path} is invalid:\n{error}", err=True)
        raise typer.Exit(1) from None

    kinds = Counter(q.kind for q in question_set.questions)
    held_out = sum(q.holdout for q in question_set.questions)
    typer.echo(f"questions: {path}")
    typer.echo(f"total:     {len(question_set.questions)} ({held_out} held out)")
    for kind in QuestionKind:
        typer.echo(f"  {kind:<13} {kinds[kind]}")


@app.command()
def ingest(
    ctx: typer.Context,
    offline: Annotated[bool, typer.Option(help="Rebuild documents from the raw cache without calling GitHub.")] = False,
) -> None:
    """Fetch the configured repos from GitHub into the raw cache and count the documents they yield."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    paths: AppPaths = ctx.obj
    config = _config_or_exit(paths.config_file)
    client = None
    try:
        client = None if offline else GitHubClient.connect(resolve_token())
        documents = ingest_repos(config.repos, paths.data_dir / "raw", client)
    except GitHubError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from None
    except FileNotFoundError as error:
        typer.echo(f"No raw cache at {Path(error.filename).parent}; run `repolore ingest` without --offline.", err=True)
        raise typer.Exit(1) from None
    finally:
        if client:
            client.close()
    _print_counts(config.repos, documents)


def _config_or_exit(path: Path) -> Config:
    try:
        return load_config(path)
    except FileNotFoundError:
        typer.echo(f'No config at {path}. Create it with the repos to ingest: repos = ["owner/name"]', err=True)
    except (tomllib.TOMLDecodeError, ValidationError) as error:
        typer.echo(f"{path} is invalid:\n{error}", err=True)
    raise typer.Exit(1)


def _print_counts(repos: list[str], documents: list[Document]) -> None:
    counts = Counter((document.repo.lower(), document.kind) for document in documents)
    width = max(len(repo) for repo in repos)
    typer.echo(f"{'repo':<{width}}" + "".join(f"{kind:>8}" for kind in DocumentKind))
    for repo in repos:
        typer.echo(f"{repo:<{width}}" + "".join(f"{counts[repo.lower(), kind]:>8}" for kind in DocumentKind))
    totals = Counter(document.kind for document in documents)
    typer.echo(f"{'total':<{width}}" + "".join(f"{totals[kind]:>8}" for kind in DocumentKind))
