from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError

from repolore import __version__
from repolore.evals.questions import QuestionKind, load_questions
from repolore.paths import AppPaths, resolve_paths

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
