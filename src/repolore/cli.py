from pathlib import Path
from typing import Annotated

import typer

from repolore import __version__
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
