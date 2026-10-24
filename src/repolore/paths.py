from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir

APP_NAME = "repolore"
CONFIG_FILENAME = "repolore.toml"
QUESTIONS_FILENAME = "questions.yaml"


@dataclass(frozen=True)
class AppPaths:
    config_file: Path
    questions_file: Path
    data_dir: Path


def resolve_paths(data_dir: Path | None = None) -> AppPaths:
    # Roaming config follows the user across machines; the data dir holds rebuildable caches and DBs, so it stays local.
    config_dir = Path(user_config_dir(APP_NAME, appauthor=False, roaming=True))
    default_data_dir = Path(user_data_dir(APP_NAME, appauthor=False))
    return AppPaths(
        config_file=config_dir / CONFIG_FILENAME,
        questions_file=config_dir / "evals" / QUESTIONS_FILENAME,
        data_dir=(data_dir or default_data_dir).resolve(),
    )
