from pathlib import Path

import pytest

from repolore.paths import CONFIG_FILENAME, QUESTIONS_FILENAME, resolve_paths


def test_default_paths_are_per_user_app_dirs() -> None:
    paths = resolve_paths()

    assert paths.config_file.name == CONFIG_FILENAME
    assert paths.config_file.parent.name == "repolore"
    assert paths.data_dir.name == "repolore"
    assert paths.data_dir.is_absolute()


def test_data_dir_override_is_resolved_to_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    paths = resolve_paths(Path("relative-data"))

    assert paths.data_dir == tmp_path.resolve() / "relative-data"


def test_data_dir_override_leaves_config_location_alone(tmp_path: Path) -> None:
    assert resolve_paths(tmp_path).config_file == resolve_paths().config_file


def test_questions_file_lives_in_the_config_dir() -> None:
    paths = resolve_paths()

    assert paths.questions_file == paths.config_file.parent / "evals" / QUESTIONS_FILENAME
