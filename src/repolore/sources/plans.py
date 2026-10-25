import json
import logging
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

PLANS_DIR = ".plans"
PLANS_FILE = "plans.json"
DOT_SEGMENTS = frozenset({".", ".."})
GITHUB_REMOTE = re.compile(r"github\.com[:/]([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")

log = logging.getLogger(__name__)


class PlanFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    url: str
    text: str


def find_plan_checkouts(roots: Iterable[Path]) -> dict[str, Path]:
    """Git checkouts directly under the roots that hold a .plans/ folder, keyed by their GitHub repo."""
    checkouts: dict[str, Path] = {}
    for root in (root.expanduser() for root in roots):
        if not root.is_dir():
            log.warning("local root %s doesn't exist", root)
            continue
        for directory in sorted(root.iterdir()):
            if (directory / PLANS_DIR).is_dir() and (repo := github_repo(directory)):
                checkouts[repo] = directory
    return checkouts


def github_repo(checkout: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), "remote", "get-url", "origin"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        log.info("%s has no origin remote, so its plans are skipped", checkout)
        return None
    match = GITHUB_REMOTE.search(result.stdout.strip())
    if match and not {match[1], match[2]} & DOT_SEGMENTS:
        return f"{match[1]}/{match[2]}"
    log.info("%s isn't a GitHub checkout, so its plans are skipped", checkout)
    return None


def read_plans(checkout: Path) -> list[PlanFile]:
    return [
        PlanFile(
            path=path.relative_to(checkout).as_posix(),
            url=path.resolve().as_uri(),
            text=path.read_text(encoding="utf-8", errors="replace"),
        )
        for path in sorted((checkout / PLANS_DIR).rglob("*.md"))
        if path.is_file()
    ]


def save_plans(directory: Path, plans: list[PlanFile]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    content = json.dumps([plan.model_dump() for plan in plans], ensure_ascii=False, indent=1)
    (directory / PLANS_FILE).write_text(content, encoding="utf-8")


def load_plans(directory: Path) -> list[PlanFile]:
    path = directory / PLANS_FILE
    if not path.exists():
        return []
    return [PlanFile.model_validate(plan) for plan in json.loads(path.read_text(encoding="utf-8"))]
