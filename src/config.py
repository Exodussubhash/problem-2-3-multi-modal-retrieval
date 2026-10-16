"""Shared project configuration loaded from the root config.yaml."""

from pathlib import Path
import os

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path=CONFIG_PATH):
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_path(value):
    """Expand ~ and resolve relative paths from the project root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


CONFIG = load_config()
CONFIG.setdefault("data", {})
CONFIG["data"]["video_dir"] = os.environ.get(
    "BDD_VIDEO_DIR", CONFIG["data"].get("video_dir", "./sample")
)
CONFIG["data"]["labels_dir"] = os.environ.get(
    "BDD_LABELS_DIR", CONFIG["data"].get("labels_dir", "./labels")
)
