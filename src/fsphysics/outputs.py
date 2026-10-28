"""Stage folders hold the outputs of their stage's last run, and nothing else.

Each stage's run() checks its settings and inputs, then calls start(): the
folder is emptied (the stability stage keeps its saved refits, so it can
resume) and config_used.yaml records the settings. A bad setting therefore
fails before anything is removed. The pipeline empties plots/ before every
plot, so the figures there are exactly those of the last plot.
"""

import shutil
from pathlib import Path

from . import config as _config

BAD_FILE_CHARS = '\\/:*?"<>|'      # not allowed in Windows file names


def file_stem(name):
    """A descriptor name as a file name (% -> pct; characters Windows does
    not allow -> _)."""
    stem = name.replace("%", "pct")
    for ch in BAD_FILE_CHARS:
        stem = stem.replace(ch, "_")
    return stem


def clear(folder, keep=()):
    """Delete everything in folder except the names in keep."""
    folder = Path(folder)
    if not folder.is_dir():
        return
    for path in folder.iterdir():
        if path.name in keep:
            continue
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        except PermissionError as exc:
            raise PermissionError(
                f"Could not remove {path}: it may be open in another program "
                f"(Excel, a PDF viewer) or read-only. Close it or make it "
                f"writable, then run again.") from exc


def start(out_dir, cfg, keep=()):
    """Empty a stage folder (except keep) and record the run's settings."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clear(out_dir, keep)
    _config.save_config(cfg, out_dir / "config_used.yaml")
