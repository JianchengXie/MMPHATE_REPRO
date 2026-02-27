from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

def data_dir() -> Path:
    return REPO_ROOT / "data"

def cache_dir() -> Path:
    return REPO_ROOT / "results_cache"

def figures_dir() -> Path:
    return REPO_ROOT / "figures"

def ensure_dirs():
    for p in [data_dir(), cache_dir(), figures_dir()]:
        p.mkdir(parents=True, exist_ok=True)