import importlib
import json
from pathlib import Path
from mmphate_repro.utils.paths import ensure_dirs, REPO_ROOT

def _import_callable(spec: str):
    mod_name, func_name = spec.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, func_name)

def reproduce_main(mode: str, manifest_path: str):
    ensure_dirs()
    manifest = json.loads(Path(manifest_path).read_text())
    figs = manifest["figures"]

    for f in figs:
        if mode not in f["mode"]:
            continue
        fn = _import_callable(f["pipeline"])
        fn(mode=mode, spec=f, repo_root=REPO_ROOT)