"""Working directory, optional resume from earlier output directories."""

import os, shutil
from pathlib import Path

WORKDIR = os.path.abspath(RUNTIME["workdir"])
os.makedirs(WORKDIR, exist_ok=True)

# Optional resume: copy finished work from earlier output directories (never overwrites).
MARKERS = ("ck_*.pt", "res_*.json", "pred_*.npz", "all_results*.csv", "hist_*.csv")
for _src in RUNTIME.get("resume_from") or []:
    _src = Path(_src)
    if not _src.is_dir():
        raise FileNotFoundError(f"--resume-from {_src}: not a directory")
    _root = _src / "ddi_work" if (_src / "ddi_work").is_dir() else _src
    _n = 0
    for f in _root.rglob("*"):
        if f.is_file():
            dest = Path(WORKDIR) / f.relative_to(_root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(f, dest); _n += 1
    print(f"[resume] {_src}: copied {_n} file(s) into {WORKDIR}")

_wd = Path(WORKDIR)
print(f"WORKDIR={WORKDIR}")
print(f"  {len(list(_wd.rglob('ck_*.pt')))} checkpoints, {len(list(_wd.rglob('res_*.json')))} result files, "
      f"{len(list(_wd.rglob('pred_*.npz')))} saved predictions (finished units are skipped)")
