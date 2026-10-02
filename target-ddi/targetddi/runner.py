"""Executes the pipeline stages in order inside one shared namespace.

The stages were developed and run as consecutive cells of a single notebook, and later stages
use objects created by earlier ones (the configuration, the dataset, the feature banks, the
model classes). Running them in one namespace keeps the code identical to the version that
produced the paper's numbers.
"""
import os
import sys
import time
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent / "stages"

SETUP_STAGES = [
    "s00_workdir.py", "s01_config.py", "s02_environment.py", "s03_targets.py",
    "s04_splits_metrics.py", "s05_molecular.py", "s06_model.py", "s07_selftest.py",
]
PIPELINE_STAGES = [
    "s08_run_mode.py", "s09_main_grid.py", "s10_emergnn_ddiben.py", "s11_results.py",
    "s12_attribution.py", "s13_bundle.py", "s14_emergnn_ddiben_show.py", "s15_gate_ablation.py",
    "s16_headtail.py", "s17_case_cost.py", "s18_emergnn_folds.py", "s19_twosides.py",
    "s20_twosides_rescore.py", "s21_comparison_table.py", "s22_csmddi.py", "s23_collect.py",
]

# Section name -> (description, part of "all")
SECTIONS = {
    "main-grid":        ("S1-S3 grid: Mol-Classic, Mol-Full, TARGET-DDI, TARGET-DDI-G x 3 seeds", True),
    "gate":             ("fusion ablation, Table 13", True),
    "headtail":         ("head/tail interaction types, Table 11 (needs main-grid)", True),
    "case-cost":        ("case study, Table 12, and training cost (needs main-grid)", True),
    "emergnn-folds":    ("EmerGNN released partitions from Zenodo plus warm start, Table 7", True),
    "twosides":         ("TWOSIDES emerging-drug splits, Table 14", True),
    "twosides-rescore": ("TWOSIDES re-scoring with paired negatives (needs twosides)", True),
    "csmddi":           ("CSMDDI retrained on S1-S3, Table 4 CSMDDI rows", True),
    "seed0-gaps":       ("retrain only missing seed-0 grid runs (not needed after main-grid)", False),
    "emergnn-ddiben":   ("single EmerGNN split shipped with DDI-Ben (superseded by emergnn-folds)", False),
}


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def run(workdir, sections, fast=False, budget_hours=1e6, resume_from=None, selftest_only=False):
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    flags = {name: (name in sections) for name in SECTIONS}
    runtime = {"workdir": workdir, "fast": fast, "budget_hours": budget_hours,
               "resume_from": list(resume_from or []), "flags": flags}
    stages = SETUP_STAGES if selftest_only else SETUP_STAGES + PIPELINE_STAGES

    log_path = os.path.join(workdir, f"run_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    log = open(log_path, "a", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(old_out, log), _Tee(old_err, log)
    ns = {"__name__": "__targetddi__", "RUNTIME": runtime}
    try:
        print(f"TARGET-DDI run | workdir={workdir} | sections={sorted(sections) or 'none'} "
              f"| fast={fast} | budget={budget_hours} h | log={log_path}")
        for name in stages:
            path = STAGE_DIR / name
            t0 = time.time()
            print(f"\n######## {name} ########", flush=True)
            code = compile(path.read_text(), str(path), "exec")
            exec(code, ns)
            print(f"######## {name} done in {time.time() - t0:.0f} s", flush=True)
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        log.close()
    return ns
