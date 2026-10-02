"""Which sections run (set from the command line), session budget, progress reporting."""

EXTRA_VARIANT_NAMES = []   # extension slot for further variants (see README, Extending)

_F = RUNTIME["flags"]
SKIP_MAIN_GRID        = not _F["main-grid"]
FILL_SEED0_GAPS       = _F["seed0-gaps"]
RUN_EMERGNN_BENCHMARK = _F["emergnn-ddiben"]
RUN_HEADTAIL          = _F["headtail"]
RUN_GATE              = _F["gate"]
RUN_ZENODO            = _F["emergnn-folds"]
RUN_TWOSIDES          = _F["twosides"]
RUN_CASE_COST         = _F["case-cost"]
RUN_TWOSIDES_RESCORE  = _F["twosides-rescore"]
RUN_CSMDDI            = _F["csmddi"]

# Every loop checks the clock before starting a new unit and exits cleanly when the budget is
# spent; finished work is on disk and is skipped on the next run.
SESSION_BUDGET_HOURS = float(RUNTIME["budget_hours"])

import time as _time
SESSION_START = _time.time()


def hours_left():
    return SESSION_BUDGET_HOURS - (_time.time() - SESSION_START) / 3600


def budget_ok(need_hours=0.0, label=""):
    """True if there is time to start another unit of roughly need_hours."""
    left = hours_left()
    if left <= max(need_hours, 0.02):
        print(f"  session budget reached ({SESSION_BUDGET_HOURS} h): stopping before {label or 'the next unit'}.", flush=True)
        print("  Everything finished so far is saved; rerun the same command to resume.", flush=True)
        return False
    return True


def progress(done, total, started, label="units"):
    """One line after every finished unit: how many are left and the projected finish time.
    Also appends to STATUS.txt inside the output directory, so a run that is stopped without
    reaching Section 18 still leaves a readable record of how far it got."""
    import os as _os
    left = total - done
    el = _time.time() - started
    per = el / max(done, 1)
    eta = per * left
    filled = int(20 * done / max(total, 1))
    bar = "#" * filled + "." * (20 - filled)
    msg = f"  [{bar}] {done}/{total} {label} done, {left} left"
    if done:
        msg += f" | {per/60:.1f} min each | ~{eta/60:.0f} min ({eta/3600:.1f} h) to go"
    print(msg, flush=True)
    try:
        _wd = globals().get("CFG").workdir if globals().get("CFG") is not None else "."
        with open(_os.path.join(_wd, "STATUS.txt"), "a") as fh:
            fh.write(f"{_time.strftime('%Y-%m-%d %H:%M:%S')}  {label}: {done}/{total} done, {left} left, "
                     f"{per/60:.1f} min each, {hours_left():.1f} h of budget remaining\n")
    except Exception:
        pass
    return left


for _k, _v in [("SKIP_MAIN_GRID", SKIP_MAIN_GRID), ("FILL_SEED0_GAPS", FILL_SEED0_GAPS),
               ("RUN_EMERGNN_BENCHMARK", RUN_EMERGNN_BENCHMARK),
               ("RUN_HEADTAIL", RUN_HEADTAIL), ("RUN_GATE", RUN_GATE), ("RUN_ZENODO", RUN_ZENODO),
               ("RUN_TWOSIDES", RUN_TWOSIDES), ("RUN_TWOSIDES_RESCORE", RUN_TWOSIDES_RESCORE), ("RUN_CASE_COST", RUN_CASE_COST), ("RUN_CSMDDI", RUN_CSMDDI)]:
    print(f"  {_k:24s} {_v}")
