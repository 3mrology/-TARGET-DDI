"""Collects every summary table into paper_results/ and zips it."""
# ===== Section 18. Flush every result into the notebook output =====
# Everything above already writes inside /kaggle/working/ddi_work, so results survive a run
# that is cut short. This cell additionally gathers the CSV/JSON summaries into one flat
# folder, writes an index, and zips it, so a scheduled run leaves a single file to download.
import os, shutil, glob, json, time
import pandas as pd

ON_KAGGLE = os.path.isdir("/kaggle/working")
OUT = "/kaggle/working/paper_results" if ON_KAGGLE else os.path.join(CFG.workdir, "paper_results")
os.makedirs(OUT, exist_ok=True)

SUBS = ["headtail", "gate_ablation", "emergnn_zenodo", "twosides", "case_study", "cost",
        "emergnn_bench", "paper_tables", "csmddi"]
copied, index = 0, []
for sub in SUBS:
    d = os.path.join(CFG.workdir, sub)
    if not os.path.isdir(d):
        continue
    dest = os.path.join(OUT, sub); os.makedirs(dest, exist_ok=True)
    for f in sorted(os.listdir(d)):
        if f.endswith((".csv", ".json")):
            shutil.copy2(os.path.join(d, f), dest); copied += 1
            index.append({"section": sub, "file": f,
                          "bytes": os.path.getsize(os.path.join(d, f))})
# main grid results and the summary table, if the grid has been run
for f in sorted(glob.glob(f"{CFG.workdir}/res_*.json") + glob.glob(f"{CFG.workdir}/all_results*.csv")):
    shutil.copy2(f, OUT); copied += 1
    index.append({"section": "main_grid", "file": os.path.basename(f), "bytes": os.path.getsize(f)})

if index:
    pd.DataFrame(index).to_csv(os.path.join(OUT, "INDEX.csv"), index=False)
arc = shutil.make_archive(OUT, "zip", OUT)
print(f"collected {copied} files -> {arc} ({os.path.getsize(arc)/1e6:.1f} MB)")

print("\nSummary tables produced so far:")
found = False
for sub in SUBS:
    for f in sorted(glob.glob(os.path.join(OUT, sub, "*summary*.csv")) +
                    glob.glob(os.path.join(OUT, sub, "*macro_f1*.csv")) +
                    glob.glob(os.path.join(OUT, sub, "*table*.csv")) +
                    glob.glob(os.path.join(OUT, sub, "*comparison*.csv"))):
        found = True
        print(f"\n--- {sub}/{os.path.basename(f)} ---")
        try:
            print(pd.read_csv(f).to_string(index=False))
        except Exception as e:
            print("   unreadable:", e)
if not found:
    print("  (none yet -- run the sections above)")

# ---- what is finished and what the next session still has to do ----
def _count(sub, pattern, total):
    d = os.path.join(CFG.workdir, sub)
    n = len(glob.glob(os.path.join(d, pattern))) if os.path.isdir(d) else 0
    return n, total

print("\nProgress by section:")
_plan = [
    ("13 gate ablation",  _count("gate_ablation", "res_gate_*.json", 36)),
    ("14 EmerGNN folds",  _count("emergnn_zenodo", "res_zen_*.json", 33)),
    ("15 TWOSIDES",       _count("twosides", "res_ts_*.json", 18)),
]
_plan = [(n, (d, t)) for n, (d, t) in _plan if not (n.startswith("13") and d == 0)]   # gate is off
_remaining = 0
for name, (done, total) in _plan:
    left = max(total - done, 0); _remaining += left
    mark = "complete" if left == 0 else f"{left} run(s) left"
    print(f"  {name:20s} {done:3d}/{total:<3d}  {mark}")
for name, sub, files in [("12 head/tail", "headtail", "headtail_macro_f1.csv"),
                         ("16 case study", "case_study", "corrected_pairs.csv"),
                         ("17 cost table", "cost", "cost_table.csv")]:
    ok = os.path.exists(os.path.join(CFG.workdir, sub, files))
    print(f"  {name:20s} {'complete' if ok else 'not produced yet (needs the paper-grid predictions attached)'}")
if _remaining:
    print(f"\n{_remaining} training run(s) remain. Save Version, then attach this run's output")
    print("as an input next session and run again: finished units are skipped automatically.")
else:
    print("\nAll training sections are complete.")

if ON_KAGGLE:
    print("\nDownload from the Output tab: paper_results.zip")
    print("Everything is also under ddi_work/ in the same output, including checkpoints,")
    print("so attaching this run's output as an input next time resumes exactly where it stopped.")
# ---- final completeness check ----
print("\n" + "=" * 62)
print("FINAL CHECK — what this run produced")
print("=" * 62)
_checks = [
    ("head/tail table (12)",      os.path.join(CFG.workdir, "headtail", "headtail_macro_f1.csv")),
    ("gate ablation (13)",        os.path.join(CFG.workdir, "gate_ablation", "gate_ablation_summary.csv")),
    ("EmerGNN folds (14)",        os.path.join(CFG.workdir, "emergnn_zenodo", "emergnn_zenodo_summary.csv")),
    ("TWOSIDES (15)",             os.path.join(CFG.workdir, "twosides", "twosides_summary.csv")),
    ("case study (16)",           os.path.join(CFG.workdir, "case_study", "corrected_pairs.csv")),
    ("cost table (17)",           os.path.join(CFG.workdir, "cost", "cost_table.csv")),
    ("baseline comparison (19)",  os.path.join(CFG.workdir, "paper_tables", "emergnn_comparison.csv")),
]
_missing = []
for _name, _path in _checks:
    _ok = os.path.exists(_path)
    print(f"  {'OK  ' if _ok else 'MISS'}  {_name}")
    if not _ok:
        _missing.append(_name)
if _missing:
    print("\nStill missing: " + ", ".join(_missing))
    print("Send paper_results.zip anyway; the sections that did finish are complete and usable.")
else:
    print("\nEverything is complete. Download paper_results.zip from the Output tab and send it.")
