"""Prints the DDI-Ben EmerGNN results of s10, if any."""
# ===== SHOW EmerGNN BENCHMARK RESULTS =====
import os, json, glob, shutil
from pathlib import Path
import pandas as pd
from IPython.display import display

_WD = CFG.workdir if "CFG" in dir() else os.path.expanduser("~/ddi_work")
_places = [Path(_WD) / "emergnn_bench", Path("/kaggle/working/ddi_work/emergnn_bench"), Path("/kaggle/input")]

# collect every finished (fold, model) result from any of those places, no duplicates
_rows = {}
for place in _places:
    if place.exists():
        for f in place.rglob("res_emergnn_*.json"):
            _rows.setdefault(f.name, json.load(open(f)))

if not _rows:
    print("No EmerGNN benchmark results exist yet, so there is nothing to show.")
    _has_data = any(Path("/kaggle/input").rglob("node2id.json")) if Path("/kaggle/input").exists() else False
    if not _has_data:
        print("Reason: the EmerGNN data-DB dataset is not attached, so Section 8b skipped the benchmark.")
        print("Fix: Kaggle -> Datasets -> New Dataset -> upload data-DB.zip; then in this notebook")
        print("     '+ Add Input' -> Datasets -> Your Datasets -> attach it; rerun Section 8b, then this cell.")
    else:
        print("The data is attached but Section 8b has not produced results. Rerun Section 8b and check its log.")
else:
    df = pd.DataFrame(list(_rows.values()))
    label = {"full_target": "TARGET-DDI", "shared_target_graph": "TARGET-DDI-G", "ablation_no_target": "Mol-Full"}
    df["model"] = df["variant"].map(label).fillna(df["variant"])
    per_fold = df[["setting", "fold", "model", "all_f1", "all_accuracy", "all_kappa",
                   "n_test_covered", "n_test_triples"]].copy()
    for m in ["all_f1", "all_accuracy", "all_kappa"]:
        per_fold[m] = (per_fold[m] * 100).round(2)
    per_fold = per_fold.rename(columns={"all_f1": "F1", "all_accuracy": "Accuracy", "all_kappa": "Kappa"})
    per_fold = per_fold.sort_values(["setting", "model", "fold"])

    summary = []
    for (setting, model), g in per_fold.groupby(["setting", "model"]):
        row = {"Setting": setting, "Model": model, "Folds": len(g),
               "Coverage": f"{(g['n_test_covered'].sum() / g['n_test_triples'].sum()) * 100:.1f}%"}
        for m in ["F1", "Accuracy", "Kappa"]:
            sd = g[m].std(ddof=1) if len(g) > 1 else 0.0
            row[m] = f"{g[m].mean():.1f} ± {sd:.1f}"
        summary.append(row)
    summary = pd.DataFrame(summary)

    print("EmerGNN DrugBank benchmark (percent, mean ± SD over folds)")
    print("S1 = one emerging drug per test pair, S2 = two emerging drugs")
    display(summary)
    print("\nPer fold:")
    display(per_fold)

    out = Path("/kaggle/working/emergnn_results") if os.path.isdir("/kaggle/working") else Path("./emergnn_results")
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "emergnn_results_summary.csv", index=False)
    per_fold.to_csv(out / "emergnn_results_per_fold.csv", index=False)
    shutil.make_archive(str(out), "zip", str(out))
    print(f"\nSaved {out}/ and {out}.zip (download from the Output tab)")
