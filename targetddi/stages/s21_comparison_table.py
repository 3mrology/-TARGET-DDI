"""Split-matched comparison with the published EmerGNN table (no training)."""
# ===== Section 19. Final comparison table against published baselines =====
# Builds the table the paper needs: our results on EmerGNN's own folds beside the published
# numbers of every method in Table 1 of the EmerGNN preprint (arXiv:2311.09261), which were
# produced on those same folds with the same metrics. Nothing is trained here.
import os, glob, json
import pandas as pd

TAB_OUT = os.path.join(CFG.workdir, "paper_tables"); os.makedirs(TAB_OUT, exist_ok=True)

# Published values: method -> {setting: (F1, Acc, Kappa) as mean, sd}. "S1" = one new drug,
# "S2" = two new drugs, in EmerGNN's naming (our S2 and S3). A star marks methods that use
# the Hetionet biomedical network.
PUBLISHED = {
    "MLP":        {"kind": "Drug features", "S1": (21.1, 0.8, 46.6, 2.1, 33.4, 2.5)},
    "Similarity": {"kind": "Drug features", "S1": (43.0, 5.0, 51.3, 3.5, 44.8, 3.8)},
    "CSMDDI":     {"kind": "Drug features", "S1": (45.5, 1.8, 62.6, 2.8, 55.0, 3.2),
                                            "S2": (19.8, 3.1, 37.3, 4.8, 22.0, 4.9)},
    "STNN-DDI":   {"kind": "Drug features", "S1": (39.7, 1.8, 56.7, 2.6, 46.5, 3.4)},
    "HIN-DDI*":   {"kind": "Graph features","S1": (37.3, 2.9, 58.9, 1.4, 47.6, 1.8),
                                            "S2": (8.8, 1.0, 27.6, 2.4, 13.8, 2.4)},
    "MSTE":       {"kind": "Embedding",     "S1": (7.0, 0.7, 51.4, 1.8, 37.4, 2.2)},
    "KG-DDI*":    {"kind": "Embedding",     "S1": (26.1, 0.9, 46.7, 1.9, 35.2, 2.5),
                                            "S2": (1.1, 0.1, 32.2, 3.6, 0.0, 0.0)},
    "CompGCN*":   {"kind": "GNN",           "S1": (26.8, 2.2, 48.7, 3.0, 37.6, 2.8)},
    "Decagon*":   {"kind": "GNN",           "S1": (24.3, 4.5, 47.4, 4.9, 35.8, 5.9)},
    "KGNN*":      {"kind": "GNN",           "S1": (23.1, 3.4, 51.4, 1.9, 40.3, 2.7)},
    "SumGNN*":    {"kind": "GNN",           "S1": (35.0, 4.3, 48.8, 8.2, 41.1, 4.7)},
    "DeepLGF*":   {"kind": "GNN",           "S1": (39.7, 2.3, 60.7, 2.4, 51.0, 2.6),
                                            "S2": (4.8, 1.9, 31.9, 3.7, 8.2, 2.3)},
    "EmerGNN*":   {"kind": "GNN",           "S1": (62.0, 2.0, 68.6, 3.7, 62.4, 4.3),
                                            "S2": (25.0, 2.8, 46.3, 3.6, 31.9, 3.8)},
}
OURS = {"ablation_no_target": "Mol-Full (no targets)", "shared_target_graph": "TARGET-DDI-G",
        "full_target": "TARGET-DDI (ours)"}

res = [json.load(open(q)) for q in sorted(glob.glob(os.path.join(CFG.workdir, "emergnn_zenodo", "res_zen_*.json")))]
if not res:
    print("Section 14 has produced no results yet, so the comparison table cannot be built.")
else:
    df = pd.DataFrame(res)
    df["model"] = df["variant"].map(OURS).fillna(df["variant"])
    have = sorted(df["setting"].unique())
    print(f"Our runs cover settings {have} "
          f"({', '.join(f'{s}: {len(df[df.setting==s])} run(s)' for s in have)})")

    def cell(mean, sd):
        return f"{mean:.1f} ± {sd:.1f}"

    rows = []
    for name, rec in PUBLISHED.items():
        row = {"Type": rec["kind"], "Method": name, "Source": "published"}
        for setting, label in [("S1", "S2"), ("S2", "S3")]:      # their name -> our name
            v = rec.get(setting)
            for k, met in enumerate(["F1", "Acc", "Kappa"]):
                row[f"{label} {met}"] = cell(v[2*k], v[2*k+1]) if v else "--"
        rows.append(row)
    for variant, label in OURS.items():
        g_all = df[df["variant"] == variant]
        if not len(g_all):
            continue
        row = {"Type": "Drug + protein features", "Method": label, "Source": "this work"}
        for setting, ours in [("S1", "S2"), ("S2", "S3")]:
            g = g_all[g_all["setting"] == setting]
            for met, col in [("F1", "all_f1"), ("Acc", "all_accuracy"), ("Kappa", "all_kappa")]:
                if len(g):
                    v = g[col] * 100
                    row[f"{ours} {met}"] = cell(v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0)
                else:
                    row[f"{ours} {met}"] = "--"
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(TAB_OUT, "emergnn_comparison.csv"), index=False)
    print("\n=== TARGET-DDI vs published methods, EmerGNN's own folds (%, mean ± SD) ===")
    print("S2 = one new drug, S3 = two new drugs (EmerGNN calls these S1 and S2)")
    print(table.to_string(index=False))

    # warm-start control, if S0 was run
    warm = df[df["setting"] == "S0"]
    if len(warm):
        print("\n=== Warm start (EmerGNN S0 = our S1), our models only ===")
        w = warm.groupby("model")[["all_f1", "all_accuracy", "all_kappa"]].agg(["mean", "std"]) * 100
        print(w.round(1).to_string())
        w.round(2).to_csv(os.path.join(TAB_OUT, "emergnn_warmstart.csv"))

    # LaTeX for the manuscript
    tex = table.to_latex(index=False, escape=True, column_format="ll" + "c" * (len(table.columns) - 2))
    open(os.path.join(TAB_OUT, "emergnn_comparison.tex"), "w").write(tex)
    print(f"\nwrote {TAB_OUT}/emergnn_comparison.csv and .tex")
    print("Published values: Table 1 of arXiv:2311.09261, same folds and metrics as our runs.")
    print("* marks methods that additionally use the Hetionet biomedical network.")
