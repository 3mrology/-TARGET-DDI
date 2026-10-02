"""Re-scores TWOSIDES runs with the paired-negative protocol of the reference implementation."""
# ===== Section 15b. Re-score the TWOSIDES runs with the correct paired-negative protocol =====
# The first TWOSIDES pass compared each type's positives against the FIRST k negative rows.
# EmerGNN's own code compares them against the negatives at the SAME row positions, because
# the release pairs every positive with one sampled negative. This cell reloads the finished
# checkpoints and recomputes the metrics; no model is retrained.
if not RUN_TWOSIDES_RESCORE:
    print('RUN_TWOSIDES_RESCORE=False -> skipped')
else:
    import os, re, json, glob
    from pathlib import Path
    import numpy as np, pandas as pd, torch
    from sklearn.metrics import roc_auc_score, average_precision_score

    TSR = os.path.join(CFG.workdir, "twosides")
    cks = sorted(glob.glob(os.path.join(TSR, "ck_ts_*.pt")))
    if not cks:
        print("No TWOSIDES checkpoints found; run Section 15 first.")
    else:
        print(f"Re-scoring {len(cks)} finished TWOSIDES run(s) with the paired-negative protocol")
        rows = []
        for ck in cks:
            tag = os.path.basename(ck)[3:-3]               # ts_<setting>_<variant>_s<seed>
            m = re.fullmatch(r"ts_(S\d)_(.+)_s(\d)", tag)
            if not m:
                continue
            setting, vname, seed = m.group(1), m.group(2), int(m.group(3))
            d, n_lab = _ts_load(setting)
            _, mode, _ = VARIANT_LOOKUP[vname]
            tr_drugs = set(d["train"]["u"]) | set(d["train"]["v"])
            tmask = torch.zeros(TS_N, dtype=torch.bool)
            for x in tr_drugs:
                _i = int(x)
                if 0 <= _i < TS_N:
                    tmask[_i] = True
            T = {"mol": TS_MOL_T, "tgt": TS_TGT_T, "tcount": TS_TCOUNT, "tgt_bin": TS_BIN_T,
                 "train_mask": tmask.to(DEVICE)}
            model = TargetDDI(TS_MOL_T.shape[1], TS_TGT_T.shape[1], n_lab, CFG, mode=mode).to(DEVICE)
            st = torch.load(ck, map_location=DEVICE, weights_only=False)
            sd = (st.get("best_state") or {}).get("model") or st.get("model")
            model.load_state_dict({k: v.to(DEVICE) for k, v in sd.items()})
            model.eval()

            @torch.no_grad()
            def score_fn(pair_list):
                z = model.nodes(T); out = []
                for i0 in range(0, len(pair_list), CFG.eval_batch):
                    part = pair_list[i0:i0 + CFG.eval_batch]
                    pu = torch.tensor([p[0] for p in part], device=DEVICE)
                    pv = torch.tensor([p[1] for p in part], device=DEVICE)
                    zi, zj = z[pu], z[pv]
                    out.append(torch.sigmoid(model.head(
                        torch.cat([zi + zj, (zi - zj).abs(), zi * zj], -1))).float().cpu().numpy())
                return np.nan_to_num(np.concatenate(out), nan=0.0, posinf=1.0, neginf=0.0)

            res = _ts_eval(d["test_pos"], d["test_neg"], score_fn, n_lab)
            res.update(dict(dataset="TWOSIDES", setting=setting, variant=vname, seed=seed,
                            protocol="paired-negative (EmerGNN)"))
            json.dump(res, open(os.path.join(TSR, f"res_fixed_{tag}.json"), "w"), indent=2)
            rows.append(res)
            print(f"  [{tag}] ROC-AUC={res['roc_auc']*100:.1f} PR-AUC={res['pr_auc']*100:.1f} "
                  f"AP@50={res['ap50']*100:.1f} ({res['n_types_scored']} types)")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if rows:
            df = pd.DataFrame(rows)
            lab = {"full_target": "TARGET-DDI", "shared_target_graph": "TARGET-DDI-G",
                   "ablation_no_target": "Mol-Full"}
            df["model"] = df["variant"].map(lab).fillna(df["variant"])
            our = {"S1": "S2 (one new drug)", "S2": "S3 (two new drugs)"}
            out = []
            for (setting, model), g in df.groupby(["setting", "model"]):
                row = {"Setting": our.get(setting, setting), "Model": model, "Runs": len(g)}
                for met, name in [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("ap50", "AP@50")]:
                    v = g[met] * 100
                    row[name] = f"{v.mean():.1f} ± {(v.std(ddof=1) if len(v) > 1 else 0.0):.1f}"
                out.append(row)
            summary = pd.DataFrame(out).sort_values(["Setting", "Model"])
            summary.to_csv(os.path.join(TSR, "twosides_summary_fixed.csv"), index=False)
            df.to_csv(os.path.join(TSR, "twosides_per_run_fixed.csv"), index=False)
            print("\n=== TWOSIDES, corrected protocol (%, mean ± SD over seeds) ===")
            print(summary.to_string(index=False))
            print("\nPublished reference points on TWOSIDES (DDI-Ben, Table 3/4; these use a")
            print("biomedical knowledge graph, which our model does not):")
            print("   one new drug : SumGNN ROC-AUC 84.9, PR-AUC 87.0 | EmerGNN ROC-AUC ~87, PR-AUC ~86")
            print("   two new drugs: SumGNN ROC-AUC 60.6, PR-AUC 58.2")
