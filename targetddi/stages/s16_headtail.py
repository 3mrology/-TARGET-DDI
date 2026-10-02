"""Head and tail interaction types (Table 11)."""
# Section 12
if not RUN_HEADTAIL:
    print('RUN_HEADTAIL=False -> skipped')
else:
    # ===== Section 12. Head/tail interaction types, recomputed consistently =====
    # Fixes the inconsistency in the submitted table (head and tail both below the overall
    # macro-F1, which cannot happen under one evaluation). Here every group score uses the
    # SAME per-type thresholds as the overall score: the thresholds tuned on validation and
    # stored in the run's checkpoint. Overall macro-F1 is recomputed from the same arrays and
    # printed next to the stored value as a self-check; group scores are means of the SAME
    # per-type F1 values over the head and tail subsets, so they bracket the overall value by
    # construction. Uses saved predictions only -- no retraining.
    import os, re, json, glob
    import numpy as np, pandas as pd
    from sklearn.metrics import f1_score

    HT_OUT = os.path.join(CFG.workdir, "headtail"); os.makedirs(HT_OUT, exist_ok=True)
    HT_VARIANTS = ["ablation_no_target", "full_target", "shared_target_graph"]
    HT_LABEL = {"ablation_no_target": "Mol-Full", "full_target": "TARGET-DDI",
                "shared_target_graph": "TARGET-DDI-G"}
    HT_SEED = 0
    HT_HEAD_FRACTION = 0.5      # head = types whose training positives are in the top half


    def _ht_thresholds(tag):
        """Per-type thresholds actually used for this run's reported metrics."""
        ck = os.path.join(CFG.workdir, f"ck_{tag}.pt")
        if not os.path.exists(ck):
            return None
        try:
            st = torch.load(ck, map_location="cpu", weights_only=False)
            bs = st.get("best_state") or {}
            thr = bs.get("thr")
            return None if thr is None else np.asarray(thr, dtype=np.float32)
        except Exception as e:
            print(f"   [{tag}] checkpoint unreadable: {e}")
            return None


    def _ht_train_counts(split, seed):
        """Training positives per interaction type for this split and seed."""
        sp, _ = make_split(pairs, split, CFG.frac, seed)
        return multihot(sp["train"]["y"], C).sum(0)


    def headtail_table(split, seed=HT_SEED):
        counts = _ht_train_counts(split, seed)
        order = np.argsort(-counts)
        n_head = int(round(HT_HEAD_FRACTION * C))
        head = np.zeros(C, bool); head[order[:n_head]] = True
        tail = ~head
        rows = []
        for vname in HT_VARIANTS:
            tag = f"{CFG.dataset}_{CFG.feature_version}_{split}_{vname}_s{seed}"
            pp = os.path.join(CFG.workdir, f"pred_{tag}.npz")
            rp = os.path.join(CFG.workdir, f"res_{tag}.json")
            if not os.path.exists(pp):
                print(f"   [skip] no predictions for {tag}")
                continue
            thr = _ht_thresholds(tag)
            if thr is None:
                print(f"   [skip] no stored thresholds for {tag}; cannot reproduce the reported metric")
                continue
            d = np.load(pp); y, p = d["y"], d["p"]
            yh = (p >= thr[None, :]).astype(int)
            per_type = f1_score(y, yh, average=None, zero_division=0)      # one F1 per type
            overall = float(per_type.mean())                               # == reported macro-F1
            stored = json.load(open(rp))["macro_f1"] if os.path.exists(rp) else float("nan")
            # a type with no test positives and no predictions contributes F1 = 0 to every mean,
            # exactly as in the overall macro-F1, so head/tail/overall stay on one scale
            rows.append({
                "split": split, "model": HT_LABEL.get(vname, vname),
                "head_f1": round(float(per_type[head].mean()) * 100, 2),
                "tail_f1": round(float(per_type[tail].mean()) * 100, 2),
                "overall_f1": round(overall * 100, 2),
                "stored_macro_f1": round(stored * 100, 2),
                "check_ok": bool(abs(overall - stored) < 1e-6) if stored == stored else None,
                "n_head_types": int(head.sum()), "n_tail_types": int(tail.sum()),
                "head_min_train_pos": int(counts[head].min()), "tail_max_train_pos": int(counts[tail].max()),
            })
        return rows


    ht_rows = []
    for split in ["S1", "S2", "S3"]:
        try:
            ht_rows += headtail_table(split)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"  FAILED {split}: {e}")

    if ht_rows:
        ht = pd.DataFrame(ht_rows)
        ht.to_csv(os.path.join(HT_OUT, "headtail_macro_f1.csv"), index=False)
        print("\nMacro-F1 by interaction-type frequency group (%, seed 0, validation-tuned thresholds)")
        print(ht[["split", "model", "head_f1", "tail_f1", "overall_f1", "stored_macro_f1", "check_ok"]].to_string(index=False))
        bad = ht[(ht["check_ok"] == False)]
        if len(bad):
            print("\nWARNING: recomputed overall macro-F1 differs from the stored value for "
                  f"{len(bad)} run(s); the saved predictions or thresholds are out of sync.")
        off = ht[(ht["overall_f1"] < ht[["head_f1", "tail_f1"]].min(axis=1) - 1e-6) |
                 (ht["overall_f1"] > ht[["head_f1", "tail_f1"]].max(axis=1) + 1e-6)]
        print("consistency: overall lies between head and tail in "
              f"{len(ht) - len(off)}/{len(ht)} rows" + (" -- OK" if len(off) == 0 else " -- CHECK"))
        print(f"grouping: head = top {int(HT_HEAD_FRACTION*100)}% of types by training positives")
    else:
        print("No head/tail rows produced. Section 12 needs pred_*.npz and ck_*.pt for the seed-0 runs;")
        print("attach the checkpoint dataset from the session that produced the paper grid and rerun.")
