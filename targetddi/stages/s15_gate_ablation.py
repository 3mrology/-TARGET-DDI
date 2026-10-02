"""Fusion ablation (Table 13)."""
# Section 13
if not RUN_GATE:
    print('RUN_GATE=False -> skipped')
else:
    # ===== Section 13. Gate ablation =====
    # The availability-conditioned gate is claimed as a component, so it is tested against the
    # alternatives it was designed to beat: a fixed 50/50 blend, a zero-initialised gate, plain
    # concatenation of the two branches, and each branch alone. Everything else -- features,
    # loss, schedule, seeds, splits -- is identical to the main runs. Crash-safe: each unit
    # writes res_*.json and is skipped on rerun.
    import os, json, gc, time
    import numpy as np, pandas as pd, torch, torch.nn as nn

    GATE_OUT = os.path.join(CFG.workdir, "gate_ablation"); os.makedirs(GATE_OUT, exist_ok=True)
    GATE_SPLITS = ["S2", "S3"]
    GATE_SEEDS = list(CFG.headline_seeds)
    GATE_VARIANTS = ["gated", "fixed_blend", "zero_init_gate", "concat", "target_only", "molecular"]
    GATE_LABEL = {"gated": "Availability-conditioned gate (proposed)", "fixed_blend": "Fixed 50/50 blend",
                  "zero_init_gate": "Zero-initialised gate", "concat": "Concatenation (no gate)",
                  "target_only": "Target branch only", "molecular": "Molecular branch only"}


    class GateVariantDDI(TargetDDI):
        """TargetDDI with the fusion rule swapped; encoders, head and loss are unchanged."""
        def __init__(self, dmol, dtgt, C_, cfg, fusion="gated"):
            base = "molecular" if fusion == "molecular" else ("target_only" if fusion == "target_only" else "gated")
            super().__init__(dmol, dtgt, C_, cfg, mode=base)
            self.fusion = fusion
            h = cfg.hidden
            if fusion == "concat" and self.use_tgt:
                self.merge = nn.Sequential(nn.Linear(2 * h, h), nn.LayerNorm(h), nn.ReLU())
            if fusion in ("fixed_blend", "concat") and hasattr(self, "gate"):
                del self.gate                 # unused, so it is not counted or optimised
            if fusion == "zero_init_gate" and self.use_tgt:
                last = [m for m in self.gate if isinstance(m, nn.Linear)][-1]
                nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

        def nodes(self, T):
            if self.fusion in ("molecular", "target_only") or not self.use_tgt:
                return super().nodes(T)
            m = self.mol(T["mol"]); t = self.tgt(T["tgt"])
            if self.fusion == "fixed_blend":
                return 0.5 * t + 0.5 * m
            if self.fusion == "concat":
                return self.merge(torch.cat([m, t], -1))
            g = torch.sigmoid(self.gate(torch.cat([m, t, T["tcount"]], -1)))   # gated / zero_init_gate
            return g * t + (1 - g) * m


    def run_gate_unit(split, fusion, seed):
        tag = f"gate_{CFG.dataset}_{CFG.feature_version}_{split}_{fusion}_s{seed}"
        rp = os.path.join(GATE_OUT, f"res_{tag}.json")
        if os.path.exists(rp):
            return json.load(open(rp))
        sp, _ = make_split(pairs, split, CFG.frac, seed)
        if len(sp["test"]) == 0 or len(sp["valid"]) == 0:
            print(f"  {split}: empty split"); return None
        audit(sp, split, id2n, None)
        tr_drugs = set(sp["train"]["u"]) | set(sp["train"]["v"])
        tmask = torch.zeros(len(drug_ids), dtype=torch.bool)
        for d in tr_drugs:
            tmask[id2n[d]] = True
        _tgt = TGT_BANK.get("features", TGT)
        T = {"mol": MOLBANK["full"], "tgt": _tgt, "tcount": TCOUNT, "tgt_bin": TGT_BIN,
             "train_mask": tmask.to(DEVICE)}
        set_seed(seed)
        m = GateVariantDDI(MOLBANK["full"].shape[1], _tgt.shape[1], C, CFG, fusion=fusion).to(DEVICE)
        n_par = sum(p.numel() for p in m.parameters() if p.requires_grad)
        t0 = time.time()
        m, thr, hist = train_one(m, sp, T, id2n, C, ckpt=os.path.join(GATE_OUT, f"ck_{tag}.pt"), log=False)
        train_s = time.time() - t0
        yte = multihot(sp["test"]["y"], C); pt = predict(m, sp["test"], T, id2n, C)
        res = evaluate(yte, pt, thr)
        res.update(dict(split=split, fusion=fusion, seed=seed, n_params=int(n_par),
                        train_seconds=round(train_s, 1), n_epochs=len(hist), n_test=len(sp["test"])))
        tmp = rp + ".tmp"; json.dump(res, open(tmp, "w"), indent=2); os.replace(tmp, rp)
        print(f"  [{tag}] microAP={res['micro_auprc']:.4f} macroF1={res['macro_f1']:.4f} "
              f"({res['n_epochs']} ep, {train_s/60:.1f} min)")
        del m; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return res


    gate_rows = []
    def _gate_missing():
        return [(s, f, sd) for s in GATE_SPLITS for f in GATE_VARIANTS for sd in GATE_SEEDS
                if not os.path.exists(os.path.join(GATE_OUT,
                   f"res_gate_{CFG.dataset}_{CFG.feature_version}_{s}_{f}_s{sd}.json"))]
    _total = len(GATE_SPLITS) * len(GATE_VARIANTS) * len(GATE_SEEDS)
    _todo = _gate_missing()
    print(f"SECTION 13 gate ablation: {_total-len(_todo)}/{_total} already complete, "
          f"{len(_todo)} to run now")
    _t0 = time.time(); _per = 0.0
    for _i, (split, fusion, seed) in enumerate(_todo, 1):
        if not budget_ok(_per, f"gate run {_i}/{len(_todo)} ({split}/{fusion}/s{seed})"):
            break
        try:
            r = run_gate_unit(split, fusion, seed)
            if r:
                gate_rows.append(r)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"  FAILED {split}/{fusion}/s{seed}: {e}")
        progress(_i, len(_todo), _t0, "gate runs")
        _per = (time.time() - _t0) / _i / 3600      # hours per unit, for the next budget check
    print(f"SECTION 13 done: {_total-len(_gate_missing())}/{_total} complete overall")

    _all = [json.load(open(p)) for p in sorted(__import__("glob").glob(os.path.join(GATE_OUT, "res_gate_*.json")))]
    if _all:
        df = pd.DataFrame(_all)
        rows = []
        for (split, fusion), g in df.groupby(["split", "fusion"]):
            row = {"Split": split, "Fusion": GATE_LABEL.get(fusion, fusion), "Runs": len(g)}
            for met, name in [("micro_auprc", "Micro-AP"), ("macro_f1", "Macro-F1")]:
                sd = g[met].std(ddof=1) if len(g) > 1 else 0.0
                row[name] = f"{g[met].mean():.4f} ± {sd:.4f}"
            row["Params"] = f"{int(g['n_params'].mean()):,}"
            rows.append(row)
        summary = pd.DataFrame(rows).sort_values(["Split", "Fusion"])
        summary.to_csv(os.path.join(GATE_OUT, "gate_ablation_summary.csv"), index=False)
        df.to_csv(os.path.join(GATE_OUT, "gate_ablation_per_run.csv"), index=False)
        print("\n=== Gate ablation (mean ± SD over seeds) ===")
        print(summary.to_string(index=False))
