"""CSMDDI retrained on the S1-S3 partitions (Table 4, CSMDDI rows)."""
# Section 21
if not globals().get("RUN_CSMDDI", False):
    print("RUN_CSMDDI=False -> skipped")
else:
    # ===== Section 21. CSMDDI retrained on the S1-S3 partitions =====
    # Reimplementation of CSMDDI (Liu et al., BMC Bioinformatics 23:75, 2022), following the
    # paper and the authors' code (github.com/itsosy/csmddi):
    #   1. RESCAL on the TRAINING interaction tensor only: A_k ~ E M_k E^T, one relation matrix
    #      per interaction type, rows of E L2-normalised (as in their torch Rescal), d = 200,
    #      xavier init, Adam, at most 1000 full-batch iterations, stop after 3 non-improving
    #      losses. Grid over learning rate and penalty in {0.001, 0.01, 0.1} as in their paper.
    #   2. Linear mapping drug attributes -> embedding by partial least squares regression
    #      (their PLSR class: n_components = embedding dim), fitted on training drugs.
    #   3. Score(i, j, k) = E_i M_k E_j^T; drugs seen in training keep their RESCAL embedding,
    #      unseen drugs get the mapped one. Scores are symmetrised because pairs are unordered.
    # Two attribute sets:
    #   csmddi_ctet       binary drug-protein vector over all associated proteins (their CTET)
    #   csmddi_same_input the exact inputs TARGET-DDI uses (full molecular bank + target branch)
    # Configuration is selected on validation micro-AP; per-type thresholds for macro-F1 are tuned
    # on validation exactly as for TARGET-DDI, and the same evaluate() produces every metric.
    # Crash-safe: each unit writes its grid progress and best state; finished units are skipped.
    import os, json, gc, time, hashlib
    import numpy as np, pandas as pd, torch
    import torch.nn.functional as F
    from sklearn.cross_decomposition import PLSRegression

    CS_OUT = os.path.join(CFG.workdir, "csmddi"); os.makedirs(CS_OUT, exist_ok=True)
    CS_DIM       = globals().get("CS_DIM", 200)
    CS_LR_GRID   = globals().get("CS_LR_GRID", [0.001, 0.01, 0.1])
    CS_WD_GRID   = globals().get("CS_WD_GRID", [0.001, 0.01, 0.1])
    CS_MAX_ITER  = globals().get("CS_MAX_ITER", 1000)
    CS_PATIENCE  = 3
    CS_SPLITS    = globals().get("CS_SPLITS", ["S2", "S3", "S1"])
    CS_SEEDS     = list(globals().get("CS_SEEDS", CFG.headline_seeds))
    CS_VARIANTS  = globals().get("CS_VARIANTS", ["csmddi_ctet", "csmddi_same_input"])
    CS_BOOTSTRAP = globals().get("CS_BOOTSTRAP", True)
    CS_UNIT_HOURS = 0.5          # time reserved before starting a unit (session budget check)
    CS_REQUIRE_SNAPSHOT_MATCH = True
    CS_LABEL = {"csmddi_ctet": "CSMDDI (protein vector)", "csmddi_same_input": "CSMDDI (TARGET-DDI inputs)"}
    CS_DEV = DEVICE

    def _atomic_json(obj, path):
        tmp = path + ".tmp"
        with open(tmp, "w") as fh: json.dump(obj, fh, indent=2)
        os.replace(tmp, path)

    def _atomic_npz(path, **arrs):
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **arrs); os.replace(tmp, path)

    # ---------------- attribute matrices (rows aligned with drug_ids) ----------------
    _tb = TGT_BIN.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(TGT_BIN) else np.asarray(TGT_BIN, np.float32)
    if _tb.ndim != 2 or _tb.shape[1] == 0:
        raise RuntimeError("TGT_BIN is empty: the target fetch failed this session, CSMDDI cannot run.")
    CS_FEATURES = {"csmddi_ctet": _tb,
                   "csmddi_same_input": np.concatenate([np.asarray(Xmol_full, np.float32),
                                                        np.asarray(Xtgt, np.float32)], 1)}
    for _k, _v in CS_FEATURES.items():
        assert _v.shape[0] == len(drug_ids), f"{_k}: row count {_v.shape[0]} != {len(drug_ids)} drugs"
        print(f"  {_k:20s} attributes: {_v.shape[1]} dims")

    # ---------------- annotation snapshot: fingerprint + check against the headline run ----------------
    def _fp(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
    CS_FP = {"tgt_bin": _fp(_tb), "xtgt": _fp(np.asarray(Xtgt, np.float32)),
             "xmol_full": _fp(np.asarray(Xmol_full, np.float32)),
             "shape_tgt_bin": list(_tb.shape), "shape_xtgt": list(np.asarray(Xtgt).shape)}
    _fp_path = os.path.join(CS_OUT, "feature_fingerprint.json")
    if os.path.exists(_fp_path):
        _old = json.load(open(_fp_path))
        if _old != CS_FP:
            raise RuntimeError(
                "Feature fingerprint differs from the one recorded when earlier CSMDDI units ran.\n"
                f"  recorded: {_old}\n  now:      {CS_FP}\n"
                "Mixing units trained on different annotation snapshots would make the rows incomparable. "
                "Either delete the csmddi/ folder and rerun all units, or restore the original caches.")
        print("  feature fingerprint matches earlier CSMDDI units")
    else:
        _atomic_json(CS_FP, _fp_path)

    def _snapshot_check():
        """Re-score the stored TARGET-DDI S3 seed-0 checkpoint on THIS session's features.
        If micro-AP matches the stored value, the protein annotations equal the headline run's."""
        tag = f"{CFG.dataset}_{CFG.feature_version}_S3_full_target_s0"
        ck = os.path.join(CFG.workdir, f"ck_{tag}.pt"); rp = os.path.join(CFG.workdir, f"res_{tag}.json")
        if not (os.path.exists(ck) and os.path.exists(rp)):
            print(f"  [snapshot] {os.path.basename(ck)} not found: cannot verify that this session's protein "
                  "annotations match the headline runs. Attach the paper-grid output to verify.")
            return None
        sp, _ = make_split(pairs, "S3", CFG.frac, 0)
        tr = set(sp["train"]["u"]) | set(sp["train"]["v"])
        tmask = torch.zeros(len(drug_ids), dtype=torch.bool)
        for d in tr: tmask[id2n[d]] = True
        T = {"mol": MOLBANK["full"], "tgt": TGT, "tcount": TCOUNT, "tgt_bin": TGT_BIN, "train_mask": tmask.to(DEVICE)}
        try:
            m = TargetDDI(MOLBANK["full"].shape[1], TGT.shape[1], C, CFG, mode="gated").to(DEVICE)
            st = torch.load(ck, map_location=DEVICE, weights_only=False)
            sd = (st.get("best_state") or {}).get("model") or st["model"]
            m.load_state_dict(sd)
        except Exception as e:
            print(f"  [snapshot] checkpoint does not load on this session's features ({e}); feature widths differ.")
            return False
        p = predict(m, sp["test"], T, id2n, C); y = multihot(sp["test"]["y"], C)
        now = _ap_micro(y, p); stored = float(json.load(open(rp))["micro_auprc"])
        ok = abs(now - stored) < 1e-3
        print(f"  [snapshot] TARGET-DDI S3 s0 micro-AP: stored {stored:.4f}, re-scored now {now:.4f} -> "
              f"{'MATCH' if ok else 'MISMATCH'}")
        del m; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()
        return ok

    CS_SNAPSHOT = _snapshot_check()
    if CS_SNAPSHOT is False and CS_REQUIRE_SNAPSHOT_MATCH:
        raise RuntimeError(
            "This session's protein annotations differ from the headline runs, so CSMDDI would not be "
            "trained on the same inputs as TARGET-DDI. Restore the cached annotation files from the "
            "headline session (do not purge drug_targets*.json / ec_go_class.json) and rerun.")

    # ---------------- RESCAL with an exact, memory-light loss ----------------
    # ||A_k - E M_k E^T||_F^2 = |A_k| - 2 sum_{(i,j) in A_k} E_i M_k E_j^T + tr(M_k G M_k^T G),
    # with G = E^T E and A binary. Identical to the dense loss of the original code but never
    # builds the C x n x n tensor. Verified against the dense form below before any run.
    def _rescal_loss(En, M, ki, ii, jj, nnz):
        EM = torch.matmul(En, M)                          # (C, n, d)
        pos = (EM[ki, ii] * En[jj]).sum()
        G = En.T @ En
        quad = torch.einsum("kab,kba->", M @ G, M.transpose(1, 2) @ G)
        return nnz - 2.0 * pos + quad

    def _selftest_rescal():
        g = torch.Generator().manual_seed(0); n, d, K = 9, 4, 3
        E = F.normalize(torch.randn(n, d, generator=g), dim=1); M = torch.randn(K, d, d, generator=g)
        A = (torch.rand(K, n, n, generator=g) < 0.3).float(); A = ((A + A.transpose(1, 2)) > 0).float()
        dense = ((A - E @ M @ E.T) ** 2).sum()
        k, i, j = A.nonzero(as_tuple=True)
        fast = _rescal_loss(E, M, k, i, j, float(A.sum()))
        assert torch.allclose(dense, fast, rtol=1e-4, atol=1e-3), (dense.item(), fast.item())
        print(f"  [self-test] RESCAL loss: dense {dense.item():.4f} == exact sparse {fast.item():.4f}")
    _selftest_rescal()

    def _train_tensor(train_df, loc):
        k, i, j = [], [], []
        for u, v, ys in zip(train_df["u"], train_df["v"], train_df["y"]):
            a, b = loc[u], loc[v]
            for c in ys:
                k += [c, c]; i += [a, b]; j += [b, a]
        idx = np.unique(np.stack([k, i, j], 1), axis=0)   # dedupe (also covers u == v)
        t = torch.tensor(idx, dtype=torch.long, device=CS_DEV)
        return t[:, 0], t[:, 1], t[:, 2], float(len(idx))

    def _rescal_fit(n, ki, ii, jj, nnz, lr, wd, seed):
        torch.manual_seed(seed)
        E = torch.nn.Parameter(torch.empty(n, CS_DIM, device=CS_DEV))
        M = torch.nn.Parameter(torch.empty(C, CS_DIM, CS_DIM, device=CS_DEV))
        torch.nn.init.xavier_uniform_(E); torch.nn.init.xavier_uniform_(M)
        opt = torch.optim.Adam([E, M], lr=lr, weight_decay=wd)
        best, bad, it = float("inf"), 0, 0
        bE, bM = E.detach().clone(), M.detach().clone()
        for it in range(1, CS_MAX_ITER + 1):
            opt.zero_grad(set_to_none=True)
            loss = _rescal_loss(F.normalize(E, dim=1), M, ki, ii, jj, nnz)
            if not torch.isfinite(loss):
                break
            loss.backward(); opt.step()
            lv = float(loss.item())
            if lv < best:
                best, bad = lv, 0; bE.copy_(E.detach()); bM.copy_(M.detach())
            else:
                bad += 1
                if bad >= CS_PATIENCE: break
        return F.normalize(bE, dim=1), bM, it, best

    def _map_unseen(Fx, tr_idx, other_idx, Etr):
        """PLS regression attributes -> embedding, fitted on training drugs (their PLSR)."""
        Eall = np.zeros((Fx.shape[0], CS_DIM), np.float32)
        Eall[tr_idx] = Etr
        if len(other_idx):
            Xtr = Fx[tr_idx]
            keep = Xtr.std(0) > 0                        # constant columns carry no signal
            ncomp = int(min(CS_DIM, keep.sum(), len(tr_idx) - 1))
            if ncomp < 1:                                 # no informative attribute: mean embedding
                Eall[other_idx] = Etr.mean(0, keepdims=True)
            else:
                pls = PLSRegression(n_components=ncomp, scale=True, max_iter=500)
                pls.fit(Xtr[:, keep], Etr)
                Eall[other_idx] = pls.predict(Fx[other_idx][:, keep]).astype(np.float32)
        return Eall

    def _score(Eall_t, M, df, bs=2048):
        u = torch.tensor([id2n[x] for x in df["u"]], device=CS_DEV)
        v = torch.tensor([id2n[x] for x in df["v"]], device=CS_DEV)
        out = []
        with torch.no_grad():
            for s in range(0, len(u), bs):
                eu, ev = Eall_t[u[s:s+bs]], Eall_t[v[s:s+bs]]
                a = torch.einsum("pd,kde,pe->pk", eu, M, ev)
                b = torch.einsum("pd,kde,pe->pk", ev, M, eu)
                out.append(((a + b) / 2).float().cpu().numpy())
        P = np.concatenate(out) if out else np.zeros((0, C), np.float32)
        return np.nan_to_num(P, nan=0.0, posinf=1e6, neginf=-1e6)

    def run_csmddi_unit(split, vname, seed):
        tag = f"{CFG.dataset}_{CFG.feature_version}_{split}_{vname}_s{seed}"
        rp = os.path.join(CS_OUT, f"res_{tag}.json")
        if os.path.exists(rp): return json.load(open(rp))
        Fx = CS_FEATURES[vname]
        sp, info = make_split(pairs, split, CFG.frac, seed)
        if len(sp["test"]) == 0 or len(sp["valid"]) == 0:
            print(f"  {split}: empty split"); return None
        audit(sp, split, id2n, None)
        tr_drugs = sorted(set(sp["train"]["u"]) | set(sp["train"]["v"]), key=str)
        tr_idx = np.array([id2n[d] for d in tr_drugs]); other_idx = np.setdiff1d(np.arange(len(drug_ids)), tr_idx)
        loc = {d: i for i, d in enumerate(tr_drugs)}
        ki, ii, jj, nnz = _train_tensor(sp["train"], loc)
        yv = multihot(sp["valid"]["y"], C)

        gp = os.path.join(CS_OUT, f"grid_{tag}.json"); bp = os.path.join(CS_OUT, f"best_{tag}.npz")
        grid = json.load(open(gp)) if os.path.exists(gp) else {"configs": [], "best": None}
        done = {(c["lr"], c["wd"]) for c in grid["configs"]}
        for lr in CS_LR_GRID:
            for wd in CS_WD_GRID:
                if (lr, wd) in done: continue
                t0 = time.time()
                Etr, M, iters, loss = _rescal_fit(len(tr_drugs), ki, ii, jj, nnz, lr, wd, seed)
                Eall = _map_unseen(Fx, tr_idx, other_idx, Etr.cpu().numpy())
                Eall_t = torch.tensor(Eall, device=CS_DEV)
                pv = _score(Eall_t, M, sp["valid"])
                vap = float(_ap_micro(yv, pv))
                vap = vap if vap == vap else -1.0
                rec = {"lr": lr, "wd": wd, "val_micro_ap": vap, "iters": iters, "train_loss": loss,
                       "sec": round(time.time() - t0, 1)}
                if grid["best"] is None or vap > grid["best"]["val_micro_ap"]:
                    _atomic_npz(bp, Eall=Eall, M=M.cpu().numpy())
                    grid["best"] = rec
                grid["configs"].append(rec); _atomic_json(grid, gp)
                print(f"    [{tag}] lr={lr} wd={wd} iters={iters} val microAP={vap:.4f} ({rec['sec']}s)", flush=True)
                del Etr, M, Eall_t; gc.collect()
                if CS_DEV == "cuda": torch.cuda.empty_cache()
        st = np.load(bp)
        Eall_t = torch.tensor(st["Eall"], device=CS_DEV); M = torch.tensor(st["M"], device=CS_DEV)
        pv = _score(Eall_t, M, sp["valid"]); thr = tune_thresholds(yv, pv)
        yte = multihot(sp["test"]["y"], C); pt = _score(Eall_t, M, sp["test"])
        res = evaluate(yte, pt, thr)
        _atomic_npz(os.path.join(CS_OUT, f"pred_{tag}.npz"), y=yte, p=pt, thr=thr)
        res.update(dict(dataset=CFG.dataset, split=split, variant=vname, seed=seed, n_test=len(sp["test"]),
                        best_lr=grid["best"]["lr"], best_wd=grid["best"]["wd"],
                        val_micro_ap=grid["best"]["val_micro_ap"], attr_dims=int(Fx.shape[1]),
                        embedding_dim=CS_DIM, snapshot_verified=CS_SNAPSHOT, feature_fp=CS_FP["xtgt"]))
        _atomic_json(res, rp)
        print(f"  [{tag}] microAP={res['micro_auprc']:.4f} macroF1={res['macro_f1']:.4f} PR-lift={res['pr_lift']:.1f}x", flush=True)
        del Eall_t, M; gc.collect()
        if CS_DEV == "cuda": torch.cuda.empty_cache()
        return res

    # ---------------- run: protein-vector variant on cold-start splits first ----------------
    units = [(sp_, v, s) for v in CS_VARIANTS for sp_ in CS_SPLITS for s in CS_SEEDS]
    units.sort(key=lambda u: (u[0] == "S1", CS_VARIANTS.index(u[1]), CS_SPLITS.index(u[0]), u[2]))
    _t0 = time.time(); _done = 0
    pending = [u for u in units if not os.path.exists(os.path.join(
        CS_OUT, f"res_{CFG.dataset}_{CFG.feature_version}_{u[0]}_{u[1]}_s{u[2]}.json"))]
    print(f"\nCSMDDI: {len(units) - len(pending)}/{len(units)} units already complete, {len(pending)} to run")
    for split, vname, seed in pending:
        if not budget_ok(CS_UNIT_HOURS, f"CSMDDI {split}/{vname}/s{seed}"): break
        try:
            run_csmddi_unit(split, vname, seed)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"  FAILED {split}/{vname}/s{seed}: {e}")
        _done += 1; progress(_done, len(pending), _t0, "CSMDDI units")

    # ---------------- summary, Table 4 rows, side-by-side with TARGET-DDI ----------------
    rows = [json.load(open(os.path.join(CS_OUT, f))) for f in sorted(os.listdir(CS_OUT))
            if f.startswith("res_") and f.endswith(".json")]
    if rows:
        R = pd.DataFrame(rows)
        agg = (R.groupby(["split", "variant"])[["micro_auprc", "macro_f1", "pr_lift"]]
                .agg(["mean", "std", "count"]).reset_index())
        agg.columns = ["_".join([c for c in col if c]) for col in agg.columns]
        agg.to_csv(os.path.join(CS_OUT, "csmddi_summary.csv"), index=False)
        R.to_csv(os.path.join(CS_OUT, "csmddi_runs_raw.csv"), index=False)
        print("\n" + agg.round(4).to_string(index=False))

        # reference rows from the headline grid, if attached
        ref = []
        for v, lab in [("full_target", "TARGET-DDI"), ("ablation_no_target", "Mol-Full")]:
            for sp_ in ["S1", "S2", "S3"]:
                for s in CS_SEEDS:
                    p_ = os.path.join(CFG.workdir, f"res_{CFG.dataset}_{CFG.feature_version}_{sp_}_{v}_s{s}.json")
                    if os.path.exists(p_):
                        d = json.load(open(p_)); ref.append({"model": lab, "split": sp_, "seed": s,
                                                             "micro_auprc": d["micro_auprc"], "macro_f1": d["macro_f1"]})
        if ref:
            print("\nReference (headline runs on the same partitions):")
            print(pd.DataFrame(ref).groupby(["model", "split"])[["micro_auprc", "macro_f1"]].agg(["mean", "std"]).round(4).to_string())

        # LaTeX rows in the column order of Table 4 (S1 AP, F1 | S2 AP, F1, lift | S3 AP, F1, lift)
        def _cell(v, sp_, key, sd=False, lift=False):
            q = R[(R.variant == v) & (R.split == sp_)][key]
            if len(q) == 0: return "--"
            if lift: return f"${q.mean():.1f}\\times$"
            if sd and len(q) > 1: return f"${q.mean():.4f}\\pm{q.std(ddof=1):.4f}$"
            return f"{q.mean():.4f}"
        tex = []
        for v in CS_VARIANTS:
            if v not in set(R.variant): continue
            c = [_cell(v, "S1", "micro_auprc", True), _cell(v, "S1", "macro_f1"),
                 _cell(v, "S2", "micro_auprc", True), _cell(v, "S2", "macro_f1"), _cell(v, "S2", "pr_lift", lift=True),
                 _cell(v, "S3", "micro_auprc", True), _cell(v, "S3", "macro_f1"), _cell(v, "S3", "pr_lift", lift=True)]
            n_seeds = R[R.variant == v].groupby("split").size().to_dict()
            tex.append(f"% {CS_LABEL[v]}  seeds per split: {n_seeds}  snapshot_verified={CS_SNAPSHOT}\n"
                       f"{CS_LABEL[v]} & " + " & ".join(c) + " \\\\")
        open(os.path.join(CS_OUT, "csmddi_table4_rows.tex"), "w").write("\n".join(tex) + "\n")
        print("\nTable 4 rows:\n" + "\n".join(tex))

        # paired bootstrap: TARGET-DDI minus CSMDDI on the seed-0 partitions (same test pairs)
        if CS_BOOTSTRAP:
            bs_rows = []
            for v in CS_VARIANTS:
                for sp_ in ["S2", "S3"]:
                    tag_c = f"{CFG.dataset}_{CFG.feature_version}_{sp_}_{v}_s0"
                    tag_t = f"{CFG.dataset}_{CFG.feature_version}_{sp_}_full_target_s0"
                    pc = os.path.join(CS_OUT, f"pred_{tag_c}.npz"); pt_ = os.path.join(CFG.workdir, f"pred_{tag_t}.npz")
                    bsp = os.path.join(CS_OUT, f"bootstrap_{tag_c}.json")
                    if os.path.exists(bsp): bs_rows.append(json.load(open(bsp))); continue
                    if not (os.path.exists(pc) and os.path.exists(pt_)):
                        print(f"  [bootstrap] skip {sp_}/{v}: predictions missing"); continue
                    a, b = np.load(pc), np.load(pt_)
                    if a["y"].shape != b["y"].shape or not np.array_equal(a["y"], b["y"]):
                        print(f"  [bootstrap] skip {sp_}/{v}: test pairs differ from the TARGET-DDI run"); continue
                    if not budget_ok(0.3, f"bootstrap {sp_}/{v}"): break
                    dlt, lo, hi = paired_bootstrap(a["y"], b["p"], a["p"], CFG.n_bootstrap)
                    r = {"split": sp_, "csmddi_variant": v, "delta_micro_ap": dlt, "ci_lo": lo, "ci_hi": hi,
                         "n_bootstrap": CFG.n_bootstrap}
                    _atomic_json(r, bsp); bs_rows.append(r)
            if bs_rows:
                B = pd.DataFrame(bs_rows); B.to_csv(os.path.join(CS_OUT, "csmddi_bootstrap_summary.csv"), index=False)
                print("\nTARGET-DDI minus CSMDDI (seed 0, paired bootstrap 95% CI):\n" + B.round(4).to_string(index=False))
    print(f"\nSection 21 outputs in {CS_OUT}  (snapshot_verified={CS_SNAPSHOT})")
