"""TWOSIDES emerging-drug splits (Table 14)."""
# Section 15
if not RUN_TWOSIDES:
    print('RUN_TWOSIDES=False -> skipped')
else:
    # ===== Section 15. Second dataset: TWOSIDES =====
    # Answers the single-dataset limitation. TWOSIDES is multi-label (209 side-effect types per
    # drug pair) and is evaluated as binary classification per type against one sampled negative
    # pair, reporting ROC-AUC, PR-AUC and AP@50 -- the protocol of the EmerGNN/DDI-Ben release,
    # whose emerging-drug splits (S1: one new drug, S2: two new drugs) we reuse.
    #
    # Protein features transfer because the release maps each TWOSIDES drug to its DrugBank ID
    # (id2drug.json), so the same UniProt/GO/Pfam pipeline applies; drugs with no DrugBank ID or
    # no annotation get an all-zero target row, which the gate handles through tcount.
    # Molecular features are rebuilt from the SMILES in the same file.
    # Needs Internet ON (clones the DDI-Ben repository). Crash-safe per (fold, variant, seed).
    import os, re, json, gc, time, glob, subprocess, math
    from pathlib import Path
    import numpy as np, pandas as pd, torch
    from sklearn.metrics import roc_auc_score, average_precision_score

    TS_OUT = os.path.join(CFG.workdir, "twosides"); os.makedirs(TS_OUT, exist_ok=True)
    TS_REPO = Path(CFG.workdir) / "DDI-Bench"
    TS_VARIANTS = ["full_target", "shared_target_graph", "ablation_no_target"]
    TS_SEEDS = list(CFG.headline_seeds)
    TS_SETTINGS = ["S1", "S2"]           # their names; = our S2 and S3
    TS_OUR = {"S1": "S2 (one new drug)", "S2": "S3 (two new drugs)"}


    def _ts_repo():
        if (TS_REPO / "DDI_Ben" / "EmerGNN" / "TWOSIDES" / "data").exists():
            return True
        try:
            print("   cloning DDI-Bench (TWOSIDES splits) ...")
            subprocess.run(["git", "clone", "--depth", "1",
                            "https://github.com/LARS-research/DDI-Bench.git", str(TS_REPO)],
                           check=True, capture_output=True, timeout=900)
            return (TS_REPO / "DDI_Ben" / "EmerGNN" / "TWOSIDES" / "data").exists()
        except Exception as e:
            print("   clone failed:", e)
            return False


    TS_ROOT = None
    if _ts_repo():
        TS_ROOT = TS_REPO / "DDI_Ben" / "EmerGNN" / "TWOSIDES" / "data"
    else:
        for base in [Path("/kaggle/input"), Path(CFG.workdir)]:
            if base.exists():
                for p in base.rglob("id2drug.json"):
                    if (p.parent / "S1" / "test_ddi.txt").exists():
                        TS_ROOT = p.parent; break
            if TS_ROOT:
                break

    if TS_ROOT is None:
        print("TWOSIDES data not found -> Section 15 skipped; the rest of the notebook is unaffected.")
        print("Fix: turn Internet ON and rerun (it clones github.com/LARS-research/DDI-Bench).")
    else:
        # id2drug.json covers only 604 drugs, while the split files reference ids up to 644.
        # cid2id.json + cid2smiles.json in the same repository cover all of them, so those are the
        # primary source; id2drug.json is used only for the DrugBank cross-reference that carries
        # the protein features over.
        id2drug = json.load(open(TS_ROOT / "id2drug.json"))
        _init = TS_REPO / "DDI_Ben" / "DDI_Ben" / "data" / "initial" / "twosides"
        if not (_init / "cid2id.json").exists():
            _hits = [q.parent for q in Path(TS_REPO).rglob("cid2id.json")] if TS_REPO.exists() else []
            _init = _hits[0] if _hits else _init
        cid2id = json.load(open(_init / "cid2id.json")) if (_init / "cid2id.json").exists() else {}
        cid2smiles = json.load(open(_init / "cid2smiles.json")) if (_init / "cid2smiles.json").exists() else {}

        _max_id = max([int(k) for k in id2drug] + list(cid2id.values()) + [0])
        for _f in sorted(TS_ROOT.rglob("*_ddi.txt")):      # never index past what the splits reference
            for _ln in open(_f):
                _q = _ln.split()
                if len(_q) >= 2:
                    _max_id = max(_max_id, int(_q[0]), int(_q[1]))
        TS_N = _max_id + 1

        id2cid = {v: k for k, v in cid2id.items()}
        ts_smiles, ts_db = {}, {}
        for i in range(TS_N):
            rec = id2drug.get(str(i), {})
            cid = id2cid.get(i)
            ts_smiles[i] = rec.get("smiles") or (cid2smiles.get(cid, "") if cid else "")
            ts_db[i] = rec.get("db") or None
        n_db = sum(1 for v in ts_db.values() if v)
        n_sm = sum(1 for v in ts_smiles.values() if v)
        print(f"TWOSIDES: {TS_N} drugs | {n_sm} with SMILES ({100*n_sm/TS_N:.0f}%) | "
              f"{n_db} with a DrugBank ID ({100*n_db/TS_N:.0f}%)")
        if n_sm < TS_N:
            print(f"   {TS_N-n_sm} drug(s) without SMILES get a zero molecular row")

        # ---- features: molecular from SMILES, protein rows reused via the DrugBank ID ----
        _ts_cache = os.path.join(CFG.workdir, f"twosides_feats_{CFG.feature_version}.npz")
        if os.path.exists(_ts_cache):
            _z = np.load(_ts_cache)
            TS_XMOL, TS_XTGT, TS_BIN = _z["mol"], _z["tgt"], _z["bin"]
            print("   TWOSIDES feature cache hit")
        else:
            sm = [ts_smiles[i] for i in range(TS_N)]
            _cb = chemberta(sm, os.path.join(CFG.workdir, "emb_twosides.npy"))
            TS_XMOL = np.concatenate([_cb, morgan(sm), physchem_features(sm)], 1).astype(np.float32)
            row = {d: k for k, d in enumerate(drug_ids)}          # DrugBank ID -> row in our tables
            _tg = TGT_BANK.get("features", TGT).detach().cpu().numpy()
            _bin = TGT_BIN.detach().cpu().numpy() if TGT_BIN.shape[1] else np.zeros((len(drug_ids), 0), np.float32)
            TS_XTGT = np.zeros((TS_N, _tg.shape[1]), np.float32)
            TS_BIN = np.zeros((TS_N, _bin.shape[1]), np.float32)
            hit = 0
            for i in range(TS_N):
                d = ts_db[i]
                if d in row:
                    TS_XTGT[i] = _tg[row[d]]
                    if _bin.shape[1]:
                        TS_BIN[i] = _bin[row[d]]
                    hit += 1
            np.savez_compressed(_ts_cache, mol=TS_XMOL, tgt=TS_XTGT, bin=TS_BIN)
            print(f"   features: mol={TS_XMOL.shape} tgt={TS_XTGT.shape} | "
                  f"{hit}/{TS_N} drugs got protein annotations ({100*hit/TS_N:.0f}%)")
        TS_MOL_T = torch.tensor(TS_XMOL).to(DEVICE)
        TS_TGT_T = torch.tensor(TS_XTGT).to(DEVICE)
        TS_BIN_T = torch.tensor(TS_BIN).to(DEVICE)
        TS_TCOUNT = torch.tensor(np.log1p(TS_XTGT.sum(1, keepdims=True)).astype(np.float32)).to(DEVICE)
        TS_ID2N = {str(i): i for i in range(TS_N)}

        def _ts_read(path):
            """Each line: head tail comma-separated-209-label-vector pos/neg-flag."""
            pos, neg = [], []
            with open(path) as f:
                for line in f:
                    q = line.strip().split(" ")
                    if len(q) < 4:
                        continue
                    x, y, z, w = int(q[0]), int(q[1]), q[2], int(q[3])
                    lab = np.fromstring(z, sep=",", dtype=np.float32)
                    (pos if w == 1 else neg).append((x, y, lab))
            return pos, neg

        def _ts_frame(rows, n_lab):
            return pd.DataFrame([{"u": str(x), "v": str(y), "y": np.where(l > 0)[0].tolist()} for x, y, l in rows])

        def _ts_load(setting):
            d = {}
            n_lab = None
            for split in ["train", "valid", "test"]:
                pos, neg = _ts_read(TS_ROOT / setting / f"{split}_ddi.txt")
                n_lab = n_lab or (len(pos[0][2]) if pos else 209)
                d[split] = _ts_frame(pos, n_lab)
                d[split + "_neg"] = neg
                d[split + "_pos"] = pos
            return d, n_lab

        def _ts_eval(pos, neg, score_fn, n_lab):
            """EmerGNN's TWOSIDES protocol (base_model.py): for each side-effect type, the
            positive pairs carrying that type are scored against the negative pairs at the
            SAME row positions, because the release pairs each positive with one sampled
            negative. Metrics are mean ROC-AUC, PR-AUC and AP@50 over types."""
            ps = score_fn([(x, y) for x, y, _ in pos])
            ns = score_fn([(x, y) for x, y, _ in neg])
            L = np.stack([l for _, _, l in pos])
            n = min(len(ps), len(ns))
            roc, prc, ap = [], [], []
            for r in range(n_lab):
                idx = np.where(L[:n, r] > 0)[0]
                if len(idx) == 0:
                    continue
                score = np.concatenate([ps[idx, r], ns[idx, r]])        # paired, not the first k
                label = np.concatenate([np.ones(len(idx)), np.zeros(len(idx))])
                roc.append(roc_auc_score(label, score))
                prc.append(average_precision_score(label, score))
                k = max(1, len(label) // 2)
                top = label[np.argsort(-score)][:k]
                ap.append(float(top.sum()) / k)
            return {"roc_auc": float(np.mean(roc)), "pr_auc": float(np.mean(prc)),
                    "ap50": float(np.mean(ap)), "n_types_scored": len(roc)}

        def run_ts_unit(setting, vname, seed):
            tag = f"ts_{setting}_{vname}_s{seed}"
            rp = os.path.join(TS_OUT, f"res_{tag}.json")
            if os.path.exists(rp):
                return json.load(open(rp))
            d, n_lab = _ts_load(setting)
            tr_drugs = set(d["train"]["u"]) | set(d["train"]["v"])
            tmask = torch.zeros(TS_N, dtype=torch.bool)
            for x in tr_drugs:
                _i = int(x)
                if 0 <= _i < TS_N:
                    tmask[_i] = True
            _, mode, _mk = VARIANT_LOOKUP[vname]
            T = {"mol": TS_MOL_T, "tgt": TS_TGT_T, "tcount": TS_TCOUNT, "tgt_bin": TS_BIN_T,
                 "train_mask": tmask.to(DEVICE)}
            set_seed(seed)
            m = TargetDDI(TS_MOL_T.shape[1], TS_TGT_T.shape[1], n_lab, CFG, mode=mode).to(DEVICE)
            n_par = sum(p.numel() for p in m.parameters() if p.requires_grad)
            t0 = time.time()
            m, _thr, hist = train_one(m, {"train": d["train"], "valid": d["valid"], "test": d["test"]},
                                      T, TS_ID2N, n_lab, ckpt=os.path.join(TS_OUT, f"ck_{tag}.pt"), log=False)
            train_s = time.time() - t0

            @torch.no_grad()
            def score_fn(pair_list):
                m.eval(); z = m.nodes(T); out = []
                for i0 in range(0, len(pair_list), CFG.eval_batch):
                    part = pair_list[i0:i0 + CFG.eval_batch]
                    pu = torch.tensor([p[0] for p in part], device=DEVICE)
                    pv = torch.tensor([p[1] for p in part], device=DEVICE)
                    zi, zj = z[pu], z[pv]
                    out.append(torch.sigmoid(m.head(torch.cat([zi + zj, (zi - zj).abs(), zi * zj], -1))).float().cpu().numpy())
                return np.nan_to_num(np.concatenate(out), nan=0.0, posinf=1.0, neginf=0.0)

            res = _ts_eval(d["test_pos"], d["test_neg"], score_fn, n_lab)
            res.update(dict(dataset="TWOSIDES", setting=setting, variant=vname, seed=seed,
                            n_params=int(n_par), train_seconds=round(train_s, 1), n_epochs=len(hist),
                            n_test_pos=len(d["test_pos"]), n_test_neg=len(d["test_neg"])))
            tmp = rp + ".tmp"; json.dump(res, open(tmp, "w"), indent=2); os.replace(tmp, rp)
            print(f"  [{tag}] ROC-AUC={res['roc_auc']*100:.1f} PR-AUC={res['pr_auc']*100:.1f} "
                  f"AP@50={res['ap50']*100:.1f} ({train_s/60:.1f} min)")
            del m; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return res

        _units = [(s, v, sd) for s in TS_SETTINGS for v in TS_VARIANTS for sd in TS_SEEDS if v in VARIANT_LOOKUP]
        def _ts_missing():
            return [u for u in _units if not os.path.exists(os.path.join(TS_OUT, f"res_ts_{u[0]}_{u[1]}_s{u[2]}.json"))]
        _todo = _ts_missing()
        print(f"SECTION 15 TWOSIDES: {len(_units)-len(_todo)}/{len(_units)} already complete, "
              f"{len(_todo)} to run now")
        _t0 = time.time(); _per = 0.0
        for _i, (setting, vname, seed) in enumerate(_todo, 1):
            if not budget_ok(_per, f"TWOSIDES run {_i}/{len(_todo)} ({setting}/{vname})"):
                break
            try:
                run_ts_unit(setting, vname, seed)
            except Exception as e:
                import traceback; traceback.print_exc(); print(f"  FAILED {setting}/{vname}/s{seed}: {e}")
            progress(_i, len(_todo), _t0, "TWOSIDES runs")
            _per = (time.time() - _t0) / _i / 3600
        print(f"SECTION 15 done: {len(_units)-len(_ts_missing())}/{len(_units)} complete overall")

        _res = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(TS_OUT, "res_ts_*.json")))]
        if _res:
            df = pd.DataFrame(_res)
            lab = {"full_target": "TARGET-DDI", "shared_target_graph": "TARGET-DDI-G", "ablation_no_target": "Mol-Full"}
            df["model"] = df["variant"].map(lab).fillna(df["variant"])
            rows = []
            for (setting, model), g in df.groupby(["setting", "model"]):
                row = {"Setting": TS_OUR.get(setting, setting), "Model": model, "Runs": len(g)}
                for met, name in [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("ap50", "AP@50")]:
                    v = g[met] * 100
                    row[name] = f"{v.mean():.1f} ± {(v.std(ddof=1) if len(v) > 1 else 0.0):.1f}"
                rows.append(row)
            summary = pd.DataFrame(rows).sort_values(["Setting", "Model"])
            summary.to_csv(os.path.join(TS_OUT, "twosides_summary.csv"), index=False)
            df.to_csv(os.path.join(TS_OUT, "twosides_per_run.csv"), index=False)
            print("\n=== TWOSIDES, emerging-drug splits (%, mean ± SD over seeds) ===")
            print(summary.to_string(index=False))
