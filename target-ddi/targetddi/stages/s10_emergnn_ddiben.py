"""Optional: TARGET-DDI on the single EmerGNN split shipped with DDI-Ben (superseded by s18)."""
# Section 8b: TARGET-DDI on EmerGNN's DrugBank splits (see markdown above).
if not RUN_EMERGNN_BENCHMARK:
    print('RUN_EMERGNN_BENCHMARK=False -> skipping EmerGNN benchmark')
else:
    import os, re, json, gc, time
    from pathlib import Path
    import numpy as np, pandas as pd, torch
    from sklearn.metrics import f1_score, cohen_kappa_score

    EMERGNN_VARIANTS = ["full_target", "shared_target_graph", "ablation_no_target"]
    EMERGNN_OUT = os.path.join(CFG.workdir, "emergnn_bench")
    os.makedirs(EMERGNN_OUT, exist_ok=True)
    # Persisted copy that survives "Save Version". Laid out as ddi_work/ so the checkpoint
    # loader (Section 0) restores it into WORKDIR when this output is attached next session.
    PERSIST_ROOT = "/kaggle/working/ddi_work" if os.path.isdir("/kaggle/working") else None
    PERSIST_BENCH = os.path.join(PERSIST_ROOT, "emergnn_bench") if PERSIST_ROOT else None
    import shutil

    def _restore_finished_units():
        """Never recompute: pull res_emergnn_*.json from any attached input or earlier output."""
        n = 0
        for root in [Path("/kaggle/input"), Path("/kaggle/working")]:
            if not root.exists():
                continue
            for f in root.rglob("res_emergnn_*.json"):
                dest = Path(EMERGNN_OUT) / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)
                    n += 1
        done = len(list(Path(EMERGNN_OUT).glob("res_emergnn_*.json")))
        print(f"  restored {n} finished benchmark units; {done} already complete (will be skipped)")

    def _persist(files=None):
        """Mirror benchmark outputs and reusable feature caches to /kaggle/working (no .pt files)."""
        if not PERSIST_ROOT:
            return
        os.makedirs(PERSIST_BENCH, exist_ok=True)
        for f in (files or os.listdir(EMERGNN_OUT)):
            src = os.path.join(EMERGNN_OUT, f)
            if os.path.isfile(src) and not f.endswith((".pt", ".tmp")):
                shutil.copy2(src, PERSIST_BENCH)
        for f in os.listdir(CFG.workdir):
            src = os.path.join(CFG.workdir, f)
            # feature caches, so the next session skips ChemBERTa/ATC/GO-Pfam work; the ESM cache
            # is skipped on purpose (it failed to fetch this session and should be refetched)
            if os.path.isfile(src) and f.endswith((".npy", ".json")) and not f.startswith("esm"):
                shutil.copy2(src, PERSIST_ROOT)


    _SEARCH_ROOTS = [Path("/kaggle/input"), Path("/kaggle/working"), Path(CFG.workdir), Path(".")]

    def _extract_zips():
        """If data-DB.zip was attached without being unzipped, extract it once into the workdir."""
        import zipfile
        dest = Path(CFG.workdir) / "emergnn_data"
        for root in _SEARCH_ROOTS:
            if not root.exists():
                continue
            for z in root.rglob("*.zip"):
                try:
                    with zipfile.ZipFile(z) as zf:
                        if any(n.endswith("node2id.json") for n in zf.namelist()):
                            if not dest.exists():
                                print(f"  extracting {z} -> {dest}")
                                zf.extractall(dest)
                            return
                except Exception:
                    continue

    _RAW = "https://raw.githubusercontent.com/LARS-research/DDI-Bench/main/DDI_Ben/EmerGNN/DrugBank/data"
    _FILES = ["node2id.json", "relation2id.json"] + [f"{s}/{p}_ddi.txt" for s in ("S1", "S2")
                                                      for p in ("train", "valid", "test")]

    def _download_emergnn_data():
        """Fetch EmerGNN's DrugBank S1/S2 splits from the DDI-Bench GitHub repo (same authors)."""
        import urllib.request
        dest = Path(CFG.workdir) / "emergnn_data" / "data"
        if all((dest / f).exists() and (dest / f).stat().st_size > 0 for f in _FILES):
            return
        print(f"  downloading EmerGNN DrugBank splits from GitHub (DDI-Bench) -> {dest}")
        for f in _FILES:
            out = dest / f
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(out.suffix + ".part")
            for attempt in range(3):
                try:
                    urllib.request.urlretrieve(f"{_RAW}/{f}", tmp)
                    os.replace(tmp, out)
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"   download failed for {f}: {e}")
                    time.sleep(3)

    def _find_emergnn_data_dir():
        """Return the directory holding node2id.json and the S1_k / S2_k fold folders, or None."""
        for attempt in range(2):
            for root in _SEARCH_ROOTS + [Path(CFG.workdir) / "emergnn_data"]:
                if not root.exists():
                    continue
                for p in root.rglob("test_ddi.txt"):
                    fold = p.parent
                    if not re.fullmatch(r"S[12](_\d+)?", fold.name):
                        continue
                    d = fold.parent
                    if (d / "node2id.json").exists():
                        return d
            if attempt == 0:
                _extract_zips()
                try:
                    _download_emergnn_data()
                except Exception as e:
                    print("   GitHub download skipped:", e)
        return None

    def _print_input_tree(root=Path("/kaggle/input"), depth=3, limit=40):
        if not root.exists():
            print("  /kaggle/input does not exist (not on Kaggle?)")
            return
        n = 0
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root)
            if len(rel.parts) <= depth:
                print("   ", rel, "/" if p.is_dir() else "")
                n += 1
                if n >= limit:
                    print("    ...")
                    break

    def _read_triples(path):
        rows = []
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
        return rows


    DATA_DIR = _find_emergnn_data_dir()
    if DATA_DIR is None:
        print("EmerGNN DrugBank data NOT found -> benchmark skipped; the rest of the notebook is unaffected.")
        print("Expected: <dir>/node2id.json and <dir>/S1/test_ddi.txt (or S1_1/...).")
        print("Attached inputs seen under /kaggle/input:")
        _print_input_tree()
        print("The automatic GitHub download also failed: turn Internet ON in the notebook settings, then rerun ONLY this cell.")
    else:
        node2id = json.load(open(DATA_DIR / "node2id.json"))
        int2name = {int(v): str(k) for k, v in node2id.items()}
        _known = set(id2n.keys())
        _overlap = sum(1 for n in int2name.values() if n in _known)
        print(f"EmerGNN data: {DATA_DIR}")
        print(f"  node2id entries: {len(int2name)} | matched to our DrugBank IDs: {_overlap}")
        if _overlap < 0.5 * len(int2name):
            raise ValueError(
                "node2id.json keys do not look like the DrugBank IDs used by this notebook. "
                f"Sample EmerGNN keys: {list(int2name.values())[:5]} | sample ours: {drug_ids[:5]}")

        FOLDS = sorted([f.name for f in DATA_DIR.iterdir()
                        if f.is_dir() and re.fullmatch(r"S[12](_\d+)?", f.name)],
                       key=lambda s: (s.split("_")[0], int(s.split("_")[1]) if "_" in s else 0))
        print(f"  folds found: {FOLDS}")
        _restore_finished_units()


        def _load_fold(fold):
            """Train/valid become canonical pair tables with label sets (what train_one expects).
            Test stays one row per EmerGNN triple so metrics match their protocol exactly."""
            parts, stats = {}, {}
            for split in ["train", "valid", "test"]:
                trip = _read_triples(DATA_DIR / fold / f"{split}_ddi.txt")
                for _, _, r in trip:
                    if not (0 <= r < C):
                        raise ValueError(f"{fold}/{split}: relation {r} outside 0..{C-1}")
                cov = [(int2name.get(h), int2name.get(t), r) for h, t, r in trip]
                ok = [(a, b, r) for a, b, r in cov if a in _known and b in _known]
                stats[split] = {"triples": len(trip), "covered": len(ok)}
                if split == "test":
                    parts["test_all"] = [(int2name.get(h), int2name.get(t), r) for h, t, r in trip]
                    parts["test"] = pd.DataFrame([{"u": a, "v": b, "y": [r]} for a, b, r in ok])
                else:
                    pl = {}
                    for a, b, r in ok:
                        pl.setdefault(canonical(a, b), set()).add(r)
                    parts[split] = pd.DataFrame([{"u": k[0], "v": k[1], "y": sorted(v)} for k, v in pl.items()])
            return parts, stats


        def _soft_audit(parts, fold):
            """Report (never crash on) pair overlap and unseen-endpoint counts of EmerGNN's own splits."""
            tr_pairs = set(zip(parts["train"]["u"], parts["train"]["v"]))
            tr_drugs = set(parts["train"]["u"]) | set(parts["train"]["v"])
            te = parts["test"]
            overlap = sum(1 for u, v in zip(te["u"], te["v"]) if canonical(u, v) in tr_pairs)
            unseen = pd.Series([(u not in tr_drugs) + (v not in tr_drugs) for u, v in zip(te["u"], te["v"])])
            dist = unseen.value_counts().sort_index().to_dict()
            want = 1 if fold.startswith("S1") else 2
            print(f"  [audit {fold}] test pairs also in train: {overlap} | unseen-endpoint counts: {dist} "
                  f"(expected all = {want})")
            return tr_drugs, {"train_test_pair_overlap": int(overlap),
                              "unseen_count_dist": {int(k): int(v) for k, v in dist.items()}}


        def _emergnn_metrics(label, pred):
            return {"f1": float(f1_score(label, pred, average="macro")),
                    "accuracy": float(np.mean(pred == label)),
                    "kappa": float(cohen_kappa_score(label, pred))}


        def run_emergnn_unit(fold, vname, seed):
            tag = f"emergnn_{fold}_{vname}_s{seed}"
            rp = os.path.join(EMERGNN_OUT, f"res_{tag}.json")
            if os.path.exists(rp):
                return json.load(open(rp))
            parts, stats = _load_fold(fold)
            tr_drugs, aud = _soft_audit(parts, fold)
            tmask = torch.zeros(len(drug_ids), dtype=torch.bool)
            for d in tr_drugs:
                tmask[id2n[d]] = True
            _, mode, mol_key = VARIANT_LOOKUP[vname]
            _tgt = TGT_BANK.get("features", TGT)
            T = {"mol": MOLBANK[mol_key], "tgt": _tgt, "tcount": TCOUNT, "tgt_bin": TGT_BIN,
                 "train_mask": tmask.to(DEVICE)}
            set_seed(seed)
            m = TargetDDI(MOLBANK[mol_key].shape[1], _tgt.shape[1], C, CFG, mode=mode).to(DEVICE)
            sp = {"train": parts["train"], "valid": parts["valid"], "test": parts["test"]}
            m, _thr, hist = train_one(m, sp, T, id2n, C, ckpt=os.path.join(EMERGNN_OUT, f"ck_{tag}.pt"), log=True)
            pd.DataFrame(hist).to_csv(os.path.join(EMERGNN_OUT, f"hist_{tag}.csv"), index=False)

            P = predict(m, parts["test"], T, id2n, C)
            y_cov = np.array([r[0] for r in parts["test"]["y"]], dtype=int)
            pred_cov = P.argmax(1)
            res = {f"covered_{k}": v for k, v in _emergnn_metrics(y_cov, pred_cov).items()}

            # Conservative score on EmerGNN's FULL test set: any triple we could not featurize
            # (drug missing from our feature table) is counted as a wrong prediction.
            y_all = np.array([r for _, _, r in parts["test_all"]], dtype=int)
            pred_all = (y_all + 1) % C
            cov_iter = iter(pred_cov.tolist())
            for i, (a, b, _) in enumerate(parts["test_all"]):
                if a in _known and b in _known:
                    pred_all[i] = next(cov_iter)
            res.update({f"all_{k}": v for k, v in _emergnn_metrics(y_all, pred_all).items()})
            res.update(dict(fold=fold, setting=fold.split("_")[0], variant=vname, seed=seed,
                            n_test_triples=stats["test"]["triples"], n_test_covered=stats["test"]["covered"],
                            **aud))
            np.savez_compressed(os.path.join(EMERGNN_OUT, f"pred_{tag}.npz"), y=y_cov, p=P)
            tmp = rp + ".tmp"
            json.dump(res, open(tmp, "w"), indent=2)
            os.replace(tmp, rp)
            _persist([os.path.basename(rp), f"hist_{tag}.csv", f"pred_{tag}.npz"])
            print(f"  [{tag}] F1={res['all_f1']*100:.1f} Acc={res['all_accuracy']*100:.1f} "
                  f"Kappa={res['all_kappa']*100:.1f} (covered {res['n_test_covered']}/{res['n_test_triples']})")
            del m
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return res


        # ---------------- run every fold x variant, missing-only ----------------
        results = []
        for k, fold in enumerate(FOLDS):
            for vname in EMERGNN_VARIANTS:
                if vname not in VARIANT_LOOKUP:
                    print(f"  [skip] {vname} not in VARIANTS")
                    continue
                for seed in ([int(fold.split("_")[1])] if "_" in fold else list(CFG.headline_seeds)):
                    try:
                        r = run_emergnn_unit(fold, vname, seed=seed)
                        if r:
                            results.append(r)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"  FAILED {fold}/{vname}/s{seed}: {e}")

        if results:
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(EMERGNN_OUT, "emergnn_results_per_fold.csv"), index=False)
            rows = []
            for (setting, vname), g in df.groupby(["setting", "variant"]):
                row = {"setting": setting, "variant": vname, "n_runs": len(g),
                       "test_coverage_min": float((g["n_test_covered"] / g["n_test_triples"]).min())}
                for met in ["f1", "accuracy", "kappa"]:
                    vals = g[f"all_{met}"] * 100
                    row[met] = f"{vals.mean():.1f} ± {vals.std(ddof=1) if len(vals) > 1 else 0.0:.1f}"
                rows.append(row)
            summary = pd.DataFrame(rows).sort_values(["setting", "variant"])
            summary.to_csv(os.path.join(EMERGNN_OUT, "emergnn_results_summary.csv"), index=False)
            print("\n=== TARGET-DDI on EmerGNN DrugBank (percent, mean ± SD over runs) ===")
            print(summary.to_string(index=False))
            _persist()
            if PERSIST_ROOT:
                print(f"persisted results to {PERSIST_BENCH} (kept on Save Version)")
