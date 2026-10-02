"""EmerGNN's released drug partitions from Zenodo, plus warm start (Table 7)."""
# Section 14
if not RUN_ZENODO:
    print('RUN_ZENODO=False -> skipped')
else:
    # ===== Section 14. EmerGNN's original five folds (Zenodo) + warm start =====
    # Section 8b used the single split released with DDI-Ben, so the comparison with EmerGNN's
    # published table was not split-matched. This section downloads EmerGNN's own five drug
    # partitions from the Zenodo record cited in their paper and evaluates on exactly those,
    # plus the warm-start setting (their S0), making the comparison direct.
    #
    # Naming: EmerGNN S0/S1/S2 = our S1 (warm start) / S2 (one new drug) / S3 (two new drugs).
    # Their folds are named S1_1..S1_5 and S2_1..S2_5.
    #
    # Needs Internet ON. If the download fails, attach data-DB.zip as a Kaggle dataset instead.
    # Crash-safe: one res_*.json per (fold, variant, seed); finished units are skipped.
    import os, re, json, gc, time, glob, zipfile, shutil, urllib.request
    from pathlib import Path
    import numpy as np, pandas as pd, torch
    from sklearn.metrics import f1_score, cohen_kappa_score

    ZEN_OUT = os.path.join(CFG.workdir, "emergnn_zenodo"); os.makedirs(ZEN_OUT, exist_ok=True)
    ZEN_DATA = Path(CFG.workdir) / "emergnn_zenodo_data"
    ZEN_DOI = "10.5281/zenodo.10016715"
    ZEN_VARIANTS = ["full_target", "shared_target_graph", "ablation_no_target"]
    ZEN_SEEDS = [0]                      # one seed per fold; the five folds carry the variance
    ZEN_SETTINGS = ["S0", "S1", "S2"]    # EmerGNN names; S0 is the warm-start control
    OUR_NAME = {"S0": "S1 (warm start)", "S1": "S2 (one new drug)", "S2": "S3 (two new drugs)"}


    def _unzip_all(root, depth=0):
        """Extract every archive into its OWN folder. The Zenodo record holds data-DB.zip
        (DrugBank) and data-TS.zip (TWOSIDES), each containing a directory called "data";
        extracting them side by side would merge two different datasets into one folder."""
        if depth > 3:
            return
        again = False
        for z in sorted(Path(root).rglob("*.zip")):
            marker = z.with_suffix(z.suffix + ".unpacked")
            if marker.exists():
                continue
            try:
                dest = z.parent / z.stem
                dest.mkdir(exist_ok=True)
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(dest)
                marker.write_text("done"); again = True
                print(f"   extracted {z.name} -> {dest.name}/")
            except Exception as e:
                print(f"   could not extract {z.name}: {e}")
        if again:
            _unzip_all(root, depth + 1)


    def _zen_fetch(url, dest, size=0, md5=None, chunk=1 << 20, stall_s=90, attempts=8):
        """Streaming download with progress, stall detection and byte-range resume, so a slow
        or dropped connection cannot hang an unattended run."""
        import hashlib
        for attempt in range(1, attempts + 1):
            have = dest.stat().st_size if dest.exists() else 0
            if size and have == size:
                break
            if size and have > size:
                dest.unlink(); have = 0
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            if have:
                req.add_header("Range", f"bytes={have}-")
            mode = "ab" if have else "wb"
            t0 = last = time.time(); start = have
            try:
                with urllib.request.urlopen(req, timeout=60) as r, open(dest, mode) as fh:
                    if have and getattr(r, "status", 200) != 206:
                        fh.close(); dest.unlink()
                        print("   server ignored the resume request, restarting the file", flush=True)
                        continue
                    while True:
                        buf = r.read(chunk); now = time.time()
                        if not buf:
                            break
                        fh.write(buf); have += len(buf); last = now
                        if have % (25 * chunk) < chunk:
                            rate = (have - start) / max(now - t0, 1e-6) / 1e6
                            pct = f"{100*have/size:.0f}%" if size else "?"
                            print(f"      {have/1e6:.0f} / {size/1e6:.0f} MB ({pct}) at {rate:.1f} MB/s", flush=True)
                        if now - last > stall_s:
                            raise TimeoutError("stalled")
            except Exception as e:
                got = dest.stat().st_size if dest.exists() else 0
                print(f"   attempt {attempt}/{attempts} stopped after {got/1e6:.0f} MB: {e}", flush=True)
                if attempt == attempts:
                    return False
                time.sleep(min(5 * attempt, 30)); continue
            if not size or dest.stat().st_size == size:
                break
        got = dest.stat().st_size if dest.exists() else 0
        if size and got != size:
            return False
        if md5 and str(md5).startswith("md5:"):
            h = hashlib.md5()
            with open(dest, "rb") as fh:
                for b in iter(lambda: fh.read(1 << 22), b""):
                    h.update(b)
            if h.hexdigest() != md5.split(":", 1)[1]:
                print("   md5 mismatch -> deleting the file so the next attempt refetches it")
                dest.unlink(); return False
        return True


    def _zen_download():
        """Fetch and unpack EmerGNN's DrugBank splits from Zenodo (public, no auth)."""
        if any(ZEN_DATA.rglob("test_ddi.txt")):
            return True
        ZEN_DATA.mkdir(parents=True, exist_ok=True)
        try:
            api = f"https://zenodo.org/api/records/{ZEN_DOI.split('.')[-1]}"
            with urllib.request.urlopen(api, timeout=60) as r:
                rec = json.load(r)
            files = rec.get("files", [])
            print(f"   Zenodo record: {len(files)} file(s)")
            for f in files:
                key, url, size = f.get("key"), f.get("links", {}).get("self"), f.get("size", 0)
                if not (key and url):
                    continue
                dest = ZEN_DATA / key
                if dest.exists() and size and dest.stat().st_size == size:
                    print(f"   {key} already downloaded")
                    continue
                print(f"   downloading {key} ({size/1e6:.0f} MB) ...", flush=True)
                _zen_fetch(url, dest, size, f.get("checksum"))
            _unzip_all(ZEN_DATA)
            return any(ZEN_DATA.rglob("test_ddi.txt"))
        except Exception as e:
            print("   Zenodo download failed:", e)
            return False


    def _fold_format(fp):
        """A DrugBank line is "head tail <int type>"; a TWOSIDES line carries a label vector."""
        with open(fp) as fh:
            for ln in fh:
                q = ln.split()
                if len(q) >= 3:
                    return "drugbank" if q[2].isdigit() else "twosides"
        return "empty"


    def _zen_find_root():
        """Pick the DrugBank fold tree, identified by its file format, and the node2id.json
        that belongs with it."""
        roots = [ZEN_DATA, Path("/kaggle/input"), Path(CFG.workdir)]
        groups = {}
        for root in roots:
            if not root.exists():
                continue
            for p in root.rglob("test_ddi.txt"):
                fold = p.parent
                if not re.fullmatch(r"S[012](_\d+)?", fold.name):
                    continue
                groups.setdefault(fold.parent, set()).add(fold.name)
        best = None
        for d, folds in groups.items():
            db = sorted(f for f in folds if _fold_format(d / f / "test_ddi.txt") == "drugbank")
            if not db:
                continue
            n2i = None
            for up in [d, d.parent, d.parent.parent]:
                if (up / "node2id.json").exists():
                    n2i = up / "node2id.json"; break
            if n2i is None:
                hits = list(d.rglob("node2id.json"))
                n2i = hits[0] if hits else None
            if n2i is None:
                continue
            if best is None or len(db) > len(best[1]):
                best = (d, db, n2i)
        return best if best else (None, [], None)


    if not any(ZEN_DATA.rglob("test_ddi.txt")):
        _zen_download()
    _stale = ZEN_DATA / "data"          # merged folder left by an earlier extraction
    if _stale.exists() and any(_stale.rglob("test_ddi.txt")):
        _fmts = {_fold_format(q) for q in _stale.rglob("test_ddi.txt")}
        if len(_fmts) > 1:
            shutil.rmtree(_stale)
            for _m in ZEN_DATA.rglob("*.unpacked"):
                _m.unlink()
            print("   cleared a previously merged extraction folder")
    _unzip_all(ZEN_DATA)                       # also unpacks a zip attached as a dataset input
    ZEN_ROOT, ZEN_FOLDS, ZEN_MAP = _zen_find_root()

    if ZEN_ROOT is None:
        print("EmerGNN split data not found -> Section 14 skipped; the rest of the notebook is unaffected.")
        print(f"Fix: turn Internet ON and rerun this cell (it downloads {ZEN_DOI}),")
        print("or attach the unzipped data-DB dataset via '+ Add Input'.")
        print("What is present under the download folder:")
        _n = 0
        for _p in sorted(ZEN_DATA.rglob("*")):
            if _n < 40:
                print("   ", _p.relative_to(ZEN_DATA), "/" if _p.is_dir() else f"({_p.stat().st_size/1e6:.1f} MB)")
            _n += 1
        print(f"   ({_n} entries total)")
    else:
        node2id = json.load(open(ZEN_MAP))
        int2name = {int(v): str(k) for k, v in node2id.items()}
        _known = set(id2n.keys())
        ZEN_FOLDS = [f for f in ZEN_FOLDS if f.split("_")[0] in ZEN_SETTINGS]
        print(f"EmerGNN data: {ZEN_ROOT}")
        print(f"  drugs {len(int2name)} | matched to our features {sum(1 for n in int2name.values() if n in _known)}")
        print(f"  folds: {ZEN_FOLDS}")
        if not any("_" in f for f in ZEN_FOLDS):
            print("  NOTE: only the single-split release was found, so these results repeat Section 8b's")
            print("        partition rather than EmerGNN's own five folds.")

        def _zen_read(path):
            rows = []
            with open(path) as f:
                for line in f:
                    q = line.split()
                    if len(q) >= 3:
                        rows.append((int(q[0]), int(q[1]), int(q[2])))
            return rows

        def _zen_load(fold):
            parts, stats = {}, {}
            for split in ["train", "valid", "test"]:
                trip = _zen_read(ZEN_ROOT / fold / f"{split}_ddi.txt")
                named = [(int2name.get(h), int2name.get(t), r) for h, t, r in trip]
                ok = [(a, b, r) for a, b, r in named if a in _known and b in _known and 0 <= r < C]
                stats[split] = {"triples": len(trip), "covered": len(ok)}
                if split == "test":
                    parts["test_all"] = named
                    parts["test"] = pd.DataFrame([{"u": a, "v": b, "y": [r]} for a, b, r in ok])
                else:
                    pl = {}
                    for a, b, r in ok:
                        pl.setdefault(canonical(a, b), set()).add(r)
                    parts[split] = pd.DataFrame([{"u": k[0], "v": k[1], "y": sorted(v)} for k, v in pl.items()])
            return parts, stats

        def _zen_metrics(label, pred):
            return {"f1": float(f1_score(label, pred, average="macro")),
                    "accuracy": float(np.mean(pred == label)),
                    "kappa": float(cohen_kappa_score(label, pred))}

        def run_zen_unit(fold, vname, seed):
            tag = f"zen_{fold}_{vname}_s{seed}"
            rp = os.path.join(ZEN_OUT, f"res_{tag}.json")
            if os.path.exists(rp):
                return json.load(open(rp))
            parts, stats = _zen_load(fold)
            if len(parts["test"]) == 0 or len(parts["valid"]) == 0:
                print(f"  [{tag}] empty split, skipped"); return None
            tr_pairs = set(zip(parts["train"]["u"], parts["train"]["v"]))
            tr_drugs = set(parts["train"]["u"]) | set(parts["train"]["v"])
            overlap = sum(1 for u, v in zip(parts["test"]["u"], parts["test"]["v"]) if canonical(u, v) in tr_pairs)
            unseen = [int(u not in tr_drugs) + int(v not in tr_drugs)
                      for u, v in zip(parts["test"]["u"], parts["test"]["v"])]
            dist = pd.Series(unseen).value_counts().sort_index().to_dict()
            print(f"  [audit {fold}] train/test pair overlap: {overlap} | unseen endpoints: {dist}")
            tmask = torch.zeros(len(drug_ids), dtype=torch.bool)
            for d in tr_drugs:
                tmask[id2n[d]] = True
            _, mode, mol_key = VARIANT_LOOKUP[vname]
            _tgt = TGT_BANK.get("features", TGT)
            T = {"mol": MOLBANK[mol_key], "tgt": _tgt, "tcount": TCOUNT, "tgt_bin": TGT_BIN,
                 "train_mask": tmask.to(DEVICE)}
            set_seed(seed)
            m = TargetDDI(MOLBANK[mol_key].shape[1], _tgt.shape[1], C, CFG, mode=mode).to(DEVICE)
            n_par = sum(p.numel() for p in m.parameters() if p.requires_grad)
            t0 = time.time()
            m, _thr, hist = train_one(m, {"train": parts["train"], "valid": parts["valid"], "test": parts["test"]},
                                      T, id2n, C, ckpt=os.path.join(ZEN_OUT, f"ck_{tag}.pt"), log=False)
            train_s = time.time() - t0
            P = predict(m, parts["test"], T, id2n, C)
            y_cov = np.array([r[0] for r in parts["test"]["y"]], dtype=int)
            pred_cov = P.argmax(1)
            res = {f"covered_{k}": v for k, v in _zen_metrics(y_cov, pred_cov).items()}
            # score the FULL released test set; unfeaturisable triples count as errors
            y_all = np.array([r for _, _, r in parts["test_all"]], dtype=int)
            pred_all = (y_all + 1) % C
            it = iter(pred_cov.tolist())
            for i, (a, b, _) in enumerate(parts["test_all"]):
                if a in _known and b in _known:
                    pred_all[i] = next(it)
            res.update({f"all_{k}": v for k, v in _zen_metrics(y_all, pred_all).items()})
            res.update(dict(fold=fold, setting=fold.split("_")[0], variant=vname, seed=seed,
                            n_params=int(n_par), train_seconds=round(train_s, 1), n_epochs=len(hist),
                            n_test_triples=stats["test"]["triples"], n_test_covered=stats["test"]["covered"],
                            train_test_pair_overlap=int(overlap)))
            tmp = rp + ".tmp"; json.dump(res, open(tmp, "w"), indent=2); os.replace(tmp, rp)
            print(f"  [{tag}] F1={res['all_f1']*100:.1f} Acc={res['all_accuracy']*100:.1f} "
                  f"Kappa={res['all_kappa']*100:.1f} ({train_s/60:.1f} min)")
            del m; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return res

        _units = [(f, v, s) for f in ZEN_FOLDS for v in ZEN_VARIANTS for s in ZEN_SEEDS if v in VARIANT_LOOKUP]
        def _zen_missing():
            return [u for u in _units if not os.path.exists(os.path.join(ZEN_OUT, f"res_zen_{u[0]}_{u[1]}_s{u[2]}.json"))]
        _todo = _zen_missing()
        print(f"SECTION 14 EmerGNN folds: {len(_units)-len(_todo)}/{len(_units)} already complete, "
              f"{len(_todo)} to run now")
        _t0 = time.time(); _per = 0.0
        for _i, (fold, vname, seed) in enumerate(_todo, 1):
            if not budget_ok(_per, f"fold run {_i}/{len(_todo)} ({fold}/{vname})"):
                break
            try:
                run_zen_unit(fold, vname, seed)
            except Exception as e:
                import traceback; traceback.print_exc(); print(f"  FAILED {fold}/{vname}/s{seed}: {e}")
            progress(_i, len(_todo), _t0, "fold runs")
            _per = (time.time() - _t0) / _i / 3600
        print(f"SECTION 14 done: {len(_units)-len(_zen_missing())}/{len(_units)} complete overall")

        _res = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(ZEN_OUT, "res_zen_*.json")))]
        if _res:
            df = pd.DataFrame(_res)
            lab = {"full_target": "TARGET-DDI", "shared_target_graph": "TARGET-DDI-G", "ablation_no_target": "Mol-Full"}
            df["model"] = df["variant"].map(lab).fillna(df["variant"])
            rows = []
            for (setting, model), g in df.groupby(["setting", "model"]):
                row = {"Setting": OUR_NAME.get(setting, setting), "Model": model, "Folds": len(g),
                       "Coverage": f"{100*g['n_test_covered'].sum()/g['n_test_triples'].sum():.1f}%"}
                for met, name in [("all_f1", "F1"), ("all_accuracy", "Accuracy"), ("all_kappa", "Kappa")]:
                    v = g[met] * 100
                    row[name] = f"{v.mean():.1f} ± {(v.std(ddof=1) if len(v) > 1 else 0.0):.1f}"
                rows.append(row)
            summary = pd.DataFrame(rows).sort_values(["Setting", "Model"])
            summary.to_csv(os.path.join(ZEN_OUT, "emergnn_zenodo_summary.csv"), index=False)
            df.to_csv(os.path.join(ZEN_OUT, "emergnn_zenodo_per_fold.csv"), index=False)
            print("\n=== TARGET-DDI on EmerGNN's own folds (%, mean ± SD across folds) ===")
            print(summary.to_string(index=False))
