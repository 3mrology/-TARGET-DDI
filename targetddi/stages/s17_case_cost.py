"""Case study of pairs corrected by the target branch (Table 12) and training cost."""
# Sections 16-17
if not RUN_CASE_COST:
    print('RUN_CASE_COST=False -> skipped')
else:
    # ===== Section 16. Case study: pairs the target branch corrects =====
    # Finds seed-0 S3 test pairs that the molecular-only model ranks wrongly and TARGET-DDI ranks
    # correctly, then reports the proteins the two drugs share and, for the graph model, which of
    # those proteins the prediction actually depends on (measured by ablating each shared protein
    # and recording the change in the pair's logits). Protein names come from UniProt.
    import os, json, glob
    import numpy as np, pandas as pd

    CASE_OUT = os.path.join(CFG.workdir, "case_study"); os.makedirs(CASE_OUT, exist_ok=True)
    CASE_SPLIT, CASE_SEED, CASE_TOP = "S3", 0, 12


    def _case_tag(v):
        return f"{CFG.dataset}_{CFG.feature_version}_{CASE_SPLIT}_{v}_s{CASE_SEED}"


    def _uniprot_names(accs):
        """Accession -> protein name and gene symbol (best effort; blank on failure)."""
        out = {}
        accs = [a for a in accs if a]
        for i0 in range(0, len(accs), 20):
            part = accs[i0:i0 + 20]
            try:
                r = requests.get("https://rest.uniprot.org/uniprotkb/search",
                                 params={"query": " OR ".join(f"accession:{a}" for a in part),
                                         "fields": "accession,protein_name,gene_primary",
                                         "format": "tsv", "size": 50},
                                 headers=_UNIPROT_HEADERS, timeout=60)
                if r.status_code != 200:
                    continue
                for ln in r.text.strip().split("\n")[1:]:
                    c = ln.split("\t")
                    if len(c) >= 3:
                        out[c[0]] = {"name": c[1], "gene": c[2]}
            except Exception:
                continue
        return out


    _pm = os.path.join(CFG.workdir, f"pred_{_case_tag('ablation_no_target')}.npz")
    _pt = os.path.join(CFG.workdir, f"pred_{_case_tag('full_target')}.npz")
    if not (os.path.exists(_pm) and os.path.exists(_pt)):
        print("Case study needs the seed-0 S3 predictions of Mol-Full and TARGET-DDI; run the grid first.")
    else:
        dm, dt = np.load(_pm), np.load(_pt)
        ym, pm, pt = dm["y"], dm["p"], dt["p"]
        assert ym.shape == pt.shape, "prediction files disagree on shape"
        sp, _ = make_split(pairs, CASE_SPLIT, CFG.frac, CASE_SEED)
        test = sp["test"].reset_index(drop=True)
        assert len(test) == len(ym), "split and saved predictions are out of sync"
        true_top = [set(np.where(ym[i] == 1)[0]) for i in range(len(ym))]
        mol_ok = np.array([pm[i].argmax() in true_top[i] if true_top[i] else False for i in range(len(ym))])
        tgt_ok = np.array([pt[i].argmax() in true_top[i] if true_top[i] else False for i in range(len(ym))])
        fixed = np.where((~mol_ok) & tgt_ok)[0]
        broke = np.where(mol_ok & (~tgt_ok))[0]
        print(f"S3 seed 0: {len(fixed)} pairs corrected by the target branch, {len(broke)} lost "
              f"({len(ym)} test pairs)")

        prot_ids = sorted(set().union(*d2p_union.values())) if d2p_union else []
        margin = np.array([pt[i].max() - pm[i].max() for i in fixed])
        order = fixed[np.argsort(-margin)]
        rows = []
        for i in order[:CASE_TOP]:
            u, v = test.loc[i, "u"], test.loc[i, "v"]
            shared = sorted(set(d2p_union.get(u, ())) & set(d2p_union.get(v, ())))
            rows.append({"drug_1": u, "drug_2": v,
                         "true_type": sorted(true_top[i])[0] if true_top[i] else None,
                         "mol_only_top": int(pm[i].argmax()), "target_top": int(pt[i].argmax()),
                         "mol_only_score_true": float(pm[i][sorted(true_top[i])[0]]) if true_top[i] else None,
                         "target_score_true": float(pt[i][sorted(true_top[i])[0]]) if true_top[i] else None,
                         "n_shared_proteins": len(shared), "shared_proteins": ";".join(shared[:8])})
        case = pd.DataFrame(rows)
        names = _uniprot_names(sorted({a for r in rows for a in r["shared_proteins"].split(";") if a})) if rows else {}
        case["shared_genes"] = [";".join(names.get(a, {}).get("gene", "") for a in r.split(";") if a)
                                for r in case.get("shared_proteins", [])]
        case.to_csv(os.path.join(CASE_OUT, "corrected_pairs.csv"), index=False)
        print("\nPairs corrected by the target branch (largest score gain first):")
        print(case[["drug_1", "drug_2", "true_type", "mol_only_top", "target_top",
                    "n_shared_proteins", "shared_genes"]].head(CASE_TOP).to_string(index=False))

        with_shared = sum(1 for i in fixed if set(d2p_union.get(test.loc[i, "u"], ())) &
                          set(d2p_union.get(test.loc[i, "v"], ())))
        print(f"\n{with_shared}/{len(fixed)} corrected pairs share at least one protein "
              f"({100*with_shared/max(len(fixed),1):.0f}%)")

        # which shared proteins the graph model's prediction actually depends on
        _gtag = _case_tag("shared_target_graph")
        _gck = os.path.join(CFG.workdir, f"ck_{_gtag}.pt")
        if os.path.exists(_gck) and prot_ids:
            try:
                tr_drugs = set(sp["train"]["u"]) | set(sp["train"]["v"])
                tmask = torch.zeros(len(drug_ids), dtype=torch.bool)
                for d in tr_drugs:
                    tmask[id2n[d]] = True
                T = {"mol": MOLBANK["full"], "tgt": TGT_BANK.get("features", TGT), "tcount": TCOUNT,
                     "tgt_bin": TGT_BIN, "train_mask": tmask.to(DEVICE)}
                gm = TargetDDI(MOLBANK["full"].shape[1], TGT_BANK.get("features", TGT).shape[1], C, CFG,
                               mode="graph").to(DEVICE)
                st = torch.load(_gck, map_location=DEVICE, weights_only=False)
                gm.load_state_dict({k: v.to(DEVICE) for k, v in st["best_state"]["model"].items()})
                gm.eval()
                att = []
                for i in order[:CASE_TOP]:
                    u, v = test.loc[i, "u"], test.loc[i, "v"]
                    for t_idx, eff in gm.attribute_pair(T, id2n[u], id2n[v], topk=3):
                        acc = prot_ids[t_idx] if t_idx < len(prot_ids) else str(t_idx)
                        att.append({"drug_1": u, "drug_2": v, "protein": acc,
                                    "gene": names.get(acc, {}).get("gene", ""),
                                    "name": names.get(acc, {}).get("name", ""), "effect": round(eff, 4)})
                if att:
                    a = pd.DataFrame(att)
                    a.to_csv(os.path.join(CASE_OUT, "shared_protein_attribution.csv"), index=False)
                    print("\nShared proteins the graph model's prediction depends on (ablation effect):")
                    print(a.head(15).to_string(index=False))
            except Exception as e:
                print("   attribution skipped:", e)

    # ===== Section 17. Training cost and model size =====
    # Standard in this venue and cheap: parameters and wall-clock training time per configuration,
    # read from the run records already on disk.
    COST_OUT = os.path.join(CFG.workdir, "cost"); os.makedirs(COST_OUT, exist_ok=True)


    def _params_for(vname):
        _, mode, mol_key = VARIANT_LOOKUP.get(vname, (None, "gated", "full"))
        _tb = {"graph_esm": "esm", "graph_both": "both"}.get(vname, "features")
        _tgt = TGT_BANK.get(_tb, TGT)
        m = TargetDDI(MOLBANK[mol_key].shape[1], _tgt.shape[1], C, CFG, mode=mode)
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        del m
        return n


    cost_rows = []
    for rp in sorted(glob.glob(f"{CFG.workdir}/res_{CFG.dataset}_{CFG.feature_version}_*.json")):
        r = json.load(open(rp))
        v, split = r.get("variant"), r.get("split")
        if v not in VARIANT_LOOKUP:
            continue
        hp = rp.replace("/res_", "/hist_").replace(".json", ".csv")
        ep = len(pd.read_csv(hp)) if os.path.exists(hp) else None
        cost_rows.append({"split": split, "variant": v, "seed": r.get("seed"), "epochs": ep})
    if cost_rows:
        cdf = pd.DataFrame(cost_rows)
        lab = {"full_target": "TARGET-DDI", "shared_target_graph": "TARGET-DDI-G",
               "ablation_no_target": "Mol-Full", "mol_only_classic": "Mol-Classic",
               "graph_esm": "ESM-G", "graph_both": "Both-G"}
        out = []
        for v, g in cdf.groupby("variant"):
            try:
                n_par = _params_for(v)
            except Exception:
                n_par = None
            out.append({"Model": lab.get(v, v), "Parameters": f"{n_par:,}" if n_par else "n/a",
                        "Runs": len(g), "Epochs (mean)": round(g["epochs"].dropna().mean(), 1)
                        if g["epochs"].notna().any() else None})
        cost = pd.DataFrame(out).sort_values("Model")
        cost.to_csv(os.path.join(COST_OUT, "cost_table.csv"), index=False)
        print("\n=== Model size and training length ===")
        print(cost.to_string(index=False))
        print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}; "
              f"batch {CFG.batch_size}, max {CFG.max_epochs} epochs, patience {CFG.patience}.")
        print("Per-epoch wall-clock times are in the hist_*.csv files; the gate ablation and the")
        print("external benchmarks additionally record train_seconds per run in their res_*.json.")
