"""Feature construction and the S1-S3 paper grid (Tables 1, 2, 4, 5 and the ladder in Table 8)."""
pairs,ds,C=load_dataset(CFG.dataset)
drug_ids=sorted(ds.keys(),key=str);id2n={d:i for i,d in enumerate(drug_ids)}
print(f"{CFG.dataset}: {len(pairs)} pairs | {len(drug_ids)} drugs | {C} classes")

# ---- TARGET DATA: categorized (target/enzyme/transporter/carrier) + coverage probe (hard gate) ----
print("\n=== TARGET COVERAGE PROBE ===")
cats = fetch_drug_targets_by_category(drug_ids) if CFG.enable_categorized_targets else {"target":fetch_drug_targets(drug_ids)}
d2p_union = {}
for cat,dd in cats.items():
    for d,ps in dd.items(): d2p_union.setdefault(d,set()).update(ps)
rep=coverage_report(drug_ids,d2p_union)
print(f"  coverage={rep['coverage']*100:.1f}%  covered={rep['covered']}/{rep['n_drugs']}  "
      f"median/drug={rep['median']:.0f}  distinct proteins={rep['distinct']}")
for cat,dd in cats.items():
    cr=coverage_report(drug_ids,dd)
    print(f"    [{cat}] coverage={cr['coverage']*100:.1f}%  distinct={cr['distinct']}")
TARGETS_OK = rep["coverage"]>=CFG.coverage_gate and rep["distinct"]>0
if not TARGETS_OK:
    print(f"  GATE FAILED: coverage < {CFG.coverage_gate*100:.0f}% or no targets fetched.")
else:
    print("  GATE PASSED -> target branch is viable.")

tgt_blocks=[]; tgt_names=[]
if TARGETS_OK:
    for cat in ["target","enzyme","transporter","carrier"]:
        Xc,keep=build_target_features(drug_ids,cats.get(cat,{}),CFG.target_min_freq,CFG.target_max)
        if Xc.shape[1]:
            tgt_blocks.append(Xc); tgt_names.append(f"{cat}({Xc.shape[1]})")
    print("  protein blocks: "+(", ".join(tgt_names) if tgt_names else "none"))

if TARGETS_OK and CFG.enable_go_pfam:
    all_prots=set().union(*d2p_union.values()) if d2p_union else set()
    prot2gopfam=fetch_go_pfam(all_prots)
    Xgo,Xpf,keep_go,keep_pf=build_go_pfam_features(drug_ids,d2p_union,prot2gopfam,CFG.go_pfam_min_freq,CFG.go_pfam_max)
    if Xgo.shape[1]: tgt_blocks.append(Xgo); tgt_names.append(f"GO({Xgo.shape[1]})")
    if Xpf.shape[1]: tgt_blocks.append(Xpf); tgt_names.append(f"Pfam({Xpf.shape[1]})")
    print(f"  +GO/Pfam blocks: GO={Xgo.shape[1]} Pfam={Xpf.shape[1]}")

Xtgt = np.concatenate(tgt_blocks,1).astype(np.float32) if tgt_blocks else np.zeros((len(drug_ids),0),np.float32)
print(f"  target-branch input width: {Xtgt.shape[1]}  [{', '.join(tgt_names) if tgt_names else 'none'}]")

Xmol_classic = molecular_features(drug_ids,ds)
mol_blocks=[Xmol_classic]; mol_names=[f"chemberta+morgan({Xmol_classic.shape[1]})"]

if CFG.enable_atc:
    print("\n=== ATC CLASSIFICATION ===")
    d2atc=fetch_atc(drug_ids)
    Xatc,keep_full,keep_l1=build_atc_features(drug_ids,d2atc,CFG.atc_min_freq)
    if Xatc.shape[1]:
        mol_blocks.append(Xatc); mol_names.append(f"ATC({Xatc.shape[1]}: {len(keep_full)} codes + {len(keep_l1)} L1 groups)")

if CFG.enable_physchem:
    sm=[ds[d] for d in drug_ids]
    Xpc=physchem_features(sm)
    mol_blocks.append(Xpc); mol_names.append(f"physchem({Xpc.shape[1]})")

if CFG.enable_cyp_roles:
    Xcyp,combo=fetch_cyp_roles(drug_ids)
    if Xcyp.shape[1]: mol_blocks.append(Xcyp); mol_names.append(f"CYP-roles({Xcyp.shape[1]})")

Xmol_full = np.concatenate(mol_blocks,1).astype(np.float32)
print(f"  molecular-branch input width: classic={Xmol_classic.shape[1]}  full={Xmol_full.shape[1]}  [{', '.join(mol_names)}]")

tcount = np.log1p(Xtgt.sum(1,keepdims=True)).astype(np.float32) if Xtgt.shape[1] else np.zeros((len(drug_ids),1),np.float32)
Xesm=np.zeros((len(drug_ids),0),np.float32); _esm_ok=False
MOLBANK={"classic":torch.tensor(Xmol_classic).to(DEVICE),"full":torch.tensor(Xmol_full).to(DEVICE)}
if TARGETS_OK and d2p_union:
    _prot_ids=sorted(set().union(*d2p_union.values()))
    _pidx={p:i for i,p in enumerate(_prot_ids)}
    _B=np.zeros((len(drug_ids),len(_prot_ids)),np.float32)
    for _i,_d in enumerate(drug_ids):
        for _p in d2p_union.get(_d,()): _B[_i,_pidx[_p]]=1.0
    TGT_BIN=torch.tensor(_B).to(DEVICE)
    SHARED=(_B@_B.T)
    if getattr(CFG,"use_esm",True):
        _esm=fetch_esm_embeddings(_prot_ids)
        Xesm,_esm_ok=build_esm_drug_features(drug_ids,d2p_union,_esm)
        print(f"   ESM-2 drug features: {Xesm.shape} (ok={_esm_ok})")
else:
    TGT_BIN=torch.zeros(len(drug_ids),0); SHARED=np.zeros((len(drug_ids),len(drug_ids)),np.float32)
TGT=torch.tensor(Xtgt).to(DEVICE); TCOUNT=torch.tensor(tcount).to(DEVICE)
_Xesm_t=torch.tensor(Xesm).to(DEVICE) if _esm_ok else torch.zeros(len(drug_ids),0).to(DEVICE)
TGT_BANK={"features":TGT,"esm":_Xesm_t,"both":(torch.cat([TGT,_Xesm_t],1) if _esm_ok else TGT)}
print(f"  features: molecular_classic={Xmol_classic.shape} molecular_full={Xmol_full.shape} target={Xtgt.shape}")

VARIANTS=[("full_target","gated","full"),
          ("shared_target_graph","graph","full"),
          ("ablation_no_target","molecular","full")]
if getattr(CFG,"run_target_only",False):
    VARIANTS.append(("target_only","target_only","full"))
if getattr(CFG,"run_esm_ablation",True) and _esm_ok:
    VARIANTS += [("graph_esm","graph","full"), ("graph_both","graph","full")]
if CFG.run_feature_ablation_ladder:
    VARIANTS.append(("mol_only_classic",False,"classic"))
# --- your pasted SIDER/dex_ddi variants (Section 3b) should already be appended to
# VARIANTS by the time this cell runs; nothing further needed here ---

def run_unit(split,vname,mode,mol_key,seed):
    tag=f"{CFG.dataset}_{CFG.feature_version}_{split}_{vname}_s{seed}";rp=f"{CFG.workdir}/res_{tag}.json"
    if os.path.exists(rp): return json.load(open(rp))
    sp,info=make_split(pairs,split,CFG.frac,seed)
    if len(sp["test"])==0 or len(sp["valid"])==0: print(f"  {split}: empty split"); return None
    audit(sp,split,id2n,None)
    tr_drugs=set(sp["train"]["u"])|set(sp["train"]["v"])
    tmask=torch.zeros(len(drug_ids),dtype=torch.bool)
    for d in tr_drugs: tmask[id2n[d]]=True
    _tbank={"graph_esm":"esm","graph_both":"both"}.get(vname,"features")
    _tgt=TGT_BANK.get(_tbank,TGT)
    T={"mol":MOLBANK[mol_key],"tgt":_tgt,"tcount":TCOUNT,"tgt_bin":TGT_BIN,"train_mask":tmask.to(DEVICE)}
    set_seed(seed);m=TargetDDI(MOLBANK[mol_key].shape[1],_tgt.shape[1],C,CFG,mode=mode).to(DEVICE)
    m,thr,hist=train_one(m,sp,T,id2n,C,ckpt=f"{CFG.workdir}/ck_{tag}.pt",log=True)
    pd.DataFrame(hist).to_csv(f"{CFG.workdir}/hist_{tag}.csv",index=False)
    yte=multihot(sp["test"]["y"],C);pt=predict(m,sp["test"],T,id2n,C)
    shr=np.array([SHARED[id2n[u],id2n[v]] for u,v in zip(sp["test"]["u"],sp["test"]["v"])],np.int32)
    res=evaluate(yte,pt,thr);np.savez_compressed(f"{CFG.workdir}/pred_{tag}.npz",y=yte,p=pt,shared=shr)
    res.update(dict(dataset=CFG.dataset,split=split,variant=vname,seed=seed,mol_key=mol_key,mode=mode,
                    coverage=rep["coverage"],n_test=len(sp["test"])))
    json.dump(res,open(rp,"w"),indent=2)
    print(f"  [{tag}] microAP={res['micro_auprc']:.4f} macroF1={res['macro_f1']:.4f} PR-lift={res['pr_lift']:.1f}x")
    del m;gc.collect();torch.cuda.empty_cache();return res

# ---------------- priority-ordered, missing-only unit list ----------------
CORE_NAMES = {"full_target", "shared_target_graph", "ablation_no_target"}
OPTIONAL_NAMES = {v[0] for v in VARIANTS if v[0] not in CORE_NAMES} | set(EXTRA_VARIANT_NAMES)
VARIANT_LOOKUP = {v[0]: v for v in VARIANTS}

GRID_TIME_BUDGET_HOURS = 0 if globals().get("SKIP_MAIN_GRID", False) else None   # 0 = skip the paper grid (SKIP_MAIN_GRID); None = no limit
_grid_start = time.time()

def _pending_units(split, names, seeds_fn):
    out = []
    for name in names:
        if name not in VARIANT_LOOKUP:
            print(f"  [skip] variant '{name}' not defined in VARIANTS -- paste its setup in Section 3b")
            continue
        _, mode, mol_key = VARIANT_LOOKUP[name]
        for seed in seeds_fn(name):
            tag = f"{CFG.dataset}_{CFG.feature_version}_{split}_{name}_s{seed}"
            rp = f"{CFG.workdir}/res_{tag}.json"
            out.append((split, name, mode, mol_key, seed, os.path.exists(rp)))
    return out

def _headline(name): return CFG.headline_seeds
def _ablation(name): return CFG.ablation_seeds

TIERS = [
    ("S3 core",     _pending_units("S3", sorted(CORE_NAMES), _headline)),
    ("S3 optional", _pending_units("S3", sorted(OPTIONAL_NAMES), _ablation)),
    ("S1 core",     _pending_units("S1", sorted(CORE_NAMES), _headline)),
    ("S1 optional", _pending_units("S1", sorted(OPTIONAL_NAMES), _ablation)),
]

print("\n" + "=" * 70)
for tier_name, units in TIERS:
    done = sum(1 for u in units if u[5]); todo = len(units) - done
    print(f"[{tier_name}] {done} already complete, {todo} pending")
print("=" * 70)

for tier_name, units in TIERS:
    print(f"\n--- TIER: {tier_name} ---")
    for split, vname, mode, mol_key, seed, already_done in units:
        if already_done:
            continue    # run_unit would also just short-circuit on this, skip the log noise
        if GRID_TIME_BUDGET_HOURS is not None and (time.time() - _grid_start) / 3600 >= GRID_TIME_BUDGET_HOURS:
            print(f"  time budget ({GRID_TIME_BUDGET_HOURS}h) reached -- stopping before starting "
                  f"{split}/{vname}/s{seed}. Safe to stop here; re-run this cell next session to resume.")
            break
        try:
            run_unit(split, vname, mode, mol_key, seed)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"  FAILED {split}/{vname}/s{seed}: {e}")
    else:
        continue
    break   # time budget hit inside the inner loop -> stop scheduling further tiers too

# ---------------------------------------------------------------------------
# Seed-0 gap filler: Sections 12, 16 and 17 read pred_*.npz and ck_*.pt for the nine
# seed-0 core runs. When those files are not attached, regenerate just the missing ones
# here rather than rerunning the whole grid, so a single session can finish everything.
if globals().get("FILL_SEED0_GAPS", False):
    _core = ["ablation_no_target", "full_target", "shared_target_graph"]
    _need = []
    for _sp in ["S1", "S2", "S3"]:
        for _v in _core:
            if _v not in VARIANT_LOOKUP:
                continue
            _tag = f"{CFG.dataset}_{CFG.feature_version}_{_sp}_{_v}_s0"
            _pred = f"{CFG.workdir}/pred_{_tag}.npz"
            _ck = f"{CFG.workdir}/ck_{_tag}.pt"
            if not (os.path.exists(_pred) and os.path.exists(_ck)):
                _need.append((_sp, _v, _tag, _pred, _ck))
    print("\n" + "=" * 70)
    print(f"SEED-0 GAP FILLER: {9 - len(_need)}/9 runs already have predictions, {len(_need)} to regenerate")
    if _need:
        print("  (needed by Sections 12, 16 and 17; each run is trained once and then reused)")
    _t0 = time.time()
    for _i, (_sp, _v, _tag, _pred, _ck) in enumerate(_need, 1):
        # a result file without its predictions would make run_unit short-circuit, so clear it
        _rp = f"{CFG.workdir}/res_{_tag}.json"
        if os.path.exists(_rp) and not os.path.exists(_pred):
            os.remove(_rp)
        try:
            _, _mode, _mol = VARIANT_LOOKUP[_v]
            run_unit(_sp, _v, _mode, _mol, 0)
            ok = os.path.exists(_pred) and os.path.exists(_ck)
            print(f"  [{_i}/{len(_need)}] {_sp}/{_v}: {'predictions saved' if ok else 'ran but files missing'}"
                  f" | {(time.time()-_t0)/60:.0f} min elapsed", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"  FAILED {_sp}/{_v}/s0: {e}")
    _left = [t for _, _, t, pr, ck in _need if not (os.path.exists(pr) and os.path.exists(ck))]
    print(f"SEED-0 GAP FILLER done: {9 - len(_left)}/9 runs now have predictions")
    print("=" * 70)

print("\nMissing-only grid pass complete (or time budget reached).")
