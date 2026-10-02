"""Mechanism attribution on the seed-0 checkpoints."""
import re as _re

CYP_UNIPROT = {
 "CYP1A2":"P05177","CYP2B6":"P20813","CYP2C8":"P10632","CYP2C9":"P11712",
 "CYP2C19":"P33261","CYP2D6":"P10635","CYP2E1":"P05181","CYP3A4":"P08684","CYP3A5":"P20815",
 "CYP2A6":"P11509","CYP2J2":"P51589","CYP1A1":"P04798","CYP1B1":"Q16678",
}
def mechanism_targets_from_text(txt):
    if not isinstance(txt,str): return set()
    acc=set()
    for gene in _re.findall(r"CYP\s?\d+[A-Z]\d+", txt.upper().replace(" ","")):
        g=gene.replace(" ","")
        if g in CYP_UNIPROT: acc.add(CYP_UNIPROT[g])
    return acc

def load_best_model(tag, dmol, dtgt, mode="graph"):
    '''FIX: read ck_{tag}.pt (the only file train_one ever writes) and load the
    BEST state dict cached at checkpoint["best_state"]["model"] -- not the top-level
    "model" key, which is just whatever epoch was running when training stopped.'''
    ck_path = f"{CFG.workdir}/ck_{tag}.pt"
    if not os.path.exists(ck_path):
        return None
    state = torch.load(ck_path, map_location=DEVICE, weights_only=False)
    best_state = state.get("best_state")
    if best_state is None or "model" not in best_state:
        print(f"  [attribution] {tag}: checkpoint has no best_state yet (still early in training?)")
        return None
    model = TargetDDI(dmol, dtgt, C, CFG, mode=mode).to(DEVICE)
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state["model"].items()})
    model.eval()
    return model

def run_attribution_validation(split="S3", seed=0, max_pairs=400):
    tag = f"{CFG.dataset}_{CFG.feature_version}_{split}_shared_target_graph_s{seed}"
    model = load_best_model(tag, MOLBANK["full"].shape[1], Xtgt.shape[1], mode="graph")
    if model is None:
        print(f"  [attribution] no usable checkpoint for {split} seed{seed}; run the grid first")
        return
    try:
        raw=DDI(name=CFG.dataset).get_data()
        desc_col=[c for c in raw.columns if raw[c].dtype==object and raw[c].astype(str).str.contains("CYP",case=False,na=False).any()]
    except Exception as e:
        print("  [attribution] could not load DDI descriptions:",e); return
    if not desc_col:
        print("  [attribution] no mechanism/description text with CYP mentions in this dataset build; skipping")
        return
    dc=desc_col[0]; cols={c.lower():c for c in raw.columns}
    id1=cols.get("drug1_id","Drug1_ID"); id2=cols.get("drug2_id","Drug2_ID")
    pair_mech={}
    for a,b,t in zip(raw[id1],raw[id2],raw[dc]):
        m=mechanism_targets_from_text(t)
        if m: pair_mech[(str(a),str(b))]=m; pair_mech[(str(b),str(a))]=m

    sp,_=make_split(pairs,split,CFG.frac,seed)
    tr_drugs=set(sp["train"]["u"])|set(sp["train"]["v"])
    tmask=torch.zeros(len(drug_ids),dtype=torch.bool)
    for d in tr_drugs: tmask[id2n[d]]=True
    T={"mol":MOLBANK["full"],"tgt":TGT,"tcount":TCOUNT,"tgt_bin":TGT_BIN,"train_mask":tmask.to(DEVICE)}

    prot_ids=_prot_ids
    hits=tot=0
    for u,v in zip(sp["test"]["u"],sp["test"]["v"]):
        mech=pair_mech.get((str(u),str(v)))
        if not mech: continue
        i,j=id2n[u],id2n[v]
        ranked=model.attribute_pair(T,i,j,topk=3)
        if not ranked: continue
        top_prots={prot_ids[t] for t,_ in ranked}
        tot+=1; hits+= 1 if (top_prots & mech) else 0
        if tot>=max_pairs: break
    if tot==0:
        print(f"  [attribution/{split}] no test pairs with a parseable CYP mechanism AND shared targets"); return
    print(f"  [MECHANISM ATTRIBUTION / {split}] top-3 attribution matches known CYP mechanism "
          f"in {hits}/{tot} = {hits/tot:.3f} of pairs")

for _sp in [x for x in ("S2","S3") if x in getattr(CFG,"splits",[])]:
    try: run_attribution_validation(_sp)
    except Exception as e:
        import traceback; traceback.print_exc(); print("  attribution error:",e)
