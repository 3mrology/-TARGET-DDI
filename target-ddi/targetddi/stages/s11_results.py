"""Aggregates results, paired-bootstrap CIs (Table 9) and the shared-protein stratification (Table 10)."""
rows=[json.load(open(p)) for p in sorted(glob.glob(f"{CFG.workdir}/res_*.json"))]
if not rows: print("no results")
else:
    R=pd.DataFrame(rows)
    agg=R.groupby(["split","variant"])[["micro_auprc","auprc_per_type","macro_f1","auroc_per_type","pr_lift","top1","top5"]].agg(["mean","std","count"]).reset_index()
    agg.columns=["_".join([c for c in col if c]).rstrip("_") for col in agg.columns]
    agg.to_csv(f"{CFG.workdir}/all_results.csv",index=False);R.to_csv(f"{CFG.workdir}/all_runs_raw.csv",index=False)
    print("Saved all_results.csv\n")
    for split in [s for s in CFG.splits if s in set(R.split)]:
        print(f"--- {split} ---")
        for v in R[R.split==split].variant.unique():
            q=agg[(agg.split==split)&(agg.variant==v)]
            if len(q):
                m=q["micro_auprc_mean"].iloc[0];s=q["micro_auprc_std"].iloc[0] if q["micro_auprc_count"].iloc[0]>1 else float("nan")
                print(f"  {v:20s} microAP={m:.4f}"+(f"±{s:.4f}" if s==s else "")+f"  macroF1={q['macro_f1_mean'].iloc[0]:.4f}")
        # bootstrap CI: full_target vs no_target, seed 0
        ff=f"{CFG.workdir}/pred_{CFG.dataset}_{CFG.feature_version}_{split}_full_target_s0.npz"
        bb=f"{CFG.workdir}/pred_{CFG.dataset}_{CFG.feature_version}_{split}_ablation_no_target_s0.npz"
        if os.path.exists(ff) and os.path.exists(bb):
            try:
                print(f"  computing paired-bootstrap CI for {split} (this can take a while, watch for [bootstrap] progress lines below)...")
                F_=np.load(ff);B_=np.load(bb)
                pt,lo,hi=paired_bootstrap(F_["y"],F_["p"],B_["p"],CFG.n_bootstrap)
                verdict="TARGETS HELP (CI>0)" if lo>0 else ("TARGETS HURT (CI<0)" if hi<0 else "INCONCLUSIVE (CI spans 0)")
                print(f"  Δ microAP (target − no-target) = {pt:+.4f}  95% CI [{lo:+.4f},{hi:+.4f}]  -> {verdict}")
            except Exception as e:
                print("  (bootstrap skipped:",e,")")
        print()

print("\n"+"="*60+"\nRUN COMPLETE — results in "+CFG.workdir+"/all_results.csv\n"+"="*60)

# ================= TARGET-OVERLAP STRATIFICATION (the mechanism figure) =================
# On S3 (both drugs unseen), does the shared-target GRAPH win *more* as the pair shares
# more targets? That is the causal story: performance tracks shared-target connectivity.
def _micro_ap(y,p):
    from sklearn.metrics import average_precision_score
    y=y.ravel(); p=p.ravel()
    return float(average_precision_score(y,p)) if y.sum()>0 else float("nan")
def stratify(split="S3"):
    import glob as _g
    fg=sorted(_g.glob(f"{CFG.workdir}/pred_{CFG.dataset}_{CFG.feature_version}_{split}_shared_target_graph_s0.npz"))
    fb=sorted(_g.glob(f"{CFG.workdir}/pred_{CFG.dataset}_{CFG.feature_version}_{split}_ablation_no_target_s0.npz"))
    if not fg or not fb: print(f"  [stratify {split}] need graph + no_target seed-0 preds"); return
    G=np.load(fg[0]); B=np.load(fb[0])
    if "shared" not in G.files: print("  [stratify] no shared-count saved (re-run needed)"); return
    shr=G["shared"]
    buckets=[("0",shr==0),("1",shr==1),("2-3",(shr>=2)&(shr<=3)),("4+",shr>=4)]
    print(f"\n  target-overlap stratification on {split} (micro-AP):")
    print(f"    {'shared':8s}{'n':>8s}{'no_target':>12s}{'graph':>10s}{'Δ':>9s}")
    for name,mask in buckets:
        if mask.sum()==0: continue
        ap_b=_micro_ap(B["y"][mask],B["p"][mask]); ap_g=_micro_ap(G["y"][mask],G["p"][mask])
        print(f"    {name:8s}{int(mask.sum()):>8d}{ap_b:>12.4f}{ap_g:>10.4f}{ap_g-ap_b:>+9.4f}")
for _sp in [x for x in ("S2","S3") if x in getattr(CFG,"splits",[])]:
    try: stratify(_sp)
    except Exception as e: print("  stratify error:",e)
