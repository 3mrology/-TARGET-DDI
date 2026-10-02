"""S1/S2/S3 entity-level splits, leakage audit, tie-correct metrics, paired bootstrap."""
def canonical(a,b): return (a,b) if str(a)<=str(b) else (b,a)
def build_pair_table(df,id1,id2,s1,s2,lab):
    ds={}
    for _,r in df[[id1,s1]].drop_duplicates(id1).iterrows(): ds[r[id1]]=r[s1]
    for _,r in df[[id2,s2]].drop_duplicates(id2).iterrows(): ds.setdefault(r[id2],r[s2])
    classes=sorted(pd.unique(df[lab]).tolist()); c2i={c:i for i,c in enumerate(classes)}
    pl={}
    for a,b,l in zip(df[id1].values,df[id2].values,df[lab].values):
        pl.setdefault(canonical(a,b),set()).add(c2i[l])
    rows=[{"u":k[0],"v":k[1],"y":sorted(v)} for k,v in pl.items()]
    return pd.DataFrame(rows), ds, len(classes)

def split_S1(pairs,frac,seed):
    rng=np.random.default_rng(seed); idx=rng.permutation(len(pairs)); n=len(pairs)
    a=int(frac[0]*n); b=a+int(frac[1]*n)
    return {"train":pairs.iloc[idx[:a]].reset_index(drop=True),
            "valid":pairs.iloc[idx[a:b]].reset_index(drop=True),
            "test":pairs.iloc[idx[b:]].reset_index(drop=True)},{"retention":1.0}

def split_entity(pairs,mode,frac,seed):
    rng=np.random.default_rng(seed); drugs=rng.permutation(pd.unique(pairs[["u","v"]].values.ravel()))
    n=len(drugs);a=int(frac[0]*n);b=a+int(frac[1]*n);tr_d,va_d,te_d=set(drugs[:a]),set(drugs[a:b]),set(drugs[b:])
    # train = pairs with BOTH endpoints assigned to train
    both=pairs.apply(lambda r: r["u"] in tr_d and r["v"] in tr_d, axis=1)
    train=pairs[both]
    # "seen" = drugs that actually appear in a TRAIN PAIR (what the model can learn),
    # which is exactly what audit() recomputes -> keeps split and audit consistent.
    seen=set(train["u"])|set(train["v"])
    need_unseen=1 if mode=="S2" else 2      # test/valid pair unseen-endpoint count
    def bucket(r):
        u,v=r["u"],r["v"]
        if u in seen and v in seen: return "train" if (u in tr_d and v in tr_d) else "drop"
        un=[d for d in (u,v) if d not in seen]
        if len(un)!=need_unseen: return "drop"
        in_va=all(d in va_d for d in un); in_te=all(d in te_d for d in un)
        if in_va and not in_te: return "valid"
        if in_te and not in_va: return "test"
        return "drop"
    bk=pairs.apply(bucket,axis=1)
    return {s:pairs[bk==s].reset_index(drop=True) for s in ["train","valid","test"]},\
           {"retention":float(bk.isin(["train","valid","test"]).mean())}

def make_split(pairs,mode,frac,seed):
    return split_S1(pairs,frac,seed) if mode=="S1" else split_entity(pairs,mode,frac,seed)

def audit(sp,mode,id2n=None,ei=None):
    ks={s:set(zip(sp[s]["u"],sp[s]["v"])) for s in ["train","valid","test"]}
    assert ks["train"].isdisjoint(ks["valid"]) and ks["train"].isdisjoint(ks["test"]) and ks["valid"].isdisjoint(ks["test"]),"pair overlap"
    tr_d=set(sp["train"]["u"])|set(sp["train"]["v"])
    if mode!="S1":
        want=1 if mode=="S2" else 2
        for u,v in ks["test"]: assert (u not in tr_d)+(v not in tr_d)==want,f"{mode} unseen-count"
    if ei is not None:
        es=set(map(tuple,ei.t().tolist())); tr_nodes={id2n[d] for d in tr_d}; leak=0
        for s in ["valid","test"]:
            for u,v in zip(sp[s]["u"],sp[s]["v"]):
                iu,iv=id2n[u],id2n[v]
                if (iu,iv) in es or (iv,iu) in es: leak+=1
        assert leak==0,f"{leak} held-out edges in graph"
        bad=sum(1 for s2,d in zip(ei[0].tolist(),ei[1].tolist()) if d in tr_nodes and s2 not in tr_nodes)
        assert bad==0,f"{bad} unseen->train edges"
    print(f"  [audit OK] {mode}: train={len(sp['train'])} valid={len(sp['valid'])} test={len(sp['test'])}")
    if len(sp["test"])<CFG.min_test_pairs: print(f"  WARN {len(sp['test'])} test pairs (<{CFG.min_test_pairs}); high variance")

def multihot(ys,C):
    Y=np.zeros((len(ys),C),np.float32)
    for i,l in enumerate(ys):
        for k in l: Y[i,k]=1.0
    return Y
def _auroc_cols(y,p):
    y=y.astype(bool);N,C=y.shape;out=np.full(C,np.nan)
    R=np.apply_along_axis(lambda c:rankdata(c,method="average"),0,p)
    npos=y.sum(0).astype(float);nneg=N-npos;ok=(npos>0)&(nneg>0);sr=(R*y).sum(0)
    with np.errstate(all="ignore"): out[ok]=((sr-npos*(npos+1)/2)/(npos*nneg))[ok]
    return out
def _ap_cols(y,p):
    from sklearn.metrics import average_precision_score
    N,C=y.shape;out=np.full(C,np.nan)
    for c in range(C):
        s=int(y[:,c].sum())
        if 0<s<N: out[c]=average_precision_score(y[:,c],p[:,c])
    return out
def _ap_micro(y,p):
    y=y.ravel().astype(np.float64);p=p.ravel()
    o=np.argsort(-p,kind="mergesort");ys=y[o];ps=p[o]
    tp=np.cumsum(ys);fp=np.cumsum(1-ys);P=tp[-1]
    if P==0: return float("nan")
    last=np.r_[np.diff(ps)!=0,True];tpg=tp[last];fpg=fp[last]
    prec=tpg/(tpg+fpg);rec=tpg/P;prev=np.r_[0.0,rec[:-1]]
    return float(np.sum((rec-prev)*prec))
def tune_thresholds(yv,pv):
    grid=np.linspace(0.05,0.95,19);y=yv.astype(bool);C=y.shape[1]
    best=np.zeros(C);thr=np.full(C,0.5,np.float32);has=y.any(0)
    for t in grid:
        pr=pv>=t;tp=(pr&y).sum(0,dtype=np.float64);fp=(pr&~y).sum(0,dtype=np.float64);fn=(~pr&y).sum(0,dtype=np.float64)
        den=2*tp+fp+fn;f1=np.where(den>0,2*tp/np.where(den>0,den,1),0.0)
        imp=has&(f1>best);best[imp]=f1[imp];thr[imp]=t
    return thr
def evaluate(y,p,thr):
    from sklearn.metrics import f1_score
    yh=(p>=thr[None,:]).astype(int);o={}
    o["micro_f1"]=float(f1_score(y,yh,average="micro",zero_division=0))
    o["macro_f1"]=float(f1_score(y,yh,average="macro",zero_division=0))
    au=_auroc_cols(y,p);o["auroc_per_type"]=float(np.nanmean(au)) if np.any(~np.isnan(au)) else float("nan")
    o["auprc_per_type"]=float(np.nanmean(_ap_cols(y,p)))
    o["micro_auprc"]=_ap_micro(y,p)
    prev=y.mean();o["pr_lift"]=float(o["micro_auprc"]/prev) if prev>0 else float("nan")
    for k in (1,3,5):
        ok=np.argsort(-p,axis=1)[:,:k];hit=tot=0
        for i in range(len(y)):
            tr=set(np.where(y[i]==1)[0])
            if not tr: continue
            tot+=1;hit+=1 if tr&set(ok[i].tolist()) else 0
        o[f"top{k}"]=hit/max(tot,1)
    return o
def paired_bootstrap(y,pf,pb,nb,seed=0):
    rng=np.random.default_rng(seed);N=len(y);d=np.empty(nb)
    print(f"    [bootstrap] N={N} pairs x C={y.shape[1]} classes, nb={nb} resamples -- timing first 10 iters...")
    t0=time.time();report_every=max(1,nb//20)
    for b in range(nb):
        idx=rng.integers(0,N,N);d[b]=_ap_micro(y[idx],pf[idx])-_ap_micro(y[idx],pb[idx])
        if b==9:
            per_iter=(time.time()-t0)/10
            print(f"    [bootstrap] ~{per_iter*1000:.0f}ms/iter -> est. total {per_iter*nb/60:.1f} min for nb={nb}")
        if (b+1)%report_every==0 or b+1==nb:
            elapsed=time.time()-t0;frac=(b+1)/nb;eta=elapsed/frac-elapsed
            print(f"    [bootstrap] {b+1}/{nb} ({frac*100:.0f}%) elapsed={elapsed:.0f}s  eta={eta:.0f}s")
    print(f"    [bootstrap] done in {time.time()-t0:.0f}s")
    lo,hi=np.percentile(d,[2.5,97.5]);return float(_ap_micro(y,pf)-_ap_micro(y,pb)),float(lo),float(hi)
