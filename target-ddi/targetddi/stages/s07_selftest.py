"""Synthetic cold-start self-test; stops the run if the target branch cannot beat the molecular-only model."""
def selftest():
    # target-driven synthetic cold-start world; target model must beat mol-only on S3
    from sklearn.metrics import average_precision_score
    rng=np.random.default_rng(0);Nd,Ntg,C=250,80,12
    tg=(rng.random((Nd,Ntg))<0.06).astype(np.float32);mo=rng.standard_normal((Nd,48)).astype(np.float32)
    W=rng.standard_normal((Ntg,C)).astype(np.float32)
    P=[(i,j) for i in range(Nd) for j in range(i+1,Nd) if rng.random()<0.05]
    def lab(i,j):
        u=np.clip(tg[i]+tg[j],0,1);p=1/(1+np.exp(-(u@W)));y=(p>0.7).astype(np.float32)
        if y.sum()==0:y[p.argmax()]=1
        return y
    Y={p:lab(*p) for p in P};perm=rng.permutation(Nd);tr_d=set(perm[:170])
    tr=[p for p in P if p[0] in tr_d and p[1] in tr_d];te=[p for p in P if p[0] not in tr_d and p[1] not in tr_d]
    id2n={i:i for i in range(Nd)}
    T={"mol":torch.tensor(mo).to(DEVICE),"tgt":torch.tensor(tg).to(DEVICE),
       "tcount":torch.tensor(np.log1p(tg.sum(1,keepdims=True)).astype(np.float32)).to(DEVICE)}
    import pandas as pd
    mk=lambda L: pd.DataFrame([{"u":i,"v":j,"y":list(np.where(Y[(i,j)]==1)[0])} for i,j in L])
    sp={"train":mk(tr[:int(.9*len(tr))]),"valid":mk(tr[int(.9*len(tr)):]),"test":mk(te)}
    saved=CFG.max_epochs;CFG.max_epochs=60;res={}
    for at in [True,False]:
        set_seed(0);m=TargetDDI(48,Ntg,C,CFG,add_target=at).to(DEVICE)
        m,thr,h=train_one(m,sp,T,id2n,C)
        p=predict(m,sp["test"],T,id2n,C);res[at]=evaluate(multihot(sp["test"]["y"],C),p,thr)["micro_auprc"]
    CFG.max_epochs=saved
    print(f"  self-test S3 micro-AP: WITH target={res[True]:.3f}  mol-only={res[False]:.3f}  Δ={res[True]-res[False]:+.3f}")
    assert res[True]>res[False]+0.05,"target branch not helping on rigged cold-start -> bug"
    assert res[False]>0.05,"mol-only degenerate -> bug"
    print("  SELF-TEST PASSED")
print("=== SELF-TEST ===");selftest()
