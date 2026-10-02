"""Model: molecular and target encoders, availability-conditioned gate, shared-target propagation, training loop."""
class Enc(nn.Module):
    def __init__(s,din,h,drop):
        super().__init__(); s.net=nn.Sequential(nn.Linear(din,h),nn.LayerNorm(h),nn.ReLU(),nn.Dropout(drop),
                          nn.Linear(h,h),nn.LayerNorm(h),nn.ReLU(),nn.Dropout(drop))
    def forward(s,x): return s.net(x)

class SharedTargetLayer(nn.Module):
    """THE EDGE: shared-target message passing over a drug<->protein bipartite graph.
    An unseen drug is enriched by the TRAINING drugs it shares targets with; only
    training-drug embeddings are sources (train_mask) so no test leak. Target
    annotations are external biology, never learned from DDI labels."""
    def __init__(s,h): super().__init__(); s.lin=nn.Linear(h,h); s.norm=nn.LayerNorm(h)
    def forward(s,h,tgt_bin,train_mask):
        Ttr=tgt_bin*train_mask.unsqueeze(1).to(h.dtype)
        tdeg=Ttr.sum(0).clamp(min=1.0)
        s._tvec=(Ttr.t()@h)/tdeg.unsqueeze(1)               # cache target embeddings for attribution
        ddeg=tgt_bin.sum(1).clamp(min=1.0)
        agg=(tgt_bin@s._tvec)/ddeg.unsqueeze(1)
        return s.norm(h+s.lin(agg))

class TargetDDI(nn.Module):
    # mode: "molecular" | "target_only" | "gated" | "graph" (gated + shared-target edge)
    def __init__(s,dmol,dtgt,C,cfg,mode="gated",add_target=None):
        super().__init__(); h=cfg.hidden
        if add_target is not None: mode="gated" if add_target else "molecular"
        s.mode=mode; s.use_tgt=(mode in ("gated","graph","target_only")) and dtgt>0
        if s.mode!="molecular" and not s.use_tgt: s.mode="molecular"
        s.mol=Enc(dmol,h,cfg.dropout)
        if s.use_tgt:
            s.tgt=Enc(dtgt,h,cfg.dropout)
            if s.mode in ("gated","graph"): s.gate=nn.Sequential(nn.Linear(2*h+1,h),nn.ReLU(),nn.Linear(h,h))
            if s.mode=="graph": s.stl=SharedTargetLayer(h)
        s.head=nn.Sequential(nn.Linear(3*h,cfg.pair_hidden),nn.BatchNorm1d(cfg.pair_hidden),
                             nn.ReLU(),nn.Dropout(cfg.dropout),nn.Linear(cfg.pair_hidden,C))
    def nodes(s,T):
        m=s.mol(T["mol"])
        if not s.use_tgt: return m
        t=s.tgt(T["tgt"])
        if s.mode=="target_only": z=t
        else:
            g=torch.sigmoid(s.gate(torch.cat([m,t,T["tcount"]],-1))); z=g*t+(1-g)*m
        if s.mode=="graph": z=s.stl(z,T["tgt_bin"],T["train_mask"])
        return z
    def forward(s,T,pu,pv):
        z=s.nodes(T); zi,zj=z[pu],z[pv]
        return s.head(torch.cat([zi+zj,(zi-zj).abs(),zi*zj],-1))
    @torch.no_grad()
    def attribute_pair(s,T,i,j,topk=5):
        """MECHANISM ATTRIBUTION (graph mode): rank the targets SHARED by drugs i,j by how much
        removing each changes the pair\'s predicted logits. Faithful by construction (ablation).
        Returns list of (target_index, effect). This is what we validate against DrugBank mechanisms."""
        if s.mode!="graph": return []
        dev=T["mol"].device
        def logit(tb):
            Tt=dict(T); Tt["tgt_bin"]=tb
            z=s.nodes(Tt); return s.head(torch.cat([z[i]+z[j],(z[i]-z[j]).abs(),z[i]*z[j]],-1))
        base=logit(T["tgt_bin"])
        shared=torch.where((T["tgt_bin"][i]*T["tgt_bin"][j])>0)[0].tolist()
        eff={}
        for t in shared:
            tb=T["tgt_bin"].clone(); tb[i,t]=0; tb[j,t]=0
            eff[t]=float((base-logit(tb)).abs().sum().item())
        return sorted(eff.items(),key=lambda kv:-kv[1])[:topk]

class AsymmetricLoss(nn.Module):
    def __init__(s,gn,gp,clip,eps=1e-8): super().__init__(); s.gn,s.gp,s.clip,s.eps=gn,gp,clip,eps
    def forward(s,logit,y):
        xs=torch.sigmoid(logit);xp=xs;xn=(1-xs)
        if s.clip>0: xn=(xn+s.clip).clamp(max=1)
        l=y*torch.log(xp.clamp(min=s.eps))+(1-y)*torch.log(xn.clamp(min=s.eps))
        pt=xp*y+xn*(1-y);g=s.gp*y+s.gn*(1-y);return -(l*torch.pow(1-pt,g)).sum(1).mean()

from torch.utils.data import DataLoader,TensorDataset
def loader(part,id2n,C,bs,sh):
    pu=torch.tensor([id2n[u] for u in part["u"]]);pv=torch.tensor([id2n[v] for v in part["v"]])
    return DataLoader(TensorDataset(pu,pv,torch.tensor(multihot(part["y"],C))),batch_size=bs,shuffle=sh)
@torch.no_grad()
def predict(model,part,T,id2n,C):
    model.eval();z=model.nodes(T);out=[]
    for pu,pv,_ in loader(part,id2n,C,CFG.eval_batch,False):
        zi,zj=z[pu.to(DEVICE)],z[pv.to(DEVICE)]
        out.append(torch.sigmoid(model.head(torch.cat([zi+zj,(zi-zj).abs(),zi*zj],-1))).float().cpu().numpy())
    P=np.concatenate(out)
    return np.nan_to_num(P,nan=0.0,posinf=1.0,neginf=0.0)     # metrics never see NaN
def atomic_save(o,p): tmp=p+".tmp";torch.save(o,tmp);os.replace(tmp,p)

def train_one(model,sp,T,id2n,C,ckpt=None,log=False):
    opt=torch.optim.AdamW(model.parameters(),CFG.lr,weight_decay=CFG.weight_decay)
    warm=torch.optim.lr_scheduler.LambdaLR(opt,lambda e:min(1.0,(e+1)/max(1,CFG.warmup_epochs)))
    plateau=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=CFG.lr_factor,patience=CFG.lr_patience,min_lr=CFG.min_lr)
    scaler=torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))
    crit=AsymmetricLoss(CFG.asl_gamma_neg,CFG.asl_gamma_pos,CFG.asl_clip)
    dl=loader(sp["train"],id2n,C,CFG.batch_size,True);yv=multihot(sp["valid"]["y"],C)
    best=-1;best_ema=-1;bad=0;hist=[];start=0;best_state=None;ema=None
    if ckpt and os.path.exists(ckpt):
        try:
            st=torch.load(ckpt,map_location=DEVICE,weights_only=False)
            model.load_state_dict(st["model"]);opt.load_state_dict(st["opt"])
            warm.load_state_dict(st["warm"]);plateau.load_state_dict(st["plateau"]);scaler.load_state_dict(st["scaler"])
            start=st["epoch"]+1;best=st["best"];bad=st["bad"];hist=st["hist"];best_state=st["best_state"];ema=st.get("ema");best_ema=st.get("best_ema",best)
        except Exception as e: print("   ckpt unreadable, restart unit:",e)
    for ep in range(start,CFG.max_epochs):
        model.train();tl=0.0;t0=time.time()
        for pu,pv,y in dl:
            pu,pv,y=pu.to(DEVICE),pv.to(DEVICE),y.to(DEVICE);opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                logit=model(T,pu,pv)
            loss=crit(logit.float(),y.float())              # loss in fp32 (AMP-stable)
            if not torch.isfinite(loss):                     # skip a bad batch, never poison weights
                opt.zero_grad(set_to_none=True); continue
            scaler.scale(loss).backward();scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),CFG.grad_clip);scaler.step(opt);scaler.update()
            tl+=loss.item()*len(pu)
        pv_=predict(model,sp["valid"],T,id2n,C);thr=tune_thresholds(yv,pv_);mv=evaluate(yv,pv_,thr)
        if ep<CFG.warmup_epochs: warm.step()
        else: plateau.step(mv["micro_auprc"])
        hist.append({"epoch":ep,"loss":tl/max(len(sp["train"]),1),**{f"val_{k}":v for k,v in mv.items()}})
        if log: print(f"    ep {ep:03d} loss {tl/max(len(sp['train']),1):.4f} val microAP {mv['micro_auprc']:.4f} "
                      f"macroF1 {mv['macro_f1']:.4f} lr {opt.param_groups[0]['lr']:.1e} bad {bad} ({time.time()-t0:.0f}s)")
        cur=mv["micro_auprc"]
        ema=cur if ema is None else (CFG.ema_beta*ema+(1-CFG.ema_beta)*cur)   # smoothed selection signal
        # keep the TRUE best raw-val checkpoint (that is what we evaluate on test),
        # but drive early-stopping off the SMOOTHED metric so noise can't trigger a false stop.
        if cur>best+CFG.min_delta:
            best=cur
            best_state={"model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},"thr":thr,"ema":ema}
        if ema>best_ema+CFG.min_delta:
            best_ema=ema;bad=0
        else: bad+=1
        if log: print(f"      (ema {ema:.4f} bestema {best_ema:.4f})")
        if ckpt: atomic_save({"model":model.state_dict(),"opt":opt.state_dict(),"warm":warm.state_dict(),
                    "plateau":plateau.state_dict(),"scaler":scaler.state_dict(),"epoch":ep,"best":best,
                    "bad":bad,"hist":hist,"best_state":best_state,"ema":ema,"best_ema":best_ema},ckpt)
        if ep+1>=CFG.min_epochs and bad>=CFG.patience: break   # only allow stopping after min_epochs
    if best_state is not None:
        model.load_state_dict({k:v.to(DEVICE) for k,v in best_state["model"].items()});thr=best_state["thr"]
    return model,thr,hist
