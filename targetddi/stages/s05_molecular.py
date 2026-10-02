"""Dataset loading (TDC DrugBank-DDI), ChemBERTa embeddings, Morgan fingerprints, physicochemical descriptors."""
from tdc.multi_pred import DDI
from rdkit import Chem
from rdkit.Chem import AllChem
def load_dataset(name):
    d=DDI(name=name);df=d.get_data();c={k.lower():k for k in df.columns}
    id1=c.get("drug1_id","Drug1_ID");id2=c.get("drug2_id","Drug2_ID")
    s1=c.get("drug1","Drug1");s2=c.get("drug2","Drug2");lab=c.get("y","Y")
    df=df[[id1,s1,id2,s2,lab]].dropna()
    if df[lab].apply(lambda v:isinstance(v,(list,tuple,np.ndarray))).any(): df=df.explode(lab)
    return build_pair_table(df,id1,id2,s1,s2,lab)
def chemberta(smiles,cache):
    if os.path.exists(cache):
        E=np.load(cache)
        if len(E)==len(smiles): print("   ChemBERTa cache hit"); return E
    from transformers import AutoTokenizer,AutoModel
    tk=AutoTokenizer.from_pretrained(CFG.chemberta_model);m=AutoModel.from_pretrained(CFG.chemberta_model).to(DEVICE).eval()
    out=[];n=len(smiles);nb_batches=math.ceil(n/64);t0=time.time()
    print(f"   ChemBERTa: embedding {n} molecules in {nb_batches} batches on {DEVICE}...")
    with torch.no_grad():
        for bi,i in enumerate(range(0,n,64)):
            e=tk(smiles[i:i+64],padding=True,truncation=True,max_length=CFG.token_max_len,return_tensors="pt").to(DEVICE)
            h=m(**e).last_hidden_state;mask=e["attention_mask"].unsqueeze(-1).float()
            out.append(((h*mask).sum(1)/mask.sum(1).clamp(min=1)).cpu().numpy())
            if (bi+1)%max(1,nb_batches//20)==0 or bi+1==nb_batches:
                elapsed=time.time()-t0;frac=(bi+1)/nb_batches;eta=elapsed/frac-elapsed
                print(f"   ChemBERTa: batch {bi+1}/{nb_batches} ({frac*100:.0f}%) elapsed={elapsed:.0f}s eta={eta:.0f}s")
    E=np.concatenate(out).astype(np.float32);np.save(cache,E);del m;torch.cuda.empty_cache();return E
def morgan(smiles):
    from rdkit.Chem import rdFingerprintGenerator
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=CFG.morgan_radius,fpSize=CFG.morgan_bits)
    X=np.zeros((len(smiles),CFG.morgan_bits),np.float32); bad=0
    for i,s in enumerate(smiles):
        mm=Chem.MolFromSmiles(s)
        if mm is None: bad+=1; continue          # unparseable SMILES -> zero row (kept, handled downstream)
        X[i]=np.array(gen.GetFingerprint(mm),np.float32)
    if bad: print(f"   note: {bad} SMILES unparseable -> zero fingerprint (handled)")
    return X
def molecular_features(drug_ids,ds):
    sm=[ds[d] for d in drug_ids]
    try: cb=chemberta(sm,f"{CFG.workdir}/emb_{CFG.dataset}.npy")
    except Exception as e: print("   ChemBERTa failed:",e); cb=np.zeros((len(sm),0),np.float32)
    return np.concatenate(([cb] if cb.shape[1] else [])+[morgan(sm)],1).astype(np.float32)

def physchem_features(smiles):
    """6 cheap RDKit descriptors (MW, logP, TPSA, HBD, HBA, rotatable bonds), z-scored.
    Free -- reuses RDKit already loaded for Morgan fingerprints, no network call."""
    from rdkit.Chem import Descriptors
    X=np.zeros((len(smiles),6),np.float32); bad=0
    for i,s in enumerate(smiles):
        mm=Chem.MolFromSmiles(s)
        if mm is None: bad+=1; continue
        X[i]=[Descriptors.MolWt(mm),Descriptors.MolLogP(mm),Descriptors.TPSA(mm),
              Descriptors.NumHDonors(mm),Descriptors.NumHAcceptors(mm),Descriptors.NumRotatableBonds(mm)]
    if bad: print(f"   note: {bad} SMILES unparseable for physchem -> zero row (handled)")
    mu=X.mean(0,keepdims=True); sd=X.std(0,keepdims=True); sd[sd<1e-6]=1.0
    return ((X-mu)/sd).astype(np.float32)
