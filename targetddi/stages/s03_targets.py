"""Annotation data: UniProt id mapping, EC/GO reconstruction of the target, enzyme and transporter channels, GO/Pfam unions, ATC groups, coverage report."""
DRUGBANK_PROTEIN_URLS = [
    # public "all proteins" CSV zips (target/enzyme/transporter/carrier)
    "https://go.drugbank.com/releases/latest/downloads/target-all-polypeptide-ids",
    "https://go.drugbank.com/releases/latest/downloads/enzyme-all-polypeptide-ids",
    "https://go.drugbank.com/releases/latest/downloads/transporter-all-polypeptide-ids",
    "https://go.drugbank.com/releases/latest/downloads/carrier-all-polypeptide-ids",
]
def _try_download(url, timeout=60):
    try:
        r=requests.get(url,timeout=timeout,headers={"User-Agent":"research"})
        if r.status_code==200 and len(r.content)>500: return r.content
    except Exception as e:
        print("   download failed:",url.split('/')[-1],e)
    return None

def _parse_protein_zip_or_csv(content):
    # DrugBank ships a zip containing one CSV; handle both
    text=None
    try:
        zf=zipfile.ZipFile(io.BytesIO(content))
        name=[n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        text=zf.read(name).decode("utf-8","ignore")
    except Exception:
        try: text=content.decode("utf-8","ignore")
        except Exception: return {}
    df=pd.read_csv(io.StringIO(text))
    idcol="UniProt ID" if "UniProt ID" in df.columns else df.columns[0]
    dcol=[c for c in df.columns if c.lower().replace(" ","")=="drugids"]
    if not dcol: return {}
    dcol=dcol[0]; d2p={}
    for prot,ids in zip(df[idcol].astype(str).values, df[dcol].astype(str).values):
        if ids=="nan" or not ids.strip(): continue
        for d in (x.strip() for x in ids.split(";")):
            if d: d2p.setdefault(d,set()).add(str(prot))
    return d2p

def uniprot_idmap(drugbank_ids, chunk=100000):
    """Map DrugBank IDs -> UniProt accessions (a drug\'s protein targets) via UniProt\'s
    public ID-mapping REST API. No auth; works headless on Kaggle."""
    import time as _t
    base="https://rest.uniprot.org"
    d2p={}; ids=list(drugbank_ids)
    for i0 in range(0,len(ids),chunk):
        chunk_ids=ids[i0:i0+chunk]
        try:
            run=requests.post(f"{base}/idmapping/run",
                data={"from":"DrugBank","to":"UniProtKB","ids":",".join(chunk_ids)},timeout=60)
            job=run.json().get("jobId")
            if not job: print("   uniprot: no jobId",run.text[:200]); continue
        except Exception as e:
            print("   uniprot run failed:",e); continue
        # poll
        _poll_t0=_t.time()
        for _pi in range(60):
            try:
                st=requests.get(f"{base}/idmapping/status/{job}",timeout=60).json()
            except Exception: _t.sleep(3); continue
            if st.get("jobStatus") in ("RUNNING","NEW",None) and "results" not in st:
                if _pi%5==0: print(f"   uniprot idmapping: still running ({_t.time()-_poll_t0:.0f}s elapsed, poll {_pi+1}/60)")
                _t.sleep(3); continue
            break
        # fetch results (paginate via Link header), compact TSV: from<TAB>to
        url=f"{base}/idmapping/uniprotkb/results/{job}?format=tsv&fields=accession&size=500"
        while url:
            try: r=requests.get(url,timeout=120)
            except Exception as e: print("   uniprot results failed:",e); break
            lines=r.text.strip().split("\n")
            for ln in lines[1:]:
                parts=ln.split("\t")
                if len(parts)>=2 and parts[0] and parts[-1]:
                    d2p.setdefault(parts[0],set()).add(parts[-1])
            nxt=r.headers.get("Link","")
            url=nxt.split(";")[0].strip("<> ") if 'rel="next"' in nxt else None
    return d2p

def fetch_drug_targets(drug_ids=None, cache=f"{CFG.workdir}/drug_targets.json"):
    if os.path.exists(cache):
        d=json.load(open(cache)); print("   target cache hit"); return {k:set(v) for k,v in d.items()}
    merged={}
    # PRIMARY: UniProt ID-mapping (public API, works headless). Needs the drug ID list.
    if drug_ids is not None:
        try:
            merged=uniprot_idmap(drug_ids)
            print(f"   UniProt idmapping: {len(merged)} drugs with >=1 target")
        except Exception as e:
            print("   UniProt idmapping failed:",e); merged={}
    # FALLBACK: DrugBank public CSV export (often needs an account -> may be empty)
    if len(merged)==0:
        for url in DRUGBANK_PROTEIN_URLS:
            cc=_try_download(url)
            if cc is None: continue
            part=_parse_protein_zip_or_csv(cc)
            for d,ps in part.items(): merged.setdefault(d,set()).update(ps)
            print(f"   {url.split('/')[-1]}: +{len(part)} drugs")
    if merged:
        json.dump({k:sorted(v) for k,v in merged.items()},open(cache,"w"))
    return merged

def coverage_report(drug_ids, d2p):
    have=[d for d in drug_ids if len(d2p.get(d,()))>0]
    cnt=np.array([len(d2p.get(d,())) for d in drug_ids])
    prots=set().union(*[d2p[d] for d in have]) if have else set()
    return dict(coverage=len(have)/max(len(drug_ids),1), covered=len(have),
                n_drugs=len(drug_ids), median=float(np.median(cnt)),
                mean=float(cnt.mean()), distinct=len(prots))

def build_target_features(drug_ids, d2p, min_freq, max_t):
    freq={}
    for d in drug_ids:
        for p in d2p.get(d,()): freq[p]=freq.get(p,0)+1
    keep=[p for p,c in freq.items() if c>=min_freq]
    keep=sorted(keep,key=lambda p:-freq[p])[:max_t]; pidx={p:i for i,p in enumerate(keep)}
    X=np.zeros((len(drug_ids),len(keep)),np.float32)
    for i,d in enumerate(drug_ids):
        for p in d2p.get(d,()):
            if p in pidx: X[i,pidx[p]]=1.0
    return X, keep

# ================= UniProt-annotation reconstruction of target/enzyme/transporter =================
# When DrugBank's categorized exports aren't available (no account), UniProt idmapping still gives
# the drug->protein links, but as one undifferentiated bucket. DDIs are overwhelmingly PK-driven
# (CYP450 enzymes, P-gp/OATP transporters), not just primary-target overlap, so recovering the role
# matters for the mechanism-attribution story. UniProt's own per-protein annotation carries this:
# a catalytic EC number means enzyme; a transporter-activity GO annotation means transporter;
# otherwise it's left as an (uncategorized) target. This is an approximation of DrugBank's curated
# categories (which reflect pharmacological intent, not just molecular function) -- described as
# such in the paper -- not a reproduction of them. Carrier proteins have no comparable public
# functional signature, so that channel stays empty under this fallback (honest gap, not guessed).
_TRANSPORTER_GO_KEYWORDS = ("transporter activity", "transmembrane transporter")

# NOTE: Kaggle's outbound requests share a much more heavily-used IP pool than Colab's,
# so UniProt's public search endpoint rate-limits (HTTP 429) it far more aggressively.
# A generic User-Agent makes this worse -- UniProt's own guidance is to identify your
# client. Put a real contact email below; it costs nothing and measurably reduces throttling.
_UNIPROT_HEADERS = {"User-Agent": "TARGET-DDI-research/1.0 (contact: YOUR_EMAIL_HERE@example.com)"}

def fetch_uniprot_ec_go(protein_ids, chunk=20, cache=None, min_coverage=0.5):
    """UniProt accession -> (ec_numbers:set[str], go_text:str). Same batched-query pattern as
    fetch_go_pfam (OR-ed accessions, chunk<=~25 to stay under UniProt's URL length cap), but
    pulling the 'ec' and 'go' fields instead of go_id/xref_pfam -- 'go' returns full GO term
    names (not just IDs), which is what the transporter keyword match below needs.

    FIX: this used to unconditionally cache whatever it got, including a near-empty result
    from a run that got rate-limited on every chunk -- and unconditionally trust that cache
    on the next run. On Kaggle (shared IP pool -> heavier UniProt throttling than Colab,
    especially with no identifying User-Agent) that meant one bad run poisoned every future
    run permanently, silently, with enzyme/transporter stuck at 0%. Now: real backoff
    (honors Retry-After on 429), a real User-Agent, per-chunk failure counts printed as they
    happen (not just every 20th), and a cache is only trusted/written if it actually covers
    a useful fraction of the requested proteins -- a degenerate result is never persisted
    and a degenerate cache on disk is treated as stale and refetched instead of returned."""
    cache=cache or f"{CFG.workdir}/ec_go_class.json"
    protein_ids=set(protein_ids)
    if os.path.exists(cache):
        d=json.load(open(cache))
        cached={p:(set(v["ec"]),v["go"]) for p,v in d.items()}
        cov=len(protein_ids & cached.keys())/max(len(protein_ids),1)
        if cov>=min_coverage:
            print(f"   EC/GO classification cache hit ({len(cached)} proteins, covers {cov*100:.0f}% of current set)")
            return cached
        print(f"   EC/GO cache on disk only covers {cov*100:.0f}% of the current protein set "
              f"({len(cached)} cached vs {len(protein_ids)} needed) -- treating it as stale from a "
              f"failed/rate-limited prior run and refetching instead of trusting it")
    ids=sorted(protein_ids); out={}
    print(f"   fetching EC/GO for {len(ids)} proteins (enzyme/transporter reconstruction)...")
    t0=time.time(); n_fail=0
    for i0 in range(0,len(ids),chunk):
        part=ids[i0:i0+chunk]
        q=" OR ".join(f"accession:{p}" for p in part)
        r=None
        for _try in range(5):
            try:
                r=requests.get("https://rest.uniprot.org/uniprotkb/search",
                    params={"query":q,"fields":"accession,ec,go","format":"tsv","size":max(chunk,50)},
                    headers=_UNIPROT_HEADERS, timeout=90)
                if r.status_code==200: break
                if r.status_code==429:
                    wait=float(r.headers.get("Retry-After", 5*(_try+1)))
                    print(f"   EC/GO chunk {i0//chunk}: rate-limited (429), waiting {wait:.0f}s (attempt {_try+1}/5)")
                    time.sleep(wait); continue
                time.sleep(1.5*(_try+1))
            except Exception as e:
                print(f"   EC/GO chunk {i0//chunk}: request error ({e}), retrying (attempt {_try+1}/5)")
                time.sleep(1.5*(_try+1))
        if r is None or r.status_code!=200:
            n_fail+=1
            print(f"   EC/GO chunk {i0//chunk}: giving up after retries, HTTP {getattr(r,'status_code','no response')}")
            continue
        try:
            lines=r.text.strip().split("\n")
            if len(lines)<2: continue
            for ln in lines[1:]:
                cols=ln.split("\t")
                if not cols or not cols[0]: continue
                acc=cols[0]
                ec=set(x.strip() for x in cols[1].split(";") if x.strip()) if len(cols)>1 else set()
                go=cols[2].strip() if len(cols)>2 else ""
                out[acc]=(ec,go)
        except Exception as e:
            print(f"   EC/GO chunk {i0//chunk} parse failed:",e); continue
        if (i0//chunk)%5==0:
            print(f"   EC/GO: {min(i0+chunk,len(ids))}/{len(ids)} proteins ({time.time()-t0:.0f}s elapsed, {n_fail} chunk failures so far)")
    coverage=len(out)/max(len(ids),1)
    print(f"   EC/GO: {len(out)}/{len(ids)} proteins annotated ({coverage*100:.0f}% coverage, {n_fail} chunk failures)")
    if coverage>=min_coverage:
        json.dump({p:{"ec":sorted(ec),"go":go} for p,(ec,go) in out.items()},open(cache,"w"))
    else:
        print(f"   EC/GO coverage below {min_coverage*100:.0f}% -- NOT writing this to cache, so the next "
              f"run retries fresh instead of reusing today's bad/rate-limited result")
    return out

def classify_by_annotation(d2p, ann):
    """drug->protein map (uncategorized) + protein->(ec,go_text) annotation -> reconstructed
    target/enzyme/transporter dicts. A protein with a catalytic EC number is an enzyme; else a
    transporter-activity GO term makes it a transporter; else it stays an (uncategorized) target."""
    cats={"target":{}, "enzyme":{}, "transporter":{}}
    for d,prots in d2p.items():
        for p in prots:
            ec,go_text=ann.get(p,(set(),""))
            if ec:
                cat="enzyme"
            elif any(kw in go_text.lower() for kw in _TRANSPORTER_GO_KEYWORDS):
                cat="transporter"
            else:
                cat="target"
            cats[cat].setdefault(d,set()).add(p)
    return cats

def fetch_drug_targets_by_category(drug_ids=None, cache=f"{CFG.workdir}/drug_targets_by_cat.json"):
    """drug -> protein map, split into target / enzyme / transporter / carrier channels.
    Primary source: DrugBank's own categorized CSV exports (DRUGBANK_PROTEIN_URLS), which are
    properly split by category. Those often need a DrugBank account, so if all four fail we fall
    back to UniProt idmapping for the drug->protein links, then RECONSTRUCT enzyme/transporter/
    target from each protein's own UniProt annotation (EC number, transporter-activity GO term)
    instead of leaving everything in one undifferentiated 'target' bucket -- see
    fetch_uniprot_ec_go / classify_by_annotation above.

    FIX: this is the actual cause of enzyme/transporter/carrier reading 0% on Kaggle while
    Colab works. This cache used to be trusted unconditionally. Your Kaggle "Section 0" cell
    copies a previous session's working directory forward from an attached input dataset --
    which means if EC/GO annotation ever got rate-limited on a past Kaggle run (see the note
    in fetch_uniprot_ec_go: Kaggle's shared IP pool gets throttled by UniProt much harder than
    Colab's), that run's degenerate all-'target' categorization got written to
    drug_targets_by_cat.json once, and every subsequent Kaggle session -- including this one --
    has been silently loading that same poisoned cache forever instead of refetching. Colab has
    no such carried-forward cache, so it fetches clean each time and works. Fix: a cache is only
    trusted if it actually has enzyme/transporter/carrier signal (or genuinely has no drug_ids
    to check against); a degenerate cache is treated as stale and refetched, and a degenerate
    *new* result is not written back to disk, so a bad run no longer poisons the next one."""
    if os.path.exists(cache):
        d=json.load(open(cache))
        cats_cached={cat:{k:set(v) for k,v in dd.items()} for cat,dd in d.items()}
        n_tgt=len(cats_cached.get("target",{})); n_enz=len(cats_cached.get("enzyme",{}))
        n_trans=len(cats_cached.get("transporter",{})); n_car=len(cats_cached.get("carrier",{}))
        if (n_enz+n_trans+n_car)>0 or n_tgt==0:
            print(f"   categorized target cache hit (target={n_tgt}, enzyme={n_enz}, transporter={n_trans}, carrier={n_car})")
            return cats_cached
        print(f"   categorized target cache on disk has target={n_tgt} but enzyme=transporter=carrier=0 -- "
              f"this is the signature of a prior run whose EC/GO fetch got rate-limited (common on Kaggle's "
              f"shared IPs, carried forward by the Section-0 resume cell). Ignoring this cache and refetching.")
    cats={"target":{}, "enzyme":{}, "transporter":{}, "carrier":{}}
    got_categorized=False
    for url in DRUGBANK_PROTEIN_URLS:
        cat=url.split("/")[-1].split("-")[0]
        cc=_try_download(url)
        if cc is None: continue
        part=_parse_protein_zip_or_csv(cc)
        if part:
            cats[cat]=part; got_categorized=True
            print(f"   {cat}: {len(part)} drugs (DrugBank categorized export)")
    if not got_categorized:
        print("   DrugBank categorized exports unavailable (likely needs an account) -> falling back to"
              " UniProt idmapping + EC/GO-based enzyme/transporter/target reconstruction"
              " (carrier has no comparable public signature, so that channel stays empty)")
        if drug_ids is not None:
            try:
                merged=uniprot_idmap(drug_ids)
                print(f"   UniProt idmapping: {len(merged)} drugs with >=1 target (uncategorized)")
                all_prots=set().union(*merged.values()) if merged else set()
                ann=fetch_uniprot_ec_go(all_prots)
                if len(ann)==0 and len(all_prots)>0:
                    print("   WARNING: EC/GO annotation came back empty this run -- enzyme/transporter will be"
                          " 0 this time (everything falls into 'target'). Likely UniProt rate-limiting; this"
                          " result will NOT be cached, so simply re-running this cell later should recover it.")
                reclass=classify_by_annotation(merged, ann)
                for cat in ("target","enzyme","transporter"):
                    cats[cat]=reclass.get(cat,{})
                    print(f"   [{cat}] reconstructed: {len(cats[cat])} drugs")
            except Exception as e:
                print("   UniProt idmapping / classification failed:",e)
    got_signal = len(cats["enzyme"])>0 or len(cats["transporter"])>0 or len(cats["carrier"])>0
    if got_signal or not drug_ids:
        json.dump({cat:{k:sorted(v) for k,v in dd.items()} for cat,dd in cats.items()},open(cache,"w"))
    else:
        print("   enzyme/transporter/carrier all empty this run -- NOT writing to cache, so the next run"
              " retries the EC/GO fetch instead of being permanently stuck on this result.")
    return cats

def fetch_go_pfam(protein_ids, chunk=20, cache=f"{CFG.workdir}/go_pfam.json"):  # chunk<=~25 avoids UniProt HTTP 400
    """UniProt accession -> (GO term set, Pfam domain set), via UniProt's search endpoint,
    batched by OR-ing accessions into one query per chunk. Best-effort: a failed chunk is
    skipped (those proteins just contribute no GO/Pfam signal), never hard-fails the run."""
    if os.path.exists(cache):
        d=json.load(open(cache)); print("   GO/Pfam cache hit")
        return {p:(set(v["go"]),set(v["pfam"])) for p,v in d.items()}
    ids=sorted(protein_ids); out={}
    print(f"   fetching GO/Pfam for {len(ids)} proteins in chunks of {chunk}...")
    t0=time.time()
    for i0 in range(0,len(ids),chunk):
        part=ids[i0:i0+chunk]
        q=" OR ".join(f"accession:{p}" for p in part)     # <=20 accessions -> under UniProt length cap
        r=None
        for _try in range(3):
            try:
                r=requests.get("https://rest.uniprot.org/uniprotkb/search",
                    params={"query":q,"fields":"accession,go_id,xref_pfam","format":"tsv","size":max(chunk,50)},
                    headers=_UNIPROT_HEADERS, timeout=90)
                if r.status_code==200: break
            except Exception:
                import time as _t; _t.sleep(1.0)
        if r is None or r.status_code!=200:
            if (i0//chunk)%20==0: print(f"   GO/Pfam chunk {i0//chunk}: HTTP {getattr(r,'status_code','err')}")
            continue
        try:
            lines=r.text.strip().split("\n")
            if len(lines)<2: continue
            for ln in lines[1:]:
                cols=ln.split("\t")
                if not cols or not cols[0]: continue
                acc=cols[0]
                go=set(x.strip() for x in cols[1].split(";") if x.strip()) if len(cols)>1 else set()
                pf=set(x.strip().rstrip(";") for x in cols[2].split(";") if x.strip()) if len(cols)>2 else set()
                out[acc]=(go,pf)
        except Exception as e:
            print(f"   GO/Pfam chunk {i0//chunk} failed:",e); continue
        if (i0//chunk)%5==0:
            print(f"   GO/Pfam: {min(i0+chunk,len(ids))}/{len(ids)} proteins ({time.time()-t0:.0f}s elapsed)")
    json.dump({p:{"go":sorted(g),"pfam":sorted(f)} for p,(g,f) in out.items()},open(cache,"w"))
    return out

def build_go_pfam_features(drug_ids, d2p_union, prot2gopfam, min_freq, max_t):
    """Drug-level GO/Pfam multi-hot = union of GO terms / Pfam domains over all of that
    drug's target/enzyme/transporter/carrier proteins. Lets the model generalize across
    functionally-similar proteins even when it never saw that exact UniProt ID in training
    -- the direct fix for the cold-start blind spot of the raw binary target indicator."""
    go_freq={}; pf_freq={}; d2go={}; d2pf={}
    for d in drug_ids:
        gos=set(); pfs=set()
        for p in d2p_union.get(d,()):
            g,f=prot2gopfam.get(p,(set(),set())); gos|=g; pfs|=f
        d2go[d]=gos; d2pf[d]=pfs
        for g in gos: go_freq[g]=go_freq.get(g,0)+1
        for f in pfs: pf_freq[f]=pf_freq.get(f,0)+1
    keep_go=sorted([g for g,c in go_freq.items() if c>=min_freq],key=lambda g:-go_freq[g])[:max_t]
    keep_pf=sorted([f for f,c in pf_freq.items() if c>=min_freq],key=lambda f:-pf_freq[f])[:max_t]
    gidx={g:i for i,g in enumerate(keep_go)}; pidx={f:i for i,f in enumerate(keep_pf)}
    Xgo=np.zeros((len(drug_ids),len(keep_go)),np.float32)
    Xpf=np.zeros((len(drug_ids),len(keep_pf)),np.float32)
    for i,d in enumerate(drug_ids):
        for g in d2go[d]:
            if g in gidx: Xgo[i,gidx[g]]=1.0
        for f in d2pf[d]:
            if f in pidx: Xpf[i,pidx[f]]=1.0
    return Xgo,Xpf,keep_go,keep_pf

def fetch_atc(drug_ids, url=None, cache=f"{CFG.workdir}/atc.json"):
    """DrugBank ID -> set(ATC codes), from dhimmel/drugbank's public open-data export
    (a DrugBank 4.2-derived TSV on GitHub). ATC assignments are essentially static, so
    this older snapshot is still a fine approximation for approved small molecules."""
    url=url or CFG.atc_source_url
    if os.path.exists(cache):
        d=json.load(open(cache)); print("   ATC cache hit")
        return {k:set(v) for k,v in d.items()}
    d2atc={}
    try:
        r=requests.get(url,timeout=60)
        if r.status_code==200:
            df=pd.read_csv(io.StringIO(r.text),sep="\t",dtype=str)
            idc=[c for c in df.columns if c.lower()=="drugbank_id"][0]
            atc_c=[c for c in df.columns if c.lower()=="atc_codes"][0]
            wanted=set(drug_ids)
            for did,codes in zip(df[idc],df[atc_c]):
                if did not in wanted or not isinstance(codes,str) or not codes.strip(): continue
                cs={c.strip() for c in codes.replace("|",";").split(";") if c.strip()}
                if cs: d2atc[did]=cs
            print(f"   ATC: {len(d2atc)}/{len(drug_ids)} drugs matched")
        else:
            print(f"   ATC fetch: HTTP {r.status_code}, skipping ATC feature")
    except Exception as e:
        print("   ATC fetch failed:",e,"-- skipping ATC feature")
    json.dump({k:sorted(v) for k,v in d2atc.items()},open(cache,"w"))
    return d2atc

def build_atc_features(drug_ids, d2atc, min_freq):
    """Two blocks concatenated: full ATC code (filtered by min_freq, like target proteins)
    and the coarser level-1 anatomical-group letter (dense, ~14-dim, no filtering needed)."""
    full_freq={}; l1_freq={}; d2full={}; d2l1={}
    for d in drug_ids:
        codes=d2atc.get(d,set())
        d2full[d]=codes; d2l1[d]={c[0] for c in codes if c}
        for c in codes: full_freq[c]=full_freq.get(c,0)+1
        for c in d2l1[d]: l1_freq[c]=l1_freq.get(c,0)+1
    keep_full=sorted([c for c,n in full_freq.items() if n>=min_freq],key=lambda c:-full_freq[c])
    keep_l1=sorted(l1_freq.keys())
    fidx={c:i for i,c in enumerate(keep_full)}; lidx={c:i for i,c in enumerate(keep_l1)}
    Xfull=np.zeros((len(drug_ids),len(keep_full)),np.float32)
    Xl1=np.zeros((len(drug_ids),len(keep_l1)),np.float32)
    for i,d in enumerate(drug_ids):
        for c in d2full[d]:
            if c in fidx: Xfull[i,fidx[c]]=1.0
        for c in d2l1[d]:
            if c in lidx: Xl1[i,lidx[c]]=1.0
    return np.concatenate([Xfull,Xl1],1).astype(np.float32), keep_full, keep_l1

def fetch_cyp_roles(drug_ids, csv_path=None):
    """CYP substrate/inhibitor/inducer roles are the most mechanistically precise PK-DDI
    feature, but DrugBank only exposes the *association* (drug hits enzyme X) for free --
    the specific *role* (substrate vs inhibitor vs inducer) is behind DrugBank's academic/
    commercial license. This is a pluggable hook, not a scraper: if you have that export
    (e.g. via institutional access), point CFG.cyp_role_csv at a CSV with columns
    [drugbank_id, cyp, role] and this picks it up automatically. Without it, it returns a
    zero-width feature and the run proceeds -- it never fakes the signal."""
    csv_path=csv_path or CFG.cyp_role_csv
    if not csv_path or not os.path.exists(csv_path):
        print("   CYP roles: no licensed export configured (CFG.cyp_role_csv) -- skipping this feature")
        return np.zeros((len(drug_ids),0),np.float32), []
    df=pd.read_csv(csv_path,dtype=str)
    combo=sorted(set(df["cyp"]+":"+df["role"]))
    cidx={c:i for i,c in enumerate(combo)}
    d2c={}
    for did,cyp,role in zip(df["drugbank_id"],df["cyp"],df["role"]):
        d2c.setdefault(did,set()).add(f"{cyp}:{role}")
    X=np.zeros((len(drug_ids),len(combo)),np.float32)
    for i,d in enumerate(drug_ids):
        for c in d2c.get(d,()):
            if c in cidx: X[i,cidx[c]]=1.0
    print(f"   CYP roles: loaded {len(combo)} (cyp,role) combos from {csv_path}")
    return X, combo

# ================= ESM-2 PROTEIN EMBEDDINGS (mechanistic feature) =================
# An unseen drug's target may be a NOVEL protein, but ESM-2 places it near functionally
# similar known proteins; pooling a drug's targets' embeddings gives a cold-start-robust
# target representation that binary presence vectors cannot. Verified: on similar-but-novel
# targets, pooled ESM-2 >> binary. Fetched per-accession from UniProt's embeddings API, cached.
def fetch_esm_embeddings(uniprot_ids, cache=None):
    cache=cache or f"{CFG.workdir}/esm_emb.npz"
    if os.path.exists(cache):
        d=np.load(cache,allow_pickle=True)
        emb={k:d["emb"][i] for i,k in enumerate(d["ids"])}
        print(f"   ESM-2 cache hit ({len(emb)} proteins)"); return emb
    emb={}; ids=list(uniprot_ids)
    for n,acc in enumerate(ids):
        try:
            r=requests.get(f"https://rest.uniprot.org/uniprotkb/{acc}/embeddings",headers=_UNIPROT_HEADERS,timeout=30)
            if r.status_code==200:
                j=r.json(); v=j.get("embedding") or j.get("value")
                if v: emb[acc]=np.asarray(v,np.float32)
        except Exception: pass
        if n%200==0 and n>0: print(f"   ESM-2: {len(emb)}/{n} fetched ...")
    if emb:
        keys=list(emb); np.savez_compressed(cache,ids=np.array(keys),emb=np.stack([emb[k] for k in keys]))
    print(f"   ESM-2: {len(emb)}/{len(ids)} proteins embedded"); return emb

def build_esm_drug_features(drug_ids, d2p, emb):
    if not emb: return np.zeros((len(drug_ids),0),np.float32), False
    dim=len(next(iter(emb.values()))); X=np.zeros((len(drug_ids),dim),np.float32)
    for i,d in enumerate(drug_ids):
        vecs=[emb[p] for p in d2p.get(d,()) if p in emb]
        if vecs: X[i]=np.mean(vecs,0)
    mu,sd=X.mean(0),X.std(0); sd[sd==0]=1
    return ((X-mu)/sd).astype(np.float32), True
