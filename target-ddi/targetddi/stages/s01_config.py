"""Configuration. Every hyperparameter of the paper lives here."""
from types import SimpleNamespace
import os
CFG=SimpleNamespace(
    dataset="DrugBank",                 # clean target join this round
    splits=["S2","S3","S1"],            # S2/S3 FIRST (where the claim lives); S1 last
    headline_seeds=[0,1,2], ablation_seeds=[0],

    frac=(0.7,0.1,0.2), seed=42, min_test_pairs=200,

    chemberta_model="DeepChem/ChemBERTa-77M-MTR", token_max_len=96,
    morgan_bits=1024, morgan_radius=2,

    # target features
    target_min_freq=3,                  # keep proteins hitting >= this many drugs
    target_max=4096,
    coverage_gate=0.60,                 # HARD stop below this drug->target coverage

    # expanded feature dimensions (categorized targets, GO/Pfam, ATC, physchem, CYP hook)
    feature_version="v2multi",          # bump if you change what goes into Xmol/Xtgt -> avoids stale-cache reuse
    enable_categorized_targets=True,    # split target/enzyme/transporter/carrier into separate channels
    enable_go_pfam=True,                # GO + Pfam annotations of each drug's target proteins
    go_pfam_min_freq=3, go_pfam_max=2048,
    enable_atc=True,                    # WHO ATC classification: full code + level-1 anatomical group
    atc_min_freq=2,
    atc_source_url="https://raw.githubusercontent.com/dhimmel/drugbank/gh-pages/data/drugbank-slim.tsv",
    enable_physchem=True,               # RDKit MW/logP/TPSA/HBD/HBA/RotBonds, z-scored
    enable_cyp_roles=False,             # CYP substrate/inhibitor/inducer roles -- needs licensed DrugBank export
    cyp_role_csv=None,                  # set to a local CSV path if you have institutional access; see fetch_cyp_roles()
    run_feature_ablation_ladder=True,   # adds a 3rd variant: mol_only_classic (chemberta+morgan only -- the old baseline)

    hidden=256, dropout=0.3, pair_hidden=512,

    lr=1e-3, min_lr=1e-6, weight_decay=1e-2, batch_size=1024, eval_batch=4096,
    warmup_epochs=5, max_epochs=400, patience=60, min_delta=2e-4,
    min_epochs=40,        # never early-stop before this (let each run develop)
    ema_beta=0.8,         # smooth noisy val micro-AP for stop decision
    lr_factor=0.5, lr_patience=8, grad_clip=1.0,
    asl_gamma_neg=4.0, asl_gamma_pos=0.0, asl_clip=0.05,
    n_bootstrap=2000,
    use_esm=False,              # ESM-2 embeddings are not used: the UniProt embeddings API
    run_esm_ablation=False,     # returns nothing for these accessions, so the fetch only costs time

    FAST_MODE=bool(RUNTIME.get("fast", False)),   # False = full unattended run (default). True = ~10-min smoke test of the whole pipeline.
    workdir=(globals().get("WORKDIR") or ("/kaggle/working/ddi_work" if os.path.isdir("/kaggle/working") else os.path.expanduser("~/ddi_work"))),
)
if CFG.FAST_MODE:
    CFG.max_epochs=12; CFG.patience=4; CFG.min_epochs=3; CFG.headline_seeds=[0]; CFG.n_bootstrap=500
import os; os.makedirs(CFG.workdir,exist_ok=True)
print("dataset",CFG.dataset,"| splits",CFG.splits,"| FAST",CFG.FAST_MODE)
print("workdir:",CFG.workdir,"(saved to notebook output)" if CFG.workdir.startswith("/kaggle/working") else "")
