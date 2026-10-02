"""Imports, compatibility shims for PyTDC, device and seeding."""

import importlib.util, sys, types
_need = {"rdkit": "rdkit", "scipy": "scipy", "requests": "requests", "transformers": "transformers",
         "sklearn": "scikit-learn", "torch": "torch", "pandas": "pandas", "tdc": "PyTDC (pip install --no-deps PyTDC)"}
_missing = [pkg for mod, pkg in _need.items() if importlib.util.find_spec(mod) is None]
if _missing:
    raise ImportError("Missing packages: " + ", ".join(_missing) + ". See README, section Installation.")
from rdkit import RDLogger as _RDL; _RDL.DisableLog("rdApp.*")  # silence Morgan/parse spam
import numpy as np
import pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import rankdata
import json,time,math,random,warnings,gc,glob,io,requests,zipfile
warnings.filterwarnings("ignore")
for name,attr in [("tiledbsoma",None),("tdc.multi_pred.single_cell","CellXGeneTemplate"),
                  ("tdc.multi_pred.perturboutcome","PerturbOutcome")]:
    if name not in sys.modules:
        m=types.ModuleType(name)
        if attr: setattr(m,attr,object)
        sys.modules[name]=m
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
def set_seed(s): random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
print("torch",torch.__version__,"| numpy",np.__version__,"| device",DEVICE)
try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj.to_string() if hasattr(obj, "to_string") else obj)
