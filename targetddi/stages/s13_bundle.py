"""Zips results and checkpoints into a bundle."""
import zipfile, glob, time as _t

stamp = _t.strftime("%Y%m%d_%H%M%S")
results_zip = f"{CFG.workdir}/results_bundle_{stamp}.zip"
with zipfile.ZipFile(results_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for pattern in ("res_*.json", "all_results.csv", "all_runs_raw.csv", "hist_*.csv"):
        for f in glob.glob(f"{CFG.workdir}/{pattern}"):
            z.write(f, arcname=os.path.basename(f))
print(f"results bundle: {results_zip}  ({os.path.getsize(results_zip)/1e6:.1f} MB)")

SAVE_CHECKPOINTS_TOO = False   # set True if you want ck_*.pt bundled as well (can be large)
if SAVE_CHECKPOINTS_TOO:
    ckpt_zip = f"{CFG.workdir}/checkpoints_bundle_{stamp}.zip"
    with zipfile.ZipFile(ckpt_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in glob.glob(f"{CFG.workdir}/ck_*.pt"):
            z.write(f, arcname=os.path.basename(f))
    print(f"checkpoint bundle: {ckpt_zip}  ({os.path.getsize(ckpt_zip)/1e6:.1f} MB)")

print("\nDownload via the Kaggle Output pane, or commit this session's output as a "
      "Kaggle dataset to carry results/checkpoints into your next session cleanly.")
