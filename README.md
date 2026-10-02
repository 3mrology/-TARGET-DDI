# TARGET-DDI

Code to reproduce the experiments of *TARGET-DDI: Fusing Molecular and Protein-Target Representations for Cold-Start Drug–Drug Interaction Prediction* (A. Shaarawy et al., preprint submitted to Elsevier).

TARGET-DDI predicts the interaction types of a drug pair on DrugBank-DDI (86 types, multi-label) when one or both drugs are absent from training. It combines a molecular branch (ChemBERTa embeddings, Morgan fingerprints, ATC groups and physicochemical descriptors) with a protein-target branch rebuilt from public UniProt, Gene Ontology and Pfam annotations, fused through an availability-conditioned gate.

## Requirements

- Python 3.10 or newer, Linux or macOS.
- A CUDA GPU is strongly recommended. The original runs used one NVIDIA T4 (16 GB). Everything also runs on CPU, but far more slowly.
- Internet access during the first run. The pipeline downloads DrugBank-DDI from TDC, the ChemBERTa weights from Hugging Face, annotations from the UniProt REST API, ATC codes, the EmerGNN partitions from Zenodo and the TWOSIDES splits from GitHub (`git` must be installed).
- About 10 GB of free disk space for data, caches and checkpoints.

## Installation

```bash
git clone <this repository> target-ddi
cd target-ddi
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps PyTDC
```

PyTDC is installed without its dependencies because its pinned requirements conflict with current scientific Python stacks. The modules it needs for DrugBank-DDI are covered by `requirements.txt`, and the pipeline provides stubs for the unrelated single-cell modules PyTDC imports.

The development environment used torch 2.10.0 (CUDA 12.8) and numpy 2.0.2.

## Quick start

Check the installation without downloading anything. This runs the setup stages and a synthetic cold-start self-test, which fails loudly if the target branch cannot beat the molecular-only model:

```bash
python run.py --selftest-only
```

Run a short end-to-end smoke test (few epochs, one seed):

```bash
python run.py --fast --workdir ddi_work_fast --sections main-grid
```

Reproduce everything:

```bash
python run.py --workdir ddi_work
```

A full run takes many hours on a single GPU; the CSMDDI section alone took about four hours on a T4. Every section is crash-safe. Finished units are skipped, an interrupted training run resumes from its last epoch checkpoint, and rerunning the same command continues where it stopped. `--budget-hours N` makes the run stop cleanly before starting a new unit once `N` hours have passed, which is useful on time-limited machines. Progress is written to `ddi_work/STATUS.txt`, and the complete console output to `ddi_work/run_log_*.txt`.

To run selected sections only:

```bash
python run.py --sections main-grid,gate,csmddi
python run.py --help        # lists every section
```

| Section | What it produces | Paper |
|---|---|---|
| `main-grid` | Mol-Classic, Mol-Full, TARGET-DDI and TARGET-DDI-G on S1, S2, S3 with three seeds; coverage report; split sizes; paired-bootstrap intervals; shared-protein stratification | Tables 1, 2, 4, 5, 9, 10; Table 8 (partly, see below) |
| `gate` | Fusion ablation | Table 13 |
| `headtail` | Macro-F1 on frequent and rare interaction types | Table 11 |
| `case-cost` | Pairs corrected by the target branch; training cost | Table 12 |
| `emergnn-folds` | TARGET-DDI on the EmerGNN drug partitions released on Zenodo, plus warm start, and the comparison with the published EmerGNN table | Table 7 |
| `twosides` | TWOSIDES emerging-drug splits | Table 14 |
| `twosides-rescore` | Re-scores the TWOSIDES runs with the paired-negative protocol of the reference implementation | Table 14 |
| `csmddi` | CSMDDI retrained on the S1 to S3 partitions, with its native protein vector and with the TARGET-DDI inputs | Table 4, CSMDDI rows |

Sections that analyse trained models (`headtail`, `case-cost`, `csmddi`) need the `main-grid` results in the same working directory. With the default `--sections all` they run after the grid automatically. Two further sections exist but are not part of `all`: `emergnn-ddiben`, an earlier evaluation on the single EmerGNN split shipped with DDI-Ben, and `seed0-gaps`, which retrains only missing seed-0 grid runs.

The values of Table 6 (DDIMDL, DeepDDI, DNN) are taken from the publication of Deng et al. (2020) and are not recomputed.

## Outputs

All results go to the working directory (default `ddi_work/`):

- `res_*.json`, `pred_*.npz`, `ck_*.pt`, `hist_*.csv`: per-run metrics, test predictions, checkpoints and training curves of the main grid
- `all_results.csv`, `all_runs_raw.csv`: aggregated and per-run main-grid results
- one sub-folder per section (`gate_ablation/`, `headtail/`, `case_study/`, `cost/`, `emergnn_zenodo/`, `twosides/`, `csmddi/`, `paper_tables/`) with its own summary CSV files
- `paper_results/`: every summary table collected in one place, plus a zip of it

## Not covered by this code

The following parts of the paper are not produced by this repository:

- the ESM-G and Both-G rows of Table 4. They need mean-pooled ESM-2 embeddings from the UniProt embeddings endpoint, which returns no data for these accessions at the time of release, so `CFG.use_esm` is off by default
- the context-only DEX-DDI and SIDER-G rows of Table 4
- the Mol-Desc step of the ablation ladder (Table 8); the grid trains Mol-Classic, Mol-Full and TARGET-DDI
- the TWOSIDES annotation-coverage comparison (52% against 91%); `twosides` runs the release's own DrugBank-ID mapping
- the figures, which were drawn separately from the result files

## Reproducibility notes

Protein annotations are retrieved from live UniProt, Gene Ontology and Pfam services, and these resources change between releases. Results of a fresh run can therefore differ slightly from the paper; the paper's numbers were produced with the annotations retrieved in 2026. The pipeline caches every retrieved annotation in the working directory (`drug_targets.json`, `drug_targets_by_cat.json`, `ec_go_class*`, `go_pfam.json`, `atc.json` and the embedding caches), so all later runs that reuse the same working directory see identical inputs. Keep these cache files, or archive them with your results, if you need byte-identical reruns. The `csmddi` section additionally records a fingerprint of the feature matrices and refuses to mix units trained on different annotation snapshots.

Seeds 0, 1 and 2 determine both the drug partitions and the model initialisation. GPU kernels are not bitwise deterministic, so repeated runs on the same partition can differ in the fourth decimal.

## Configuration

All hyperparameters are in `targetddi/stages/s01_config.py` (`CFG`). Changing a setting that affects the features requires bumping `CFG.feature_version`, which keeps old caches from being reused.

## Code layout

```
run.py                     command-line entry point
targetddi/runner.py        executes the stages in order
targetddi/stages/
  s00-s07                  setup: working directory, config, environment, annotation fetching,
                           splits and metrics, molecular features, model, self-test
  s08                      run mode and session budget
  s09                      feature construction and the main grid
  s10-s23                  results, analyses, external benchmarks, CSMDDI, collection
```

The stages run in a single shared namespace, in the order listed in `targetddi/runner.py`, because later stages reuse objects built by earlier ones (configuration, dataset, feature banks and model classes). This keeps the code identical to the version that produced the reported numbers.

## Data sources

DrugBank-DDI through Therapeutics Data Commons (TDC); protein annotations from UniProt, Gene Ontology and Pfam; ATC codes from the DrugBank-derived table of the `dhimmel/drugbank` repository; ChemBERTa-77M-MTR from Hugging Face; the EmerGNN drug partitions from the Zenodo record of Zhang et al. (2023); the TWOSIDES emerging-drug splits from the DDI-Bench repository (LARS-research). Each source has its own license and terms of use, which apply to the downloaded data.

## Citation

If you use this code, please cite the paper (citation details will be added on publication).
