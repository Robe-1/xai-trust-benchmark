# SHAP vs LIME trust benchmark (fidelity, stability, sparsity)

Code, configuration and result tables for the paper **"Quantitative Evaluation of Explainable AI Methods for Medical
Decision Support: A Statistically Validated Comparison of SHAP and LIME"** (ITT 2026).

## What it does
For two public datasets (Breast Cancer Wisconsin Diagnostic; Pima diabetes from R's MASS package, `Pima.tr` + `Pima.te`)
and two tree ensembles (random forest, XGBoost) it computes, for **every** test instance:
* **Fidelity**: deletion AOPC (mean replacement, ordered by |attribution|) with a random-order control and sensitivity to K.
* **Stability**: Pearson / Spearman / top-k Jaccard between explanations of original and noise-perturbed inputs
  (1/5/10 % of the training std, D independent draws, SHAP and LIME on identical perturbed inputs), plus a
  **zero-noise LIME self-consistency** level (same input, new seeds).
* **Sparsity**: features needed for 70/80/90 % of L1-normalised attribution mass.
* Paired Wilcoxon tests with Holm correction, bootstrap CIs, LIME `num_samples` in {1000, 2000, 5000},
  and (optionally) 10 additional stratified splits.

## Run (local machine or Google Colab)
```bash
pip install -r requirements.txt          # in Colab: pip install lime==0.2.0.1 shap xgboost statsmodels
python xai_trust_benchmark.py all --quick --out quick            # ~1-2 min smoke test (not for reporting)
python xai_trust_benchmark.py all --repeats 0 --out my_results   # main experiment only
python xai_trust_benchmark.py all --out my_results               # full run incl. 10 repeated splits
```
Run time depends on the machine: about 30 minutes on a 2-core workstation for the full run, several hours on a free
Colab instance. `--repeats 0` skips the 10 additional splits, which only feed the robustness paragraph.
Sub-commands: `run`, `analyze`, `figures`, `all`. Outputs are CSV tables and PNG figures.

All randomness is seeded and TreeSHAP is deterministic. Results are reproducible up to library versions; the versions
used for the paper are in `results/run_config.json` (numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, xgboost 3.4.1, shap 0.53.0, lime 0.2.0.1).
Small numerical differences with other versions are expected; conclusions should be compared by direction and significance.

## Contents
* `xai_trust_benchmark.py`: experiment, statistics and tables. `figures.py`: figures.
* `data/`: `Pima.tr.csv`, `Pima.te.csv` (R package MASS via Rdatasets; original source Smith et al., 1988, NIDDK).
  Breast Cancer Wisconsin is loaded from scikit-learn (UCI repository; Wolberg, Street and Mangasarian).
* `results/`: the tables and figures reported in the paper, plus `run_log.txt`.

## License and citation
Code: MIT (see `LICENSE`). If you use this code, please cite the paper above.
