# XAI-DIF — Explainable AI-Driven Decision Intelligence Framework for Software Project Management

Reproducible implementation and evaluation pipeline for the paper
**"Explainable AI-Driven Decision Intelligence Framework for Software Project Management"**
(M. H. Qureshi and M. F. Nasim).

XAI-DIF is a three-layer framework that goes beyond prediction:

1. **Predictive Layer** — ensemble models (Random Forest, Gradient Boosting, XGBoost) predict sprint delivery risk and cycle time.
2. **Explainability Layer** — exact TreeSHAP values (`shap`) explain *why* each sprint is flagged; LIME (`lime`) is fitted as an independent cross-check.
3. **Decision Intelligence Layer** — a PMBOK-grounded rule engine turns the top risk drivers into prioritized, actionable recommendations.

This repository reproduces every quantitative result reported in the paper (Sections 6.1–6.7).

## Repository layout

```
xai-diff-code/
├── src/
│   └── run_evaluation.py        # full pipeline: predict -> explain -> recommend -> evaluate
├── notebooks/
│   └── XAI-DIF_PoC.ipynb         # step-by-step proof-of-concept notebook
├── data/
│   └── README.md                # how to obtain the dataset (not redistributed here)
├── reference_results.json       # expected outputs, for comparison
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

## Dataset

The evaluation uses the publicly available, **synthetically generated** *Enterprise Agile Sprint Delivery Risk*
dataset on Kaggle (15,000 sprints, 38 features):

- https://www.kaggle.com/datasets/revanthtambisetty/enterprise-agile-sprint-performance-delivery-risk

Download `enterprise_agile_sprint_risk.csv` and place it in `data/` (see `data/README.md`).
The dataset is **not redistributed** in this repository — please obtain it from the link above under its Kaggle licence.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

## Run

```bash
# with the CSV placed at data/enterprise_agile_sprint_risk.csv
python src/run_evaluation.py

# or point to it explicitly
python src/run_evaluation.py /path/to/enterprise_agile_sprint_risk.csv
```

Results are written to `outputs/results.json` and figures (`roc.png`, `global_importance.png`,
`h3_multi_vs_single.png`, `cycle_time_regression.png`, `ablation.png`, `recommendation_relevance.png`,
`local_explanation.png`)
to `outputs/`. Expected values are in `reference_results.json`.

## Key results reproduced (seed = 42)

| Metric | Value |
|---|---|
| Sprint-failure classifier (RF) | Precision 0.738 · Recall 0.775 · F1 0.756 · ROC-AUC 0.787 |
| Gradient Boosting / XGBoost (F1, ROC-AUC) | 0.754, 0.79 / 0.751, 0.789 |
| Confusion matrix (test set) | TN 1076 · FP 576 · FN 473 · TP 1625 (predicted-failed = 2201) |
| H3 all signal groups vs planning-only (10-fold F1) | 0.755 vs 0.695 · Cliff's δ = 1.00 · paired-gain 95% CI [0.052, 0.068] |
| Cycle-time regression | MAE 0.96 d · RMSE 1.2 d · R² 0.601 |
| Explanation faithfulness (n = 40) | LIME local R² 0.422 (range 0.255–0.58) · SHAP–LIME Spearman ρ = 0.839 ± 0.049 |
| Relevance@3 vs recorded failure cause (2,098 failed sprints) | per-sprint SHAP 82.0% · fixed top-3 by mean \|SHAP\| 63.8% · fixed top-3 by permutation 91.0% · random 31.0% |
| …causes outside the mean-\|SHAP\| top-3 (n = 759) | SHAP 73.4% vs fixed list 0% |
| …causes outside the permutation top-3 (n = 188) | SHAP 90.4% vs fixed list 0% |
| What-if sensitivity, 2,201 flagged sprints (H2, model-based) | below threshold after top-1 / top-2: SHAP 19.1% / 33.3% · fixed \|SHAP\| list 8.1% / 13.4% · fixed permutation list 8.1% / 14.1% · prediction-only 0% |

**Caveats.** The dataset is synthetic and its recorded failure cause is generated from the same features (≈91% predictable from them), so relevance measures agreement with the generator, not real-world causation. The what-if analysis re-scores perturbed inputs with the same associational model; it measures model sensitivity, not the effect of management actions.

(The Wilcoxon p for H3 is 0.00195, the smallest two-sided value attainable with ten paired folds — see the paper, Section 6.2, for why Cliff's δ and the CI are the informative statistics.)

## Notebook

`notebooks/XAI-DIF_PoC.ipynb` walks through the pipeline interactively. The notebook falls back to `scikit-learn`
equivalents when `xgboost`, `shap` or `lime` are missing; `src/run_evaluation.py` requires all three (see `requirements.txt`)
and is the source of every number in the paper.

## Changes in v1.1

- Explanations now use exact TreeSHAP (`shap`) and LIME (`lime`); v1.0 used permutation-style approximations.
- XGBoost added to the model comparison.
- Recommendation relevance and the what-if analysis are reported against fixed top-3 and random driver baselines.
- The "ablation" is now described as a model-based what-if sensitivity analysis; H3 is described as a feature-group comparison.

## Citing

If you use this software, please cite both the paper and this repository (see `CITATION.cff`).
Archived on Zenodo: https://doi.org/10.5281/zenodo.23841471 (v1.1.0, used for the paper; v1.0: 10.5281/zenodo.23840638).

## License

MIT — see `LICENSE`.
