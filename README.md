# XAI-DIF — Explainable AI-Driven Decision Intelligence Framework for Software Project Management

Reproducible implementation and evaluation pipeline for the paper
**"Explainable AI-Driven Decision Intelligence Framework for Software Project Management"**
(M. H. Qureshi and M. F. Nasim).

XAI-DIF is a three-layer framework that goes beyond prediction:

1. **Predictive Layer** — ensemble models (Random Forest, Gradient Boosting) predict sprint delivery risk and cycle time.
2. **Explainability Layer** — permutation / SHAP-style attributions and a local surrogate explain *why* each sprint is flagged, with faithfulness metrics.
3. **Decision Intelligence Layer** — a PMBOK-grounded rule engine turns the top risk drivers into prioritized, actionable recommendations.

This repository reproduces every quantitative result reported in the paper (Sections 6.1–6.7).

## Repository layout

```
xai-dif/
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
`h3_multi_vs_single.png`, `cycle_time_regression.png`, `ablation.png`, `recommendation_relevance.png`)
to `outputs/`. Expected values are in `reference_results.json`.

## Key results reproduced (seed = 42)

| Metric | Value |
|---|---|
| Sprint-failure classifier (RF) | Precision 0.738 · Recall 0.775 · F1 0.756 · ROC-AUC 0.787 |
| Confusion matrix (test set) | TN 1076 · FP 576 · FN 473 · TP 1625 (predicted-failed = 2201) |
| H3 multi-source vs planning-only (10-fold F1) | 0.755 vs 0.695 · Cliff's δ = 1.00 · paired-gain 95% CI [0.052, 0.068] |
| Cycle-time regression | MAE 0.96 d · RMSE 1.2 d · R² 0.601 |
| Explanation faithfulness (H1) | local fidelity R² = 0.91 · SHAP/surrogate rank agreement ρ = 0.70 |
| Recommendation relevance vs recorded failure cause | 88% overall (across 2,098 failed test sprints) |
| Decision-layer ablation (H2) | 0% → 18% (top-1) → 30% (top-2) predicted-safe |

(The Wilcoxon p for H3 is 0.00195, the smallest two-sided value attainable with ten paired folds — see the paper, Section 6.2, for why Cliff's δ and the CI are the informative statistics.)

## Notebook

`notebooks/XAI-DIF_PoC.ipynb` walks through the pipeline interactively. It prefers `xgboost`, `shap`,
and `lime` when installed and falls back to `scikit-learn` equivalents otherwise, so it runs in any environment.

## Citing

If you use this software, please cite both the paper and this repository (see `CITATION.cff`).
A Zenodo DOI will be added here on first release.

## License

MIT — see `LICENSE`.
