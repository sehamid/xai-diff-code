# Dataset

This project uses the publicly available, synthetically generated
**Enterprise Agile Sprint Delivery Risk** dataset (15,000 sprints, 38 features):

https://www.kaggle.com/datasets/revanthtambisetty/enterprise-agile-sprint-performance-delivery-risk

## How to get it

1. Download the dataset from the Kaggle page above (requires a free Kaggle account).
2. Extract `enterprise_agile_sprint_risk.csv` into this `data/` folder.
3. Run `python src/run_evaluation.py` from the repository root.

The dataset is not redistributed in this repository; please obtain it from Kaggle
under its own licence. It is a synthetic benchmark that simulates enterprise Agile
sprints (it is not a record of real proprietary projects).
