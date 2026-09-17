# Employee Attrition Analysis

A classification + dashboard project on the IBM HR Analytics dataset
(1,470 employees, 35 raw features, 16.1% attrition rate).

## Structure

- `data_pipeline.py` — cleaning, feature engineering, encoding. Run standalone
  to regenerate `processed_attrition.csv`.
- `model_training.py` — trains Logistic Regression, Random Forest, and XGBoost;
  handles class imbalance; tunes the decision threshold; saves the best model
  and metrics.
- `app.py` — Streamlit dashboard (KPIs, driver analysis, model comparison,
  individual employee risk scorer).

## Running it

```bash
pip install -r requirements.txt
python model_training.py   # trains models, writes best_model.joblib etc.
streamlit run app.py
```

## Honest notes on this project (read before presenting it)

**The dataset is imbalanced (84% stayed / 16% left).** Accuracy is reported
nowhere as a headline metric because it's misleading here — a model that
predicts "no one leaves" scores ~84% while being useless. Precision/recall/F1
on the minority ("Left") class and ROC-AUC are used instead, and the decision
threshold is tuned per-model rather than left at the sklearn default of 0.5.

**Model selection involved a real trade-off, not a clean win.** XGBoost was
selected as the deployed model because it had the best F1 on the minority
class (0.513), but Logistic Regression had the higher ROC-AUC (0.804 vs
0.773). If someone asks "why not Logistic Regression," the honest answer is
the selection metric (F1 on `Left`) was chosen upfront and XGBoost won on it —
not that XGBoost is unambiguously the better model.

**Recall on people who actually leave tops out well below 1.0** (roughly
0.6–0.8 depending on model and threshold). This means the model will still
miss real resignations. Frame it to reviewers as a prioritization tool for
HR follow-up, not a certainty machine.

**"Drivers of attrition" are associations, not causes.** Feature importance
comes from a Random Forest trained on observational data, not a randomized
experiment. The dashboard's Driver tab says this explicitly — don't strip
that caveat out if you present this, since claiming causation from feature
importance is a common and easily-caught mistake in interviews.

## Dataset

Standard IBM HR Analytics Employee Attrition dataset. Constant columns
(`EmployeeCount`, `StandardHours`, `Over18`) and the row-ID column
(`EmployeeNumber`) are dropped in the pipeline since they carry no signal.
