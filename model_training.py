"""
model_training.py
------------------
Trains and compares classifiers for employee attrition prediction.

Why this isn't "just call .fit() and report accuracy":
- The target is imbalanced (~84% No / ~16% Yes). Accuracy on this dataset is
  a near-meaningless number: a model that always predicts "No" scores ~84%
  while catching zero resigners. We report recall, precision, F1, and
  ROC-AUC for the minority (Yes) class, and we pick the operating threshold
  deliberately rather than defaulting to 0.5.
- class_weight="balanced" (or scale_pos_weight for XGBoost) is used instead
  of naive oversampling/undersampling, to avoid duplicating/discarding real
  rows in a dataset this small (1470 rows).
- Feature importances are saved and exposed, but labeled as "associated
  with" attrition, not "causes of" attrition — this is observational data,
  not a randomized experiment, so causal language is not earned.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
)
from xgboost import XGBClassifier

from data_pipeline import build_dataset

RANDOM_STATE = 42


def split_data(df: pd.DataFrame):
    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def best_threshold_for_f1(y_true, y_proba):
    """Scans thresholds and returns the one maximizing F1 on the minority class,
    instead of blindly using 0.5 (which under-predicts the minority class when
    classes are imbalanced, even with class_weight='balanced')."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1])  # last point has no corresponding threshold
    return thresholds[best_idx], f1s[best_idx]


def evaluate(name, model, X_test, y_test, scaler=None):
    X_eval = scaler.transform(X_test) if scaler is not None else X_test
    y_proba = model.predict_proba(X_eval)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    threshold, f1 = best_threshold_for_f1(y_test, y_proba)
    y_pred_tuned = (y_proba >= threshold).astype(int)

    report = classification_report(y_test, y_pred_tuned, output_dict=True)
    cm = confusion_matrix(y_test, y_pred_tuned).tolist()

    print(f"\n--- {name} ---")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"Tuned threshold: {threshold:.3f} (maximizes minority-class F1)")
    print(classification_report(y_test, y_pred_tuned))

    return {
        "name": name,
        "roc_auc": roc_auc,
        "threshold": float(threshold),
        "f1_minority": report["1"]["f1-score"],
        "precision_minority": report["1"]["precision"],
        "recall_minority": report["1"]["recall"],
        "confusion_matrix": cm,
    }


def train_all(csv_path: str):
    df = build_dataset(csv_path)
    X_train, X_test, y_train, y_test = split_data(df)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    results = []
    models = {}

    # 1. Logistic Regression — interpretable baseline, coefficients are
    #    directly readable as direction + rough magnitude of association.
    log_reg = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    )
    log_reg.fit(X_train_scaled, y_train)
    results.append(evaluate("Logistic Regression", log_reg, X_test, y_test, scaler))
    models["logistic_regression"] = log_reg

    # 2. Random Forest — captures nonlinear interactions, no scaling needed.
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest", rf, X_test, y_test))
    models["random_forest"] = rf

    # 3. XGBoost — usually the strongest tabular performer; scale_pos_weight
    #    handles imbalance the way class_weight does for sklearn models.
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    xgb.fit(X_train, y_train)
    results.append(evaluate("XGBoost", xgb, X_test, y_test))
    models["xgboost"] = xgb

    # Pick best by minority-class F1 (the metric that actually reflects
    # "did we find the people who left", which is the point of this project)
    best = max(results, key=lambda r: r["f1_minority"])
    print(f"\n>>> Best model: {best['name']} (F1={best['f1_minority']:.3f})")

    best_key = {
        "Logistic Regression": "logistic_regression",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
    }[best["name"]]
    best_model = models[best_key]

    # Feature importance / coefficients for the dashboard.
    # Always from the tree models when available — more reliable for mixed
    # numeric/dummy features than raw logistic coefficients on unscaled dummies.
    importance_model = models["random_forest"]
    importances = pd.Series(
        importance_model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    # Persist everything the dashboard needs
    joblib.dump(best_model, "best_model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(list(X_train.columns), "feature_columns.joblib")

    with open("model_results.json", "w") as f:
        json.dump(
            {
                "results": results,
                "best_model": best["name"],
                "best_model_key": best_key,
                "threshold": best["threshold"],
                "feature_importance": importances.to_dict(),
            },
            f,
            indent=2,
        )

    print("\nSaved: best_model.joblib, scaler.joblib, feature_columns.joblib, model_results.json")
    return results, best_model


if __name__ == "__main__":
    train_all("Employee_Attrition.csv")
