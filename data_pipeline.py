"""
data_pipeline.py
-----------------
Loads the raw HR attrition CSV and produces a cleaned, model-ready dataframe.

Design decisions (documented, not hidden):
- Drops EmployeeCount, StandardHours, Over18: zero variance, zero information.
- Drops EmployeeNumber: a row ID, not a feature. Leaving it in invites a model
  to pick up spurious correlation with row order / hiring cohort.
- Ordinal survey fields (1-4 satisfaction/involvement scales) are kept as
  integers, not one-hot encoded, because they already have a meaningful order.
- Nominal categoricals (Department, JobRole, MaritalStatus, etc.) are one-hot
  encoded because there's no ordinal relationship to preserve.
- Binary Yes/No fields are mapped to 1/0 explicitly rather than left to
  get_dummies, so the resulting column names are predictable.
"""

import pandas as pd
import numpy as np

DROP_COLS = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]

BINARY_MAPS = {
    "Attrition": {"Yes": 1, "No": 0},
    "OverTime": {"Yes": 1, "No": 0},
    "Gender": {"Male": 1, "Female": 0},
}

NOMINAL_COLS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "JobRole",
    "MaritalStatus",
]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Sanity checks — fail loudly rather than silently proceeding on bad data
    assert df.isnull().sum().sum() == 0, "Unexpected nulls in source data"

    existing_drops = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=existing_drops)

    for col, mapping in BINARY_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a small number of derived features that are plausible attrition
    signals and not just re-encodings of existing columns."""
    df = df.copy()

    # Tenure ratio: how much of their total working life has been at THIS company.
    # A 35-year-old with 15 years experience and 1 year here reads differently
    # than a 35-year-old with 15 years experience and 14 years here.
    df["TenureRatio"] = df["YearsAtCompany"] / df["TotalWorkingYears"].replace(0, 1)

    # Income relative to job level — flags people underpaid for their level,
    # which raw MonthlyIncome alone can't distinguish from "junior and low-paid".
    level_median_income = df.groupby("JobLevel")["MonthlyIncome"].transform("median")
    df["IncomeVsLevelMedian"] = df["MonthlyIncome"] / level_median_income

    # Years since promotion relative to tenure — stagnation signal independent
    # of how long they've simply existed at the company.
    df["PromotionStagnation"] = df["YearsSinceLastPromotion"] / df["YearsAtCompany"].replace(0, 1)

    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    existing_nominal = [c for c in NOMINAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=existing_nominal, drop_first=True)
    return df


def build_dataset(path: str) -> pd.DataFrame:
    df = load_raw(path)
    df = clean(df)
    df = engineer_features(df)
    df = encode(df)
    return df


if __name__ == "__main__":
    df = build_dataset("Employee_Attrition.csv")
    print("Final shape:", df.shape)
    print("Attrition rate: {:.1%}".format(df["Attrition"].mean()))
    df.to_csv("processed_attrition.csv", index=False)
    print("Saved processed_attrition.csv")
