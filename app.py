"""
app.py
------
Streamlit dashboard for HR: attrition KPIs, driver analysis, and an
individual employee risk scorer backed by the trained model.

Run with: streamlit run app.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from data_pipeline import load_raw, clean, engineer_features, encode

st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

# ------------------------------------------------------------------
# Data & model loading (cached so the app doesn't retrain on every click)
# ------------------------------------------------------------------

@st.cache_data
def load_data():
    raw = load_raw("Employee_Attrition.csv")
    cleaned = clean(raw)
    return raw, cleaned


@st.cache_resource
def load_model_artifacts():
    model = joblib.load("best_model.joblib")
    scaler = joblib.load("scaler.joblib")
    feature_columns = joblib.load("feature_columns.joblib")
    with open("model_results.json") as f:
        results = json.load(f)
    return model, scaler, feature_columns, results


raw_df, df = load_data()
model, scaler, feature_columns, results = load_model_artifacts()

# ------------------------------------------------------------------
# Header + top-line KPIs
# ------------------------------------------------------------------

st.title("Employee Attrition Analysis")
st.caption(
    "Portfolio project — IBM HR Analytics dataset. "
    "Predictions are model-based associations, not confirmed causes of attrition."
)

total_employees = len(df)
attrition_count = int(df["Attrition"].sum())
attrition_rate = df["Attrition"].mean()
avg_income = df["MonthlyIncome"].mean()
avg_tenure = df["YearsAtCompany"].mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Employees", f"{total_employees:,}")
k2.metric("Attrition Count", f"{attrition_count}")
k3.metric("Attrition Rate", f"{attrition_rate:.1%}")
k4.metric("Avg. Tenure (yrs)", f"{avg_tenure:.1f}")

st.divider()

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------

tab_overview, tab_drivers, tab_model, tab_scorer = st.tabs(
    ["Overview", "Attrition Drivers", "Model Performance", "Employee Risk Scorer"]
)

# --- Overview tab ---------------------------------------------------
with tab_overview:
    col1, col2 = st.columns(2)

    with col1:
        dept_attr = (
            raw_df.groupby("Department")["Attrition"]
            .apply(lambda s: (s == "Yes").mean())
            .reset_index(name="AttritionRate")
        )
        fig = px.bar(
            dept_attr,
            x="Department",
            y="AttritionRate",
            title="Attrition Rate by Department",
            text_auto=True,
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width='stretch')

        ot_attr = (
            raw_df.groupby("OverTime")["Attrition"]
            .apply(lambda s: (s == "Yes").mean())
            .reset_index(name="AttritionRate")
        )
        fig2 = px.bar(
            ot_attr,
            x="OverTime",
            y="AttritionRate",
            title="Attrition Rate by OverTime Status",
            text_auto=True,
        )
        fig2.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig2, width='stretch')

    with col2:
        fig3 = px.histogram(
            raw_df,
            x="Age",
            color="Attrition",
            barmode="overlay",
            nbins=20,
            title="Age Distribution by Attrition",
            opacity=0.6,
        )
        st.plotly_chart(fig3, width='stretch')

        fig4 = px.box(
            raw_df,
            x="Attrition",
            y="MonthlyIncome",
            title="Monthly Income by Attrition",
            color="Attrition",
        )
        st.plotly_chart(fig4, width='stretch')

    st.subheader("Job Role Breakdown")
    role_attr = (
        raw_df.groupby("JobRole")["Attrition"]
        .agg(Headcount="count", AttritionRate=lambda s: (s == "Yes").mean())
        .reset_index()
        .sort_values("AttritionRate", ascending=False)
    )
    fig5 = px.bar(
        role_attr,
        x="JobRole",
        y="AttritionRate",
        title="Attrition Rate by Job Role",
        text_auto=True,
        hover_data=["Headcount"],
    )
    fig5.update_yaxes(tickformat=".0%")
    fig5.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig5, width='stretch')

# --- Drivers tab -----------------------------------------------------
with tab_drivers:
    st.warning(
        "These are factors **associated with** attrition in this dataset, "
        "based on feature importance from a trained model. This is "
        "observational data, not a controlled experiment — treat this as "
        "a starting point for HR investigation, not a causal explanation."
    )

    importance = pd.Series(results["feature_importance"]).sort_values(ascending=False).head(15)
    fig_imp = px.bar(
        x=importance.values,
        y=importance.index,
        orientation="h",
        title="Top 15 Features Associated with Attrition (Random Forest importance)",
        labels={"x": "Relative Importance", "y": "Feature"},
    )
    fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_imp, width='stretch')

    st.subheader("Satisfaction Scores by Attrition")
    sat_cols = ["JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance", "RelationshipSatisfaction"]
    sat_melt = raw_df.melt(
        id_vars="Attrition", value_vars=sat_cols, var_name="SatisfactionType", value_name="Score"
    )
    fig_sat = px.box(sat_melt, x="SatisfactionType", y="Score", color="Attrition")
    st.plotly_chart(fig_sat, width='stretch')

# --- Model performance tab -------------------------------------------
with tab_model:
    st.subheader("Model Comparison")
    comp_df = pd.DataFrame(results["results"])[
        ["name", "roc_auc", "precision_minority", "recall_minority", "f1_minority", "threshold"]
    ].rename(
        columns={
            "name": "Model",
            "roc_auc": "ROC-AUC",
            "precision_minority": "Precision (Left)",
            "recall_minority": "Recall (Left)",
            "f1_minority": "F1 (Left)",
            "threshold": "Decision Threshold",
        }
    )
    st.dataframe(comp_df.style.format({c: "{:.3f}" for c in comp_df.columns if c != "Model"}))

    st.info(
        f"**Deployed model: {results['best_model']}** — selected for best F1-score "
        f"on the minority ('Left') class, not raw accuracy, since accuracy is "
        f"misleading on this ~84/16 imbalanced dataset. Note this is a trade-off, "
        f"not a clean win: Logistic Regression scored a higher ROC-AUC overall."
    )

    st.markdown(
        """
        **Why not just report accuracy?** A model predicting "No one leaves"
        for every employee would score ~84% accuracy while catching zero
        actual resignations. Precision/recall on the "Left" class is what
        actually tells you whether the model is useful for HR intervention.

        **Limitation to be upfront about:** recall on the minority class
        tops out well below 1.0 — the model misses a real share of people
        who do leave. Treat its output as a prioritization signal for
        HR follow-up, not a certainty.
        """
    )

# --- Risk scorer tab ---------------------------------------------------
with tab_scorer:
    st.subheader("Score an Employee's Attrition Risk")
    st.caption("Pick an existing employee profile or adjust values to explore risk sensitivity.")

    row_idx = st.selectbox(
        "Load an existing employee (by row index)", options=range(len(raw_df)), index=0
    )
    sample_raw = raw_df.iloc[[row_idx]].copy()

    colA, colB, colC = st.columns(3)
    with colA:
        age = st.slider("Age", 18, 60, int(sample_raw["Age"].iloc[0]))
        income = st.number_input("Monthly Income", 1000, 20000, int(sample_raw["MonthlyIncome"].iloc[0]))
        overtime = st.selectbox("OverTime", ["Yes", "No"], index=0 if sample_raw["OverTime"].iloc[0] == "Yes" else 1)
    with colB:
        years_at_company = st.slider("Years at Company", 0, 40, int(sample_raw["YearsAtCompany"].iloc[0]))
        job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, int(sample_raw["JobSatisfaction"].iloc[0]))
        work_life = st.slider("Work-Life Balance (1-4)", 1, 4, int(sample_raw["WorkLifeBalance"].iloc[0]))
    with colC:
        distance = st.slider("Distance From Home", 1, 30, int(sample_raw["DistanceFromHome"].iloc[0]))
        total_working_years = st.slider(
            "Total Working Years", 0, 40, int(sample_raw["TotalWorkingYears"].iloc[0])
        )
        years_since_promo = st.slider(
            "Years Since Last Promotion", 0, 15, int(sample_raw["YearsSinceLastPromotion"].iloc[0])
        )

    # Build a modified record, run it through the same pipeline the model was trained on
    modified = sample_raw.copy()
    modified["Age"] = age
    modified["MonthlyIncome"] = income
    modified["OverTime"] = overtime
    modified["YearsAtCompany"] = years_at_company
    modified["JobSatisfaction"] = job_satisfaction
    modified["WorkLifeBalance"] = work_life
    modified["DistanceFromHome"] = distance
    modified["TotalWorkingYears"] = total_working_years
    modified["YearsSinceLastPromotion"] = years_since_promo

    cleaned_row = clean(modified)
    engineered_row = engineer_features(cleaned_row)
    encoded_row = encode(engineered_row)

    # Align columns to what the model expects (missing dummy columns -> 0)
    aligned = pd.DataFrame(columns=feature_columns)
    aligned = pd.concat([aligned, encoded_row], axis=0)
    aligned = aligned.reindex(columns=feature_columns, fill_value=0)
    aligned = aligned.fillna(0)

    model_key = results["best_model_key"]
    if model_key == "logistic_regression":
        X_input = scaler.transform(aligned)
    else:
        X_input = aligned

    proba = model.predict_proba(X_input)[:, 1][0]
    threshold = results["threshold"]
    prediction = "Likely to Leave" if proba >= threshold else "Likely to Stay"

    st.divider()
    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("Predicted Attrition Risk", f"{proba:.1%}")
        st.metric("Prediction", prediction)
    with r2:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={"text": "Attrition Risk (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkred" if proba >= threshold else "darkgreen"},
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": threshold * 100,
                    },
                },
            )
        )
        st.plotly_chart(fig_gauge, width='stretch')

    st.caption(
        f"Decision threshold: {threshold:.1%} (tuned to maximize F1 on the "
        f"'Left' class, not the default 50%)."
    )
