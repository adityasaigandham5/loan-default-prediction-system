import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import joblib
import json

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Slice Loan Risk Engine",
    page_icon="🏠",
    layout="wide"
)

# ---------------------------------------------------
# Load Models
# ---------------------------------------------------
@st.cache_resource
def load_models():

    return {

        'lgbm':
            joblib.load("../models/loan_default_model.pkl"),

        'cox':
            joblib.load("../models/cox_model.pkl"),

        'cox_sc':
            joblib.load("../models/cox_scaler.pkl"),

        'le_emp':
            joblib.load("../models/le_emp.pkl"),

        'le_city':
            joblib.load("../models/le_city.pkl"),

        'le_type':
            joblib.load("../models/le_type.pkl"),

        'lgbm_m':
            json.load(open("../models/lgbm_metrics.json")),

        'cox_m':
            json.load(open("../models/cox_metrics.json")),
    }

M = load_models()

# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.markdown(
"""
<div style="
background:linear-gradient(135deg,#1E3A5F,#1D4ED8);
padding:22px;
border-radius:12px;
margin-bottom:18px">

<h1 style="color:white;margin:0">
Slice Home & Vehicle Loan Risk Engine
</h1>

<p style="color:#93C5FD;margin:4px 0 0 0">
LightGBM Default Prediction + Cox Survival Analysis
</p>

</div>
""",
unsafe_allow_html=True
)

# ---------------------------------------------------
# Metrics
# ---------------------------------------------------
c1,c2,c3,c4 = st.columns(4)

c1.metric(
    "LightGBM AUC-ROC",
    M['lgbm_m']['auc_roc']
)

c2.metric(
    "Cox C-index",
    M['cox_m']['c_index']
)

c3.metric(
    "Default Rate",
    f"{M['lgbm_m']['default_rate']*100:.1f}%"
)

c4.metric(
    "Training Samples",
    f"{M['cox_m']['n_train']:,}"
)

st.markdown("---")

st.subheader("Enter Loan Application")

# ---------------------------------------------------
# Input Layout
# ---------------------------------------------------
c1,c2,c3 = st.columns(3)

# -------------------------
# Column 1
# -------------------------
with c1:

    loan_type = st.selectbox(
        "Loan Type",
        ["Home","Vehicle"]
    )

    loan_amt = st.number_input(
        "Loan Amount (INR)",
        100000,
        10000000,
        3500000,
        step=50000
    )

    prop_val = st.number_input(
        "Property Value (INR)",
        200000,
        20000000,
        5000000,
        step=100000
    )

    tenure_mo = st.selectbox(
        "Tenure (months)",
        [60,120,180,240,300,360]
        if loan_type=="Home"
        else [12,24,36,48,60,72]
    )

# -------------------------
# Column 2
# -------------------------
with c2:

    cibil = st.slider(
        "CIBIL Score",
        300,
        900,
        720
    )

    income_mo = st.number_input(
        "Monthly Income (INR)",
        15000,
        500000,
        85000,
        step=5000
    )

    employment = st.selectbox(
        "Employment",
        ["Salaried","Self-Employed","Business"]
    )

    emp_years = st.slider(
        "Employment Years",
        0,
        30,
        5
    )

# -------------------------
# Column 3
# -------------------------
with c3:

    interest = st.slider(
        "Interest Rate (%)",
        7.0,
        15.0,
        8.5,
        0.1
    )

    delinq = st.number_input(
        "Delinquency History",
        0,
        5,
        0
    )

    co_app = st.selectbox(
        "Co-Applicant",
        [("No",0),("Yes",1)],
        format_func=lambda x:x[0]
    )

    city_tier = st.selectbox(
        "City Tier",
        ["Tier1","Tier2","Tier3"]
    )

# ---------------------------------------------------
# Prediction Button
# ---------------------------------------------------
if st.button(
    "ASSESS LOAN RISK",
    type="primary",
    use_container_width=True
):

    # -----------------------------------------------
    # EMI + Ratios
    # -----------------------------------------------
    emi_rate = interest / (12 * 100)

    emi = (
        loan_amt * emi_rate * (1+emi_rate)**tenure_mo /
        ((1+emi_rate)**tenure_mo - 1)
    )

    ltv = loan_amt / prop_val

    emi_inc = emi / income_mo

    # -----------------------------------------------
    # Encoders
    # -----------------------------------------------
    emp_enc = int(
        M['le_emp'].transform([employment])[0]
    )

    city_enc = int(
        M['le_city'].transform([city_tier])[0]
    )

    type_enc = int(
        M['le_type'].transform([loan_type])[0]
    )

    # -----------------------------------------------
    # Feature DataFrame
    # -----------------------------------------------
    X = pd.DataFrame([{

        'age':35,

        'income_mo':income_mo,

        'emp_enc':emp_enc,

        'emp_years':emp_years,

        'cibil':cibil,

        'loan_amt':loan_amt,

        'ltv':ltv,

        'tenure_mo':tenure_mo,

        'interest_rate':interest,

        'emi':emi,

        'emi_income':emi_inc,

        'delinq_hist':delinq,

        'num_loans':1,

        'co_applicant':co_app[1],

        'city_enc':city_enc,

        'type_enc':type_enc,

        'loan_income_ratio':
            loan_amt/(income_mo*12+1),

        'high_ltv':
            int(ltv>0.85),

        'high_emi_stress':
            int(emi_inc>0.40),

        'bad_cibil':
            int(cibil<650),

        'good_cibil':
            int(cibil>750),

        'has_delinquency':
            int(delinq>0),

        'age_income_ratio':
            35/(income_mo+1),

    }])

    # -----------------------------------------------
    # Prediction
    # -----------------------------------------------
    prob = float(
        M['lgbm'].predict_proba(X)[0][1]
    )

    # -----------------------------------------------
    # Risk Logic
    # -----------------------------------------------
    risk = (
        "HIGH RISK"
        if prob > 0.20
        else "MEDIUM RISK"
        if prob > 0.08
        else "LOW RISK"
    )

    color = (
        "red"
        if prob > 0.20
        else "orange"
        if prob > 0.08
        else "green"
    )

    dec = (
        "REJECT"
        if prob > 0.20
        else "REVIEW"
        if prob > 0.08
        else "APPROVE"
    )

    # -----------------------------------------------
    # Layout
    # -----------------------------------------------
    col_a, col_b = st.columns(2)

    # -----------------------------------------------
    # Left Panel
    # -----------------------------------------------
    with col_a:

        st.markdown(
            f"<h2 style='color:{color}'>{dec} — {risk}</h2>",
            unsafe_allow_html=True
        )

        st.metric(
            "Default Probability",
            f"{prob*100:.2f}%"
        )

        st.metric(
            "LTV Ratio",
            f"{ltv:.3f}"
        )

        st.metric(
            "EMI / Income",
            f"{emi_inc:.3f}"
        )

        st.metric(
            "Monthly EMI",
            f"INR {emi:,.0f}"
        )

        # -------------------------------------------
        # Risk Flags
        # -------------------------------------------
        if ltv > 0.85:
            st.error(
                "High LTV (>85%) — major risk factor"
            )

        if emi_inc > 0.40:
            st.error(
                "High EMI stress (>40%) — risk factor"
            )

        if cibil < 650:
            st.error(
                "Low CIBIL score — risk factor"
            )

        if cibil > 750 and ltv < 0.70:
            st.success(
                "Strong profile — low risk"
            )

    # -----------------------------------------------
    # Gauge Chart
    # -----------------------------------------------
    with col_b:

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",

                value=prob*100,

                gauge={
                    "axis":{"range":[0,100]},

                    "bar":{"color":color},

                    "steps":[

                        {
                            "range":[0,8],
                            "color":"#D1FAE5"
                        },

                        {
                            "range":[8,20],
                            "color":"#FEF9C3"
                        },

                        {
                            "range":[20,100],
                            "color":"#FEE2E2"
                        }
                    ],

                    "threshold":{
                        "line":{
                            "color":"red",
                            "width":4
                        },
                        "value":20
                    }
                },

                title={
                    "text":"Default Probability (%)"
                }
            )
        )

        fig.update_layout(height=320)

        st.plotly_chart(
            fig,
            use_container_width=True
        )