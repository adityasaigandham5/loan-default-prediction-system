from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import joblib
import json
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------
app = FastAPI(
    title="Slice Home & Vehicle Loan Risk API",
    description="LightGBM default prediction + Cox survival model",
    version="1.0.0"
)

# ---------------------------------------------------
# CORS
# ---------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---------------------------------------------------
# Load Models
# ---------------------------------------------------
lgbm_model = joblib.load("../models/loan_default_model.pkl")

cox_model = joblib.load("../models/cox_model.pkl")

cox_scaler = joblib.load("../models/cox_scaler.pkl")

le_emp = joblib.load("../models/le_emp.pkl")

le_city = joblib.load("../models/le_city.pkl")

le_type = joblib.load("../models/le_type.pkl")

lgbm_metrics = json.load(
    open("../models/lgbm_metrics.json")
)

cox_metrics = json.load(
    open("../models/cox_metrics.json")
)

# ---------------------------------------------------
# Input Schema
# ---------------------------------------------------
class LoanApplication(BaseModel):

    loan_type: str = Field(..., example="Home")

    age: int = Field(..., ge=21, le=70, example=35)

    income_mo: float = Field(..., ge=10000, example=85000)

    employment: str = Field(..., example="Salaried")

    emp_years: int = Field(5, ge=0, example=5)

    cibil: int = Field(..., ge=300, le=900, example=720)

    loan_amt: float = Field(..., ge=50000, example=3500000)

    property_val: float = Field(..., ge=100000, example=5000000)

    tenure_mo: int = Field(..., ge=12, example=240)

    interest_rate: float = Field(
        ...,
        ge=5.0,
        le=20.0,
        example=8.5
    )

    delinq_hist: int = Field(0, ge=0, example=0)

    num_loans: int = Field(1, ge=0, example=1)

    co_applicant: int = Field(0, ge=0, le=1, example=0)

    city_tier: str = Field("Tier1", example="Tier1")

# ---------------------------------------------------
# Feature Builder
# ---------------------------------------------------
def build_lgbm_features(req: LoanApplication):

    emi_rate = req.interest_rate / (12 * 100)

    emi = (
        req.loan_amt * emi_rate * (1+emi_rate)**req.tenure_mo /
        ((1+emi_rate)**req.tenure_mo - 1)
    )

    ltv = req.loan_amt / req.property_val

    emi_income = emi / req.income_mo

    emp_enc = int(
        le_emp.transform([req.employment])[0]
    )

    city_enc = int(
        le_city.transform([req.city_tier])[0]
    )

    type_enc = int(
        le_type.transform([req.loan_type])[0]
    )

    return pd.DataFrame([{

        'age': req.age,

        'income_mo': req.income_mo,

        'emp_enc': emp_enc,

        'emp_years': req.emp_years,

        'cibil': req.cibil,

        'loan_amt': req.loan_amt,

        'ltv': ltv,

        'tenure_mo': req.tenure_mo,

        'interest_rate': req.interest_rate,

        'emi': emi,

        'emi_income': emi_income,

        'delinq_hist': req.delinq_hist,

        'num_loans': req.num_loans,

        'co_applicant': req.co_applicant,

        'city_enc': city_enc,

        'type_enc': type_enc,

        'loan_income_ratio':
            req.loan_amt / (req.income_mo * 12 + 1),

        'high_ltv':
            int(ltv > 0.85),

        'high_emi_stress':
            int(emi_income > 0.40),

        'bad_cibil':
            int(req.cibil < 650),

        'good_cibil':
            int(req.cibil > 750),

        'has_delinquency':
            int(req.delinq_hist > 0),

        'age_income_ratio':
            req.age / (req.income_mo + 1)

    }])

# ---------------------------------------------------
# Root Endpoint
# ---------------------------------------------------
@app.get("/")
def root():

    return {
        "status": "running",
        "lgbm_auc": lgbm_metrics["auc_roc"],
        "cox_cindex": cox_metrics["c_index"]
    }

# ---------------------------------------------------
# Prediction Endpoint
# ---------------------------------------------------
@app.post("/predict/risk")
def predict_risk(req: LoanApplication):

    start = time.time()

    try:

        feats = build_lgbm_features(req)

        prob = float(
            lgbm_model.predict_proba(feats)[0][1]
        )

        ltv = req.loan_amt / req.property_val

        emi_r = req.interest_rate / (12 * 100)

        emi = (
            req.loan_amt * emi_r * (1+emi_r)**req.tenure_mo /
            ((1+emi_r)**req.tenure_mo - 1)
        )

        emi_inc = emi / req.income_mo

        # ------------------------------------------
        # Risk Logic
        # ------------------------------------------
        risk = (
            "HIGH"
            if prob > 0.20
            else "MEDIUM"
            if prob > 0.08
            else "LOW"
        )

        decision = (
            "REJECT"
            if prob > 0.20
            else "REVIEW"
            if prob > 0.08
            else "APPROVE"
        )

        # ------------------------------------------
        # Risk Flags
        # ------------------------------------------
        flags = []

        if ltv > 0.85:
            flags.append(f"High LTV ({ltv:.2f})")

        if emi_inc > 0.40:
            flags.append(
                f"High EMI stress ({emi_inc:.2f})"
            )

        if req.cibil < 650:
            flags.append(
                f"Low CIBIL ({req.cibil})"
            )

        if req.delinq_hist:
            flags.append(
                f"Delinquency history ({req.delinq_hist})"
            )

        # ------------------------------------------
        # Response
        # ------------------------------------------
        return {

            "default_probability":
                round(prob,4),

            "risk_level":
                risk,

            "decision":
                decision,

            "ltv":
                round(ltv,3),

            "emi_income_ratio":
                round(emi_inc,3),

            "risk_flags":
                flags[:3],

            "latency_ms":
                round(
                    (time.time()-start)*1000,
                    2
                )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )