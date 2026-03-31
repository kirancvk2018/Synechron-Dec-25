"""
Agent 3: Backend API - FastAPI REST API for Loan Risk Assessment
================================================================
Exposes endpoints for loan risk prediction and model insights.
Integrates trained model from Agent 2.

Enterprise Context (Synechron):
- RESTful microservice architecture for banking systems
- Swagger/OpenAPI documentation for API governance
- Input validation for production-grade deployments
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

AGENT_NAME = "Agent 3: Backend API"
START_TIME = datetime.utcnow().isoformat()

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "execution_logs")

# Initialize FastAPI app
app = FastAPI(
    title="Loan Risk Assessment API",
    description=(
        "Enterprise-grade REST API for loan risk prediction and model insights. "
        "Built for Synechron's financial services platform demonstrating "
        "multi-agent AI collaboration."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model artifacts
model = None
scaler = None
label_encoders = None
feature_columns = None
evaluation_report = None
risk_insights = None
summary_stats = None


# --- Pydantic Models ---

class LoanApplication(BaseModel):
    """Input schema for loan risk prediction."""
    age: int = Field(..., ge=18, le=70, description="Applicant age (18-70)")
    income: float = Field(..., gt=0, description="Annual income in USD")
    employment_type: str = Field(
        ..., description="Employment type: Salaried, Self-Employed, Freelancer, Government, Unemployed",
        pattern="^(Salaried|Self-Employed|Freelancer|Government|Unemployed)$"
    )
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount in USD")
    loan_term: int = Field(..., gt=0, description="Loan term in months")
    existing_emi: float = Field(..., ge=0, description="Existing monthly EMI obligations in USD")
    default_history: str = Field(..., description="Default history: Yes or No", pattern="^(Yes|No)$")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "income": 75000,
                "employment_type": "Salaried",
                "credit_score": 720,
                "loan_amount": 50000,
                "loan_term": 60,
                "existing_emi": 500,
                "default_history": "No"
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for loan risk prediction."""
    applicant_risk_score: float
    risk_category: str
    default_probability: float
    recommendation: str
    risk_factors: list
    model_used: str
    prediction_timestamp: str


class SummaryResponse(BaseModel):
    """Output schema for model summary/insights."""
    model_performance: dict
    feature_importance: dict
    risk_distribution: dict
    key_insights: list
    enterprise_context: dict


# --- Startup Event ---

@app.on_event("startup")
async def load_model_artifacts():
    """Load model artifacts on startup. Generate if not available."""
    global model, scaler, label_encoders, feature_columns
    global evaluation_report, risk_insights, summary_stats

    print(f"[{AGENT_NAME}] Loading model artifacts...")

    # Check if outputs exist, if not run the pipeline
    model_path = os.path.join(OUTPUT_DIR, "trained_model.joblib")

    if not os.path.exists(model_path):
        print(f"[{AGENT_NAME}] Model artifacts not found. Running pipeline...")
        try:
            from agent1_data_engineering.data_generator import run_data_engineering_pipeline
            run_data_engineering_pipeline()
            from agent2_ml_modeling.model_trainer import run_ml_pipeline
            run_ml_pipeline()
        except Exception as e:
            print(f"[{AGENT_NAME}] WARNING: Failed to run pipeline: {e}")
            print(f"[{AGENT_NAME}] API will start with limited functionality")

    # Load model
    try:
        model = joblib.load(os.path.join(OUTPUT_DIR, "trained_model.joblib"))
        print("  Loaded: trained_model.joblib")
    except Exception as e:
        print(f"  WARNING: Could not load model: {e}")

    # Load scaler
    try:
        scaler = joblib.load(os.path.join(OUTPUT_DIR, "scaler.joblib"))
        print("  Loaded: scaler.joblib")
    except Exception as e:
        print(f"  WARNING: Could not load scaler: {e}")

    # Load label encoders
    try:
        label_encoders = joblib.load(os.path.join(OUTPUT_DIR, "label_encoders.joblib"))
        print("  Loaded: label_encoders.joblib")
    except Exception as e:
        print(f"  WARNING: Could not load label encoders: {e}")

    # Load feature columns
    try:
        with open(os.path.join(OUTPUT_DIR, "feature_columns.json"), "r") as f:
            feature_columns = json.load(f)
        print("  Loaded: feature_columns.json")
    except Exception as e:
        print(f"  WARNING: Could not load feature columns: {e}")
        feature_columns = [
            "Age", "Income", "Credit_Score", "Loan_Amount", "Loan_Term",
            "Existing_EMI", "Debt_to_Income_Ratio", "Loan_to_Income_Ratio",
            "EMI_Burden_Ratio", "Employment_Type_Encoded", "Default_History_Encoded"
        ]

    # Load evaluation report
    try:
        with open(os.path.join(OUTPUT_DIR, "evaluation_report.json"), "r") as f:
            evaluation_report = json.load(f)
        print("  Loaded: evaluation_report.json")
    except Exception as e:
        print(f"  WARNING: Could not load evaluation report: {e}")

    # Load risk insights
    try:
        with open(os.path.join(OUTPUT_DIR, "risk_insights.json"), "r") as f:
            risk_insights = json.load(f)
        print("  Loaded: risk_insights.json")
    except Exception as e:
        print(f"  WARNING: Could not load risk insights: {e}")

    # Load summary statistics
    try:
        with open(os.path.join(OUTPUT_DIR, "summary_statistics.json"), "r") as f:
            summary_stats = json.load(f)
        print("  Loaded: summary_statistics.json")
    except Exception as e:
        print(f"  WARNING: Could not load summary statistics: {e}")

    print(f"[{AGENT_NAME}] Model artifacts loaded successfully")

    # Save execution log
    os.makedirs(LOG_DIR, exist_ok=True)
    log = {
        "agent": AGENT_NAME,
        "start_time": START_TIME,
        "status": "running",
        "endpoints": ["/predict", "/summary", "/health", "/docs"],
        "dependencies": ["fastapi", "uvicorn", "joblib", "scikit-learn"],
        "decisions": [
            "Using FastAPI for automatic OpenAPI documentation",
            "CORS enabled for frontend integration",
            "Pydantic models for input validation",
            "Fallback to pipeline execution if model artifacts not found",
        ],
    }
    with open(os.path.join(LOG_DIR, "agent3_execution_log.json"), "w") as f:
        json.dump(log, f, indent=2)


# --- Endpoints ---

@app.get("/", tags=["Root"])
async def root():
    """API root endpoint."""
    return {
        "service": "Loan Risk Assessment API",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs",
        "endpoints": {
            "predict": "/predict",
            "summary": "/summary",
            "health": "/health",
        },
        "enterprise": "Synechron - Financial Services",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_loan_risk(application: LoanApplication):
    """
    Predict loan risk for a given application.

    Returns a risk score, risk category, default probability,
    and personalized recommendation.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the ML pipeline has been executed."
        )

    try:
        # Calculate derived features
        monthly_income = application.income / 12
        monthly_rate = 0.08 / 12
        estimated_payment = (
            application.loan_amount * monthly_rate /
            (1 - (1 + monthly_rate) ** (-application.loan_term))
        )
        debt_to_income = (
            (application.existing_emi + estimated_payment) / monthly_income
            if monthly_income > 0 else 0
        )
        loan_to_income = (
            application.loan_amount / application.income
            if application.income > 0 else 0
        )
        emi_burden = (
            application.existing_emi / monthly_income
            if monthly_income > 0 else 0
        )

        # Encode categorical features
        employment_mapping = {
            "Freelancer": 0, "Government": 1, "Salaried": 2,
            "Self-Employed": 3, "Unemployed": 4
        }
        default_hist_mapping = {"No": 0, "Yes": 1}

        emp_encoded = employment_mapping.get(application.employment_type, 2)
        hist_encoded = default_hist_mapping.get(application.default_history, 0)

        # Prepare feature vector
        features = np.array([[
            application.age,
            application.income,
            application.credit_score,
            application.loan_amount,
            application.loan_term,
            application.existing_emi,
            debt_to_income,
            loan_to_income,
            emi_burden,
            emp_encoded,
            hist_encoded,
        ]])

        # Only scale features if the best model requires it (Logistic Regression)
        needs_scaling = (
            evaluation_report
            and evaluation_report.get("best_model") == "Logistic Regression"
        )
        if needs_scaling and scaler is not None:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features

        # Get prediction probability
        default_prob = float(model.predict_proba(features_scaled)[0][1])
        risk_score = round(default_prob * 100, 2)

        # Determine risk category
        if risk_score <= 20:
            risk_category = "Low Risk"
            recommendation = "APPROVE - Strong credit profile with low default probability."
        elif risk_score <= 40:
            risk_category = "Moderate Risk"
            recommendation = "CONDITIONAL APPROVE - Consider additional documentation or collateral requirements."
        elif risk_score <= 60:
            risk_category = "High Risk"
            recommendation = "REVIEW - Manual underwriting review recommended. Consider higher interest rate or reduced loan amount."
        else:
            risk_category = "Very High Risk"
            recommendation = "DECLINE - High default probability. Recommend credit counseling before reapplication."

        # Identify risk factors
        risk_factors = []
        if application.credit_score < 600:
            risk_factors.append("Low credit score (below 600)")
        if debt_to_income > 0.5:
            risk_factors.append(f"High debt-to-income ratio ({debt_to_income:.2f})")
        if application.default_history == "Yes":
            risk_factors.append("Previous default history")
        if application.employment_type == "Unemployed":
            risk_factors.append("Unemployed - no stable income source")
        if loan_to_income > 5:
            risk_factors.append(f"High loan-to-income ratio ({loan_to_income:.2f})")
        if application.existing_emi > monthly_income * 0.4:
            risk_factors.append("Existing EMI exceeds 40% of monthly income")
        if not risk_factors:
            risk_factors.append("No significant risk factors identified")

        return PredictionResponse(
            applicant_risk_score=risk_score,
            risk_category=risk_category,
            default_probability=round(default_prob, 4),
            recommendation=recommendation,
            risk_factors=risk_factors,
            model_used=evaluation_report.get("best_model", "Unknown") if evaluation_report else "Unknown",
            prediction_timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/summary", tags=["Insights"])
async def get_model_summary():
    """
    Returns comprehensive model insights and risk analytics.

    Includes model performance metrics, feature importance,
    risk distribution across segments, and enterprise insights.
    """
    if evaluation_report is None and risk_insights is None:
        raise HTTPException(
            status_code=503,
            detail="Model insights not available. Please ensure the ML pipeline has been executed."
        )

    response = {
        "model_performance": {},
        "feature_importance": {},
        "risk_distribution": {},
        "key_insights": [],
        "enterprise_context": {},
        "data_summary": {},
    }

    if evaluation_report:
        response["model_performance"] = {
            "best_model": evaluation_report.get("best_model"),
            "models_evaluated": list(evaluation_report.get("model_results", {}).keys()),
            "metrics": {
                name: {
                    "accuracy": m.get("accuracy"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "f1_score": m.get("f1_score"),
                    "roc_auc": m.get("roc_auc"),
                    "cv_roc_auc_mean": m.get("cv_roc_auc_mean"),
                }
                for name, m in evaluation_report.get("model_results", {}).items()
            },
        }

        if "feature_importance" in evaluation_report:
            response["feature_importance"] = evaluation_report["feature_importance"]

    if risk_insights:
        response["risk_distribution"] = {
            "overall_default_rate": risk_insights.get("overall_default_rate"),
            "by_employment_type": risk_insights.get("risk_by_employment_type"),
            "by_credit_score_bin": risk_insights.get("risk_by_credit_score_bin"),
            "by_age_group": risk_insights.get("risk_by_age_group"),
        }
        response["key_insights"] = risk_insights.get("enterprise_insights", [])
        response["enterprise_context"] = {
            "regulatory_notes": risk_insights.get("regulatory_notes", []),
            "applicability": "Banking and Financial Services - Credit Risk Assessment",
            "compliance": "Model uses interpretable features compliant with fair lending regulations",
            "synechron_value": [
                "Reduced development lifecycle via parallel AI agents",
                "Scalable multi-agent workflow for enterprise delivery",
                "Full audit trail with governance and traceability",
            ],
        }

    if summary_stats:
        response["data_summary"] = {
            "total_records": summary_stats.get("total_records"),
            "total_features": summary_stats.get("total_features"),
            "default_rate": summary_stats.get("default_rate"),
        }

    return response


@app.get("/feature-importance", tags=["Insights"])
async def get_feature_importance():
    """Returns feature importance data for visualization."""
    if evaluation_report and "feature_importance" in evaluation_report:
        fi = evaluation_report["feature_importance"]["feature_importance"]
        # Format for chart consumption
        chart_data = [
            {"feature": name, "importance": data["importance_percentage"]}
            for name, data in sorted(
                fi.items(), key=lambda x: x[1]["rank"]
            )
        ]
        return {"feature_importance": chart_data}
    raise HTTPException(status_code=503, detail="Feature importance data not available")


@app.get("/risk-distribution", tags=["Insights"])
async def get_risk_distribution():
    """Returns risk distribution data for visualization."""
    if risk_insights:
        return {
            "by_employment": risk_insights.get("risk_by_employment_type", {}),
            "by_credit_score": risk_insights.get("risk_by_credit_score_bin", {}),
            "by_age_group": risk_insights.get("risk_by_age_group", {}),
            "overall_default_rate": risk_insights.get("overall_default_rate"),
        }
    raise HTTPException(status_code=503, detail="Risk distribution data not available")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
