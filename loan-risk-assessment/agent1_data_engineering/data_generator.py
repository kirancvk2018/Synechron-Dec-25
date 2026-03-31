"""
Agent 1: Data Engineering - Synthetic Loan Dataset Generator
=============================================================
Generates a realistic synthetic loan dataset with 10,000+ rows
featuring realistic distributions and correlations.

Enterprise Context (Synechron):
- Applicable to retail banking loan origination systems
- Simulates real-world credit risk data distributions
- Supports regulatory compliance testing (Basel III/IV)
"""

import numpy as np
import pandas as pd
import os
import json
import time
from datetime import datetime

# Execution logging
AGENT_NAME = "Agent 1: Data Engineering"
START_TIME = datetime.utcnow().isoformat()

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "execution_logs")


def generate_synthetic_loan_dataset(n_samples: int = 12000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic loan dataset with correlated features.

    Correlations enforced:
    - Higher income -> lower default probability
    - Higher credit score -> lower default probability
    - Higher debt-to-income -> higher default probability
    - Employment type affects income distribution
    """
    np.random.seed(seed)

    # --- Employment Type ---
    employment_types = ["Salaried", "Self-Employed", "Freelancer", "Government", "Unemployed"]
    employment_weights = [0.40, 0.25, 0.15, 0.12, 0.08]
    employment = np.random.choice(employment_types, size=n_samples, p=employment_weights)

    # --- Age (22-65) ---
    age = np.clip(np.random.normal(38, 10, n_samples).astype(int), 22, 65)

    # --- Income (correlated with employment type) ---
    income_means = {
        "Salaried": 65000, "Self-Employed": 80000, "Freelancer": 45000,
        "Government": 58000, "Unemployed": 12000
    }
    income_stds = {
        "Salaried": 20000, "Self-Employed": 35000, "Freelancer": 18000,
        "Government": 12000, "Unemployed": 8000
    }
    income = np.array([
        max(5000, np.random.normal(income_means[e], income_stds[e]))
        for e in employment
    ]).astype(int)

    # --- Credit Score (300-850, correlated with income) ---
    income_normalized = (income - income.min()) / (income.max() - income.min())
    credit_base = 450 + 300 * income_normalized
    credit_noise = np.random.normal(0, 50, n_samples)
    credit_score = np.clip(credit_base + credit_noise, 300, 850).astype(int)

    # --- Loan Amount ---
    loan_amount = np.clip(
        np.random.lognormal(mean=10.5, sigma=0.8, size=n_samples).astype(int),
        5000, 500000
    )

    # --- Loan Term (months) ---
    loan_terms = [12, 24, 36, 48, 60, 84, 120, 180, 240, 360]
    loan_term_weights = [0.05, 0.08, 0.15, 0.12, 0.18, 0.12, 0.10, 0.08, 0.07, 0.05]
    loan_term = np.random.choice(loan_terms, size=n_samples, p=loan_term_weights)

    # --- Existing EMI ---
    existing_emi = np.clip(
        np.random.exponential(scale=500, size=n_samples).astype(int),
        0, 15000
    )

    # --- Default History ---
    # Higher credit score -> less likely to have default history
    default_hist_prob = np.clip(0.6 - 0.0006 * credit_score, 0.02, 0.5)
    default_history = np.array(["Yes" if np.random.random() < p else "No"
                                 for p in default_hist_prob])

    # --- Loan Status (target variable) ---
    # Multi-factor default probability
    base_default_prob = 0.15

    # Income factor (higher income = lower default)
    income_factor = -0.15 * income_normalized

    # Credit score factor
    credit_factor = -0.25 * ((credit_score - 300) / 550)

    # Employment factor
    emp_risk = {
        "Salaried": -0.05, "Government": -0.08, "Self-Employed": 0.02,
        "Freelancer": 0.08, "Unemployed": 0.25
    }
    employment_factor = np.array([emp_risk[e] for e in employment])

    # Default history factor
    history_factor = np.where(default_history == "Yes", 0.15, -0.03)

    # Loan amount relative to income
    loan_to_income = loan_amount / np.maximum(income, 1)
    lti_factor = 0.05 * np.clip(loan_to_income - 3, 0, 10)

    # Existing EMI burden
    emi_factor = 0.03 * np.clip(existing_emi / np.maximum(income / 12, 1) - 0.3, 0, 1)

    # Age factor (very young and very old slightly higher risk)
    age_factor = 0.03 * (np.abs(age - 40) / 25)

    default_prob = np.clip(
        base_default_prob + income_factor + credit_factor + employment_factor +
        history_factor + lti_factor + emi_factor + age_factor +
        np.random.normal(0, 0.03, n_samples),
        0.01, 0.95
    )

    loan_status = np.array(["Default" if np.random.random() < p else "Non-Default"
                             for p in default_prob])

    # --- Introduce some missing values (realistic) ---
    missing_indices_income = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    missing_indices_credit = np.random.choice(n_samples, size=int(n_samples * 0.015), replace=False)
    missing_indices_emi = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)

    income_float = income.astype(float)
    credit_float = credit_score.astype(float)
    emi_float = existing_emi.astype(float)

    income_float[missing_indices_income] = np.nan
    credit_float[missing_indices_credit] = np.nan
    emi_float[missing_indices_emi] = np.nan

    # Build DataFrame
    df = pd.DataFrame({
        "Applicant_ID": [f"APP_{str(i+1).zfill(5)}" for i in range(n_samples)],
        "Age": age,
        "Income": income_float,
        "Employment_Type": employment,
        "Credit_Score": credit_float,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Existing_EMI": emi_float,
        "Default_History": default_history,
        "Loan_Status": loan_status,
    })

    # Add some duplicate rows for realism
    n_duplicates = int(n_samples * 0.005)
    dup_indices = np.random.choice(n_samples, size=n_duplicates, replace=False)
    duplicates = df.iloc[dup_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data cleaning pipeline:
    - Remove duplicates
    - Handle missing values (median imputation for numeric, mode for categorical)
    - Validate data ranges
    """
    print(f"[{AGENT_NAME}] Starting data cleaning...")
    print(f"  Raw dataset shape: {df.shape}")
    print(f"  Missing values:\n{df.isnull().sum()}")

    # Remove duplicates
    n_before = len(df)
    df = df.drop_duplicates(subset=["Applicant_ID"], keep="first").reset_index(drop=True)
    print(f"  Removed {n_before - len(df)} duplicate rows")

    # Impute missing values
    df["Income"] = df["Income"].fillna(df["Income"].median())
    df["Credit_Score"] = df["Credit_Score"].fillna(df["Credit_Score"].median())
    df["Existing_EMI"] = df["Existing_EMI"].fillna(df["Existing_EMI"].median())

    # Validate ranges
    df["Age"] = df["Age"].clip(18, 70)
    df["Credit_Score"] = df["Credit_Score"].clip(300, 850)
    df["Income"] = df["Income"].clip(lower=0)
    df["Loan_Amount"] = df["Loan_Amount"].clip(lower=1000)

    print(f"  Cleaned dataset shape: {df.shape}")
    print(f"  Missing values after cleaning:\n{df.isnull().sum()}")
    return df


def detect_outliers(df: pd.DataFrame) -> dict:
    """Detect outliers using IQR method for numeric columns."""
    outlier_report = {}
    numeric_cols = ["Age", "Income", "Credit_Score", "Loan_Amount", "Existing_EMI"]

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outlier_report[col] = {
            "count": len(outliers),
            "percentage": round(len(outliers) / len(df) * 100, 2),
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2),
        }
    print(f"[{AGENT_NAME}] Outlier detection complete")
    return outlier_report


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering:
    - Debt-to-Income ratio
    - Loan-to-Income ratio
    - Monthly income
    - EMI burden ratio
    - Credit score bins
    - Age group bins
    """
    print(f"[{AGENT_NAME}] Starting feature engineering...")

    # Monthly income
    df["Monthly_Income"] = (df["Income"] / 12).round(2)

    # Estimated monthly loan payment (simplified)
    monthly_rate = 0.08 / 12  # 8% annual rate assumption
    df["Estimated_Monthly_Payment"] = (
        df["Loan_Amount"] * monthly_rate /
        (1 - (1 + monthly_rate) ** (-df["Loan_Term"]))
    ).round(2)

    # Debt-to-Income Ratio
    df["Debt_to_Income_Ratio"] = (
        (df["Existing_EMI"] + df["Estimated_Monthly_Payment"]) /
        df["Monthly_Income"].replace(0, np.nan)
    ).round(4)
    df["Debt_to_Income_Ratio"] = df["Debt_to_Income_Ratio"].fillna(0)

    # Loan-to-Income Ratio
    df["Loan_to_Income_Ratio"] = (df["Loan_Amount"] / df["Income"].replace(0, np.nan)).round(4)
    df["Loan_to_Income_Ratio"] = df["Loan_to_Income_Ratio"].fillna(0)

    # EMI Burden Ratio
    df["EMI_Burden_Ratio"] = (
        df["Existing_EMI"] / df["Monthly_Income"].replace(0, np.nan)
    ).round(4)
    df["EMI_Burden_Ratio"] = df["EMI_Burden_Ratio"].fillna(0)

    # Credit Score Bins
    df["Credit_Score_Bin"] = pd.cut(
        df["Credit_Score"],
        bins=[299, 500, 600, 700, 750, 850],
        labels=["Very Poor", "Poor", "Fair", "Good", "Excellent"]
    )

    # Age Group Bins
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[17, 25, 35, 45, 55, 70],
        labels=["18-25", "26-35", "36-45", "46-55", "56+"]
    )

    print(f"  Engineered {6} new features")
    print(f"  Final dataset shape: {df.shape}")
    return df


def generate_data_dictionary(df: pd.DataFrame) -> dict:
    """Generate a data dictionary describing all columns."""
    descriptions = {
        "Applicant_ID": "Unique identifier for each loan applicant",
        "Age": "Age of the applicant in years (22-65)",
        "Income": "Annual income of the applicant in USD",
        "Employment_Type": "Type of employment (Salaried/Self-Employed/Freelancer/Government/Unemployed)",
        "Credit_Score": "Credit score of the applicant (300-850)",
        "Loan_Amount": "Requested loan amount in USD",
        "Loan_Term": "Loan term in months",
        "Existing_EMI": "Existing monthly EMI obligations in USD",
        "Default_History": "Whether the applicant has a history of default (Yes/No)",
        "Loan_Status": "Target variable - loan outcome (Default/Non-Default)",
        "Monthly_Income": "Derived: Annual income divided by 12",
        "Estimated_Monthly_Payment": "Derived: Estimated monthly payment for the requested loan",
        "Debt_to_Income_Ratio": "Derived: Total debt obligations relative to monthly income",
        "Loan_to_Income_Ratio": "Derived: Loan amount relative to annual income",
        "EMI_Burden_Ratio": "Derived: Existing EMI relative to monthly income",
        "Credit_Score_Bin": "Derived: Categorical binning of credit score",
        "Age_Group": "Derived: Categorical binning of age",
    }

    data_dict = {}
    for col in df.columns:
        data_dict[col] = {
            "description": descriptions.get(col, "N/A"),
            "dtype": str(df[col].dtype),
            "non_null_count": int(df[col].notnull().sum()),
            "null_count": int(df[col].isnull().sum()),
            "unique_values": int(df[col].nunique()),
            "sample_values": df[col].dropna().head(3).tolist(),
        }
    return data_dict


def generate_summary_statistics(df: pd.DataFrame) -> dict:
    """Generate summary statistics for the dataset."""
    stats = {
        "total_records": len(df),
        "total_features": len(df.columns),
        "default_rate": round((df["Loan_Status"] == "Default").mean() * 100, 2),
        "numeric_summary": json.loads(
            df.describe().round(2).to_json()
        ),
        "categorical_summary": {},
        "correlation_highlights": {},
    }

    # Categorical summaries
    for col in ["Employment_Type", "Default_History", "Loan_Status", "Credit_Score_Bin", "Age_Group"]:
        if col in df.columns:
            stats["categorical_summary"][col] = df[col].value_counts().to_dict()

    # Key correlations
    numeric_df = df.select_dtypes(include=[np.number])
    target_numeric = (df["Loan_Status"] == "Default").astype(int)
    correlations = numeric_df.corrwith(target_numeric).round(4).to_dict()
    stats["correlation_highlights"] = dict(
        sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    )

    return stats


def run_data_engineering_pipeline():
    """Execute the full data engineering pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    log = {
        "agent": AGENT_NAME,
        "start_time": START_TIME,
        "steps": [],
        "decisions": [],
        "dependencies": ["numpy", "pandas"],
    }

    print(f"\n{'='*60}")
    print(f"  {AGENT_NAME} - Starting Pipeline")
    print(f"{'='*60}\n")

    # Step 1: Generate synthetic data
    step_start = time.time()
    print("[Step 1] Generating synthetic loan dataset (12,000 rows)...")
    df_raw = generate_synthetic_loan_dataset(n_samples=12000)
    log["steps"].append({
        "step": "Generate Synthetic Data",
        "duration_seconds": round(time.time() - step_start, 2),
        "output_rows": len(df_raw),
        "status": "completed"
    })
    log["decisions"].append("Generated 12,000 rows with realistic correlations between income, credit score, and default probability")

    # Step 2: Clean data
    step_start = time.time()
    print("\n[Step 2] Cleaning dataset...")
    df_clean = clean_dataset(df_raw)
    log["steps"].append({
        "step": "Data Cleaning",
        "duration_seconds": round(time.time() - step_start, 2),
        "output_rows": len(df_clean),
        "status": "completed"
    })

    # Step 3: Outlier detection
    step_start = time.time()
    print("\n[Step 3] Detecting outliers...")
    outlier_report = detect_outliers(df_clean)
    log["steps"].append({
        "step": "Outlier Detection",
        "duration_seconds": round(time.time() - step_start, 2),
        "status": "completed"
    })

    # Step 4: Feature engineering
    step_start = time.time()
    print("\n[Step 4] Engineering features...")
    df_final = engineer_features(df_clean)
    log["steps"].append({
        "step": "Feature Engineering",
        "duration_seconds": round(time.time() - step_start, 2),
        "new_features": 6,
        "status": "completed"
    })
    log["decisions"].append("Added Debt-to-Income, Loan-to-Income, EMI Burden ratios, plus categorical bins for Credit Score and Age")

    # Step 5: Generate data dictionary and summary statistics
    step_start = time.time()
    print("\n[Step 5] Generating data dictionary and summary statistics...")
    data_dict = generate_data_dictionary(df_final)
    summary_stats = generate_summary_statistics(df_final)
    log["steps"].append({
        "step": "Documentation Generation",
        "duration_seconds": round(time.time() - step_start, 2),
        "status": "completed"
    })

    # Save outputs
    print("\n[Saving Outputs]...")
    df_final.to_csv(os.path.join(OUTPUT_DIR, "loan_dataset_clean.csv"), index=False)
    print(f"  Saved: loan_dataset_clean.csv ({len(df_final)} rows)")

    with open(os.path.join(OUTPUT_DIR, "data_dictionary.json"), "w") as f:
        json.dump(data_dict, f, indent=2, default=str)
    print("  Saved: data_dictionary.json")

    with open(os.path.join(OUTPUT_DIR, "summary_statistics.json"), "w") as f:
        json.dump(summary_stats, f, indent=2, default=str)
    print("  Saved: summary_statistics.json")

    with open(os.path.join(OUTPUT_DIR, "outlier_report.json"), "w") as f:
        json.dump(outlier_report, f, indent=2)
    print("  Saved: outlier_report.json")

    # Save execution log
    log["end_time"] = datetime.utcnow().isoformat()
    log["total_duration_seconds"] = round(
        sum(s["duration_seconds"] for s in log["steps"]), 2
    )
    log["status"] = "completed"

    with open(os.path.join(LOG_DIR, "agent1_execution_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print("  Saved: agent1_execution_log.json")

    print(f"\n{'='*60}")
    print(f"  {AGENT_NAME} - Pipeline Complete")
    print(f"  Total records: {len(df_final)}")
    print(f"  Default rate: {summary_stats['default_rate']}%")
    print(f"{'='*60}\n")

    return df_final, summary_stats, data_dict, outlier_report


if __name__ == "__main__":
    run_data_engineering_pipeline()
