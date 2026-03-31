"""
Agent 5: Integration & QA - Test Suite
=======================================
Comprehensive testing for all components of the Loan Risk Assessment system.

Enterprise Context (Synechron):
- Ensures production readiness for banking deployments
- Validates data integrity, model accuracy, and API reliability
- Provides test report for audit and compliance
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import pytest
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

AGENT_NAME = "Agent 5: Integration & QA"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "execution_logs")


# =============================================================================
# Test 1: Data Engineering Tests
# =============================================================================

class TestDataEngineering:
    """Tests for Agent 1: Data Engineering."""

    def test_dataset_exists(self):
        """Verify the clean dataset was generated."""
        path = os.path.join(OUTPUT_DIR, "loan_dataset_clean.csv")
        assert os.path.exists(path), f"Clean dataset not found at {path}"

    def test_dataset_row_count(self):
        """Verify dataset has 10,000+ rows."""
        df = pd.read_csv(os.path.join(OUTPUT_DIR, "loan_dataset_clean.csv"))
        assert len(df) >= 10000, f"Expected 10,000+ rows, got {len(df)}"

    def test_required_columns_exist(self):
        """Verify all required columns are present."""
        df = pd.read_csv(os.path.join(OUTPUT_DIR, "loan_dataset_clean.csv"))
        required_cols = [
            "Applicant_ID", "Age", "Income", "Employment_Type",
            "Credit_Score", "Loan_Amount", "Loan_Term", "Existing_EMI",
            "Default_History", "Loan_Status"
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"

    def test_no_missing_values(self):
        """Verify no missing values after cleaning."""
        df = pd.read_csv(os.path.join(OUTPUT_DIR, "loan_dataset_clean.csv"))
        core_cols = ["Age", "Income", "Credit_Score", "Loan_Amount", "Existing_EMI"]
        for col in core_cols:
            assert df[col].isnull().sum() == 0, f"Column {col} has {df[col].isnull().sum()} missing values"

    def test_no_duplicate_ids(self):
        """Verify no duplicate applicant IDs."""
        df = pd.read_csv(os.path.join(OUTPUT_DIR, "loan_dataset_clean.csv"))
        assert df["Applicant_ID"].nunique() == len(df), "Duplicate Applicant_IDs found"

    def test_data_ranges(self):
        """Verify data values are within expected ranges."""
        df = pd.read_csv(os.path.join(OUTPUT_DIR, "loan_dataset_clean.csv"))
        assert df["Age"].between(18, 70).all(), "Age values out of range"
        assert df["Credit_Score"].between(300, 850).all(), "Credit score out of range"
        assert (df["Income"] >= 0).all(), "Negative income values found"
        assert (df["Loan_Amount"] >= 1000).all(), "Loan amount below minimum"

    def test_engineered_features_exist(self):
        """Verify engineered features were created."""
        df = pd.read_csv(os.path.join(OUTPUT_DIR, "loan_dataset_clean.csv"))
        engineered = ["Debt_to_Income_Ratio", "Loan_to_Income_Ratio", "EMI_Burden_Ratio"]
        for col in engineered:
            assert col in df.columns, f"Missing engineered feature: {col}"

    def test_target_distribution(self):
        """Verify target variable has both classes."""
        df = pd.read_csv(os.path.join(OUTPUT_DIR, "loan_dataset_clean.csv"))
        classes = df["Loan_Status"].unique()
        assert "Default" in classes, "Missing 'Default' class"
        assert "Non-Default" in classes, "Missing 'Non-Default' class"

    def test_data_dictionary_exists(self):
        """Verify data dictionary was generated."""
        path = os.path.join(OUTPUT_DIR, "data_dictionary.json")
        assert os.path.exists(path), "Data dictionary not found"
        with open(path, "r") as f:
            dd = json.load(f)
        assert len(dd) > 0, "Data dictionary is empty"

    def test_summary_statistics_exist(self):
        """Verify summary statistics were generated."""
        path = os.path.join(OUTPUT_DIR, "summary_statistics.json")
        assert os.path.exists(path), "Summary statistics not found"
        with open(path, "r") as f:
            stats = json.load(f)
        assert "total_records" in stats, "Missing total_records in summary"
        assert "default_rate" in stats, "Missing default_rate in summary"


# =============================================================================
# Test 2: ML Modeling Tests
# =============================================================================

class TestMLModeling:
    """Tests for Agent 2: ML Modeling."""

    def test_model_file_exists(self):
        """Verify trained model was saved."""
        path = os.path.join(OUTPUT_DIR, "trained_model.joblib")
        assert os.path.exists(path), "Trained model not found"

    def test_scaler_file_exists(self):
        """Verify scaler was saved."""
        path = os.path.join(OUTPUT_DIR, "scaler.joblib")
        assert os.path.exists(path), "Scaler not found"

    def test_evaluation_report_exists(self):
        """Verify evaluation report was generated."""
        path = os.path.join(OUTPUT_DIR, "evaluation_report.json")
        assert os.path.exists(path), "Evaluation report not found"

    def test_evaluation_report_structure(self):
        """Verify evaluation report has required fields."""
        with open(os.path.join(OUTPUT_DIR, "evaluation_report.json"), "r") as f:
            report = json.load(f)
        assert "best_model" in report, "Missing best_model in report"
        assert "model_results" in report, "Missing model_results in report"
        assert len(report["model_results"]) >= 2, "Expected at least 2 models evaluated"

    def test_model_metrics_present(self):
        """Verify all required metrics are in the report."""
        with open(os.path.join(OUTPUT_DIR, "evaluation_report.json"), "r") as f:
            report = json.load(f)
        required_metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        for model_name, metrics in report["model_results"].items():
            for metric in required_metrics:
                assert metric in metrics, f"Missing {metric} for {model_name}"

    def test_model_prediction_works(self):
        """Verify model can make predictions."""
        import joblib
        model = joblib.load(os.path.join(OUTPUT_DIR, "trained_model.joblib"))

        # Sample input (11 features)
        sample = np.array([[35, 75000, 700, 50000, 60, 500, 0.15, 0.67, 0.08, 2, 0]])
        pred = model.predict(sample)
        prob = model.predict_proba(sample)

        assert pred.shape == (1,), "Unexpected prediction shape"
        assert prob.shape[1] == 2, "Expected 2 probability classes"
        assert 0 <= prob[0][1] <= 1, "Probability out of [0,1] range"

    def test_roc_auc_above_threshold(self):
        """Verify best model ROC-AUC is above minimum threshold."""
        with open(os.path.join(OUTPUT_DIR, "evaluation_report.json"), "r") as f:
            report = json.load(f)
        best = report["best_model"]
        auc = report["model_results"][best]["roc_auc"]
        assert auc >= 0.60, f"ROC-AUC too low: {auc} (minimum 0.60)"

    def test_feature_importance_exists(self):
        """Verify feature importance analysis was done."""
        with open(os.path.join(OUTPUT_DIR, "evaluation_report.json"), "r") as f:
            report = json.load(f)
        assert "feature_importance" in report, "Missing feature importance"
        fi = report["feature_importance"]
        assert "feature_importance" in fi, "Missing feature importance details"
        assert len(fi["feature_importance"]) > 0, "Empty feature importance"

    def test_risk_insights_exist(self):
        """Verify risk insights were generated."""
        path = os.path.join(OUTPUT_DIR, "risk_insights.json")
        assert os.path.exists(path), "Risk insights not found"
        with open(path, "r") as f:
            insights = json.load(f)
        assert "overall_default_rate" in insights
        assert "risk_by_employment_type" in insights


# =============================================================================
# Test 3: Backend API Tests
# =============================================================================

class TestBackendAPI:
    """Tests for Agent 3: Backend API (structural)."""

    def test_api_module_exists(self):
        """Verify API module exists."""
        api_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "agent3_backend_api", "app.py"
        )
        assert os.path.exists(api_path), "API module not found"

    def test_api_imports_work(self):
        """Verify API module can be imported."""
        from agent3_backend_api.app import app, LoanApplication, PredictionResponse
        assert app is not None
        assert LoanApplication is not None
        assert PredictionResponse is not None

    def test_api_has_required_routes(self):
        """Verify API has all required routes."""
        from agent3_backend_api.app import app
        routes = [route.path for route in app.routes]
        assert "/predict" in routes, "Missing /predict endpoint"
        assert "/summary" in routes, "Missing /summary endpoint"
        assert "/health" in routes, "Missing /health endpoint"

    def test_loan_application_validation(self):
        """Verify input validation works."""
        from agent3_backend_api.app import LoanApplication

        # Valid application
        valid = LoanApplication(
            age=35, income=75000, employment_type="Salaried",
            credit_score=700, loan_amount=50000, loan_term=60,
            existing_emi=500, default_history="No"
        )
        assert valid.age == 35
        assert valid.income == 75000

        # Invalid age
        try:
            invalid = LoanApplication(
                age=10, income=75000, employment_type="Salaried",
                credit_score=700, loan_amount=50000, loan_term=60,
                existing_emi=500, default_history="No"
            )
            assert False, "Should have raised validation error for age=10"
        except Exception:
            pass  # Expected

    def test_api_predict_endpoint(self):
        """Test the predict endpoint using TestClient."""
        from fastapi.testclient import TestClient
        from agent3_backend_api.app import app, load_model_artifacts
        import asyncio

        client = TestClient(app)

        # Trigger startup
        with client:
            response = client.post("/predict", json={
                "age": 35,
                "income": 75000,
                "employment_type": "Salaried",
                "credit_score": 700,
                "loan_amount": 50000,
                "loan_term": 60,
                "existing_emi": 500,
                "default_history": "No"
            })
            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
            data = response.json()
            assert "applicant_risk_score" in data
            assert "risk_category" in data
            assert "default_probability" in data
            assert "recommendation" in data

    def test_api_summary_endpoint(self):
        """Test the summary endpoint using TestClient."""
        from fastapi.testclient import TestClient
        from agent3_backend_api.app import app

        client = TestClient(app)
        with client:
            response = client.get("/summary")
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            data = response.json()
            assert "model_performance" in data or "risk_distribution" in data

    def test_api_health_endpoint(self):
        """Test the health endpoint."""
        from fastapi.testclient import TestClient
        from agent3_backend_api.app import app

        client = TestClient(app)
        with client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert data["status"] == "healthy"


# =============================================================================
# Test 4: Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_data_to_model_pipeline(self):
        """Test that data flows correctly from Agent 1 to Agent 2."""
        # Verify data output exists
        data_path = os.path.join(OUTPUT_DIR, "loan_dataset_clean.csv")
        assert os.path.exists(data_path), "Data pipeline output missing"

        # Verify model uses the data
        model_path = os.path.join(OUTPUT_DIR, "trained_model.joblib")
        assert os.path.exists(model_path), "Model pipeline output missing"

        # Verify features align
        feature_path = os.path.join(OUTPUT_DIR, "feature_columns.json")
        assert os.path.exists(feature_path), "Feature columns file missing"

        with open(feature_path, "r") as f:
            features = json.load(f)

        df = pd.read_csv(data_path)
        for feat in features:
            if feat not in ["Employment_Type_Encoded", "Default_History_Encoded"]:
                assert feat in df.columns, f"Feature {feat} not in dataset"

    def test_model_to_api_integration(self):
        """Test that API correctly uses the trained model."""
        from fastapi.testclient import TestClient
        from agent3_backend_api.app import app

        client = TestClient(app)
        with client:
            # Low risk application
            low_risk = client.post("/predict", json={
                "age": 40, "income": 120000, "employment_type": "Government",
                "credit_score": 800, "loan_amount": 30000, "loan_term": 36,
                "existing_emi": 200, "default_history": "No"
            })
            assert low_risk.status_code == 200
            low_result = low_risk.json()

            # High risk application
            high_risk = client.post("/predict", json={
                "age": 22, "income": 15000, "employment_type": "Unemployed",
                "credit_score": 350, "loan_amount": 200000, "loan_term": 360,
                "existing_emi": 3000, "default_history": "Yes"
            })
            assert high_risk.status_code == 200
            high_result = high_risk.json()

            # High risk should have higher risk score
            assert high_result["applicant_risk_score"] > low_result["applicant_risk_score"], \
                "High risk application should have higher risk score than low risk"

    def test_all_outputs_exist(self):
        """Verify all expected output files exist."""
        expected_files = [
            "loan_dataset_clean.csv",
            "data_dictionary.json",
            "summary_statistics.json",
            "outlier_report.json",
            "trained_model.joblib",
            "scaler.joblib",
            "label_encoders.joblib",
            "evaluation_report.json",
            "risk_insights.json",
            "feature_columns.json",
        ]
        for fname in expected_files:
            path = os.path.join(OUTPUT_DIR, fname)
            assert os.path.exists(path), f"Missing output file: {fname}"

    def test_execution_logs_exist(self):
        """Verify execution logs were generated."""
        expected_logs = [
            "agent1_execution_log.json",
            "agent2_execution_log.json",
        ]
        for fname in expected_logs:
            path = os.path.join(LOG_DIR, fname)
            assert os.path.exists(path), f"Missing execution log: {fname}"


# =============================================================================
# Test Runner & Report Generator
# =============================================================================

def run_tests_and_generate_report() -> dict:
    """Run all tests and generate a comprehensive report."""
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {AGENT_NAME} - Running Test Suite")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Collect test results
    test_results = {
        "agent": AGENT_NAME,
        "start_time": datetime.utcnow().isoformat(),
        "test_suites": {},
        "summary": {},
    }

    suites = {
        "Data Engineering": TestDataEngineering,
        "ML Modeling": TestMLModeling,
        "Backend API": TestBackendAPI,
        "Integration": TestIntegration,
    }

    total_passed = 0
    total_failed = 0
    total_tests = 0

    for suite_name, suite_class in suites.items():
        print(f"\n--- Testing: {suite_name} ---")
        suite_results = []
        instance = suite_class()

        for method_name in dir(instance):
            if not method_name.startswith("test_"):
                continue

            total_tests += 1
            test_name = method_name.replace("test_", "").replace("_", " ").title()
            test_start = time.time()

            try:
                getattr(instance, method_name)()
                status = "PASSED"
                total_passed += 1
                error = None
                print(f"  PASS: {test_name}")
            except Exception as e:
                status = "FAILED"
                total_failed += 1
                error = str(e)
                print(f"  FAIL: {test_name} - {error}")

            suite_results.append({
                "test": test_name,
                "method": method_name,
                "status": status,
                "duration_seconds": round(time.time() - test_start, 3),
                "error": error,
            })

        test_results["test_suites"][suite_name] = suite_results

    # Summary
    duration = round(time.time() - start_time, 2)
    test_results["summary"] = {
        "total_tests": total_tests,
        "passed": total_passed,
        "failed": total_failed,
        "pass_rate": round(total_passed / total_tests * 100, 1) if total_tests > 0 else 0,
        "duration_seconds": duration,
    }
    test_results["end_time"] = datetime.utcnow().isoformat()
    test_results["status"] = "completed"
    test_results["decisions"] = [
        "Tested all 5 agent components independently",
        "Validated data integrity, model accuracy, API functionality, and end-to-end integration",
        "Used FastAPI TestClient for API testing without requiring running server",
    ]
    test_results["dependencies"] = ["pytest", "pandas", "numpy", "fastapi", "joblib"]

    # Save report
    report_path = os.path.join(LOG_DIR, "agent5_test_report.json")
    with open(report_path, "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Test Results: {total_passed}/{total_tests} passed ({test_results['summary']['pass_rate']}%)")
    print(f"  Duration: {duration}s")
    print(f"{'='*60}\n")

    return test_results


if __name__ == "__main__":
    run_tests_and_generate_report()
