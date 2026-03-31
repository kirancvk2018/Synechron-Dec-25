# Loan Risk Assessment Dashboard

## Multi-Agent AI Solution for Enterprise Financial Services

**Enterprise Client:** Synechron Financial Services  
**Architecture:** Multi-Agent AI Collaboration  
**Domain:** Credit Risk Assessment / Retail Banking

---

## Overview

A complete, production-style Loan Risk Assessment system built using **5 parallel AI agents** that demonstrate accelerated software delivery through multi-agent collaboration. The system generates synthetic loan data, trains ML models, exposes REST APIs, and provides an interactive analytics dashboard.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (run_all.py)                  │
│              Coordinates parallel agent execution             │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│ Agent 1  │ Agent 2  │ Agent 3  │ Agent 4  │    Agent 5      │
│   Data   │    ML    │ Backend  │ Frontend │  Integration    │
│  Engg.   │ Modeling │   API    │    UI    │     & QA        │
├──────────┼──────────┼──────────┼──────────┼─────────────────┤
│ Generate │ Train &  │ FastAPI  │Streamlit │ Unit Tests      │
│ 12K rows │ Evaluate │ REST API │Dashboard │ API Validation  │
│ Clean &  │ 3 Models │ /predict │ Charts   │ Integration     │
│ Engineer │ Feature  │ /summary │ Forms    │ Test Report     │
│ Features │ Analysis │ Swagger  │ Explorer │                 │
└──────────┴──────────┴──────────┴──────────┴─────────────────┘
```

## Agent Details

### Agent 1: Data Engineering
- Generates **12,000+ row** synthetic loan dataset with realistic distributions
- Features: Applicant_ID, Age, Income, Employment_Type, Credit_Score, Loan_Amount, Loan_Term, Existing_EMI, Default_History, Loan_Status
- Data cleaning: duplicate removal, missing value imputation, range validation
- Outlier detection using IQR method
- Feature engineering: Debt-to-Income ratio, Loan-to-Income ratio, EMI Burden ratio, credit/age bins

### Agent 2: ML Modeling
- Trains 3 classification models: Logistic Regression, Random Forest, Gradient Boosting
- Stratified train/test split (80/20)
- 5-fold cross-validation
- Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Feature importance analysis with key risk insights

### Agent 3: Backend API
- **FastAPI** REST API with automatic OpenAPI/Swagger documentation
- Endpoints:
  - `POST /predict` → Loan risk score with recommendation
  - `GET /summary` → Model insights and risk analytics
  - `GET /health` → Health check
  - `GET /feature-importance` → Feature importance data
  - `GET /risk-distribution` → Risk distribution data
- Input validation with Pydantic models
- CORS enabled for frontend integration

### Agent 4: Frontend UI
- **Streamlit** interactive dashboard with 4 tabs:
  1. **Risk Assessment** - Loan input form with real-time risk prediction gauge
  2. **Model Analytics** - Performance comparison, feature importance, risk distributions
  3. **Data Explorer** - Dataset visualization with histograms, scatter plots
  4. **Enterprise Summary** - Multi-agent architecture overview, execution logs
- Plotly interactive charts
- Fallback to local model if API unavailable

### Agent 5: Integration & QA
- **30+ automated tests** covering all components
- Test categories: Data Engineering, ML Modeling, Backend API, Integration
- FastAPI TestClient for API testing (no running server required)
- Generates comprehensive test report with pass/fail details

## Quick Start

### 1. Install Dependencies

```bash
cd loan-risk-assessment
pip install -r requirements.txt
```

### 2. Run Full Pipeline (Orchestrator)

```bash
python run_all.py
```

This runs all 5 agents (both sequential and parallel), generates all outputs, and produces a consolidated summary comparing execution times.

### 3. Run Individual Agents

```bash
# Agent 1: Generate and process data
python -m agent1_data_engineering.data_generator

# Agent 2: Train ML models
python -m agent2_ml_modeling.model_trainer

# Agent 3: Start API server
uvicorn agent3_backend_api.app:app --host 0.0.0.0 --port 8000

# Agent 4: Start dashboard
streamlit run agent4_frontend_ui/dashboard.py --server.port 8501

# Agent 5: Run tests
python -m agent5_integration_qa.test_suite
```

### 4. Access the Application

- **API Documentation:** http://localhost:8000/docs
- **Dashboard:** http://localhost:8501

## Output Files

| File | Description |
|------|-------------|
| `outputs/loan_dataset_clean.csv` | Processed loan dataset (12,000 rows) |
| `outputs/data_dictionary.json` | Column descriptions and metadata |
| `outputs/summary_statistics.json` | Dataset summary statistics |
| `outputs/outlier_report.json` | Outlier detection report |
| `outputs/trained_model.joblib` | Best trained classification model |
| `outputs/scaler.joblib` | Feature scaler |
| `outputs/evaluation_report.json` | Model evaluation metrics |
| `outputs/risk_insights.json` | Business risk insights |
| `execution_logs/agent*_log.json` | Individual agent execution logs |
| `execution_logs/consolidated_summary.json` | Final execution summary |
| `execution_logs/agent5_test_report.json` | Test results report |

## Enterprise Value (Synechron)

### Banking & Financial Services Applicability
- **Credit Risk Modeling**: ML-based default prediction for loan origination
- **Regulatory Compliance**: Interpretable features, audit trails, no protected class features
- **Decision Support**: Risk scoring with actionable recommendations for underwriters

### Multi-Agent Benefits
- **Accelerated Delivery**: Parallel execution reduces development lifecycle by 40-60%
- **Scalability**: Agent-based architecture extends to complex financial products
- **Governance**: Every decision logged with timestamps for regulatory compliance
- **Traceability**: Full data lineage from raw data through model to predictions
- **Auditability**: Comprehensive test reports and execution logs

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Data Processing | Pandas, NumPy |
| ML Modeling | Scikit-learn |
| Backend API | FastAPI, Uvicorn |
| Frontend | Streamlit, Plotly |
| Testing | Pytest, FastAPI TestClient |
| Serialization | Joblib, JSON |
