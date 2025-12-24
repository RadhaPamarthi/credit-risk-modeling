# Credit Risk Modeling - Best Egg Take-Home Assignment

## Objective

Build a production-ready ML pipeline to predict the probability of 12-month loan default at application time.

---

## Project Structure
```
credit-risk-modeling/
│
├── README.md
├── Makefile
├── requirements.txt
├── .gitignore
│
├── config/
│   └── config.yaml                 # Environment-aware configuration (dev/staging/prod)
│
├── data/
│   └── credit_risk_data_enhanced.csv
│
├── notebooks/
│   └── modeling.ipynb              # Experimentation notebook (with outputs)
│
├── src/                            # Core library modules
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── modeling.py
│   └── evaluation.py
│
├── flow/
│   └── credit_risk_flow.py         # Metaflow training pipeline
│
├── scripts/
│   └── score_production.py         # Production scoring pipeline
│
├── tests/
│   └── test_modules.py
│
└── outputs/
    ├── models/                     # Trained model artifacts (.pkl, .json)
    ├── plots/                      # Visualization outputs (.png)
    └── scoring/                    # Production scoring results
```

---

## Quick Start

### 1. Setup Environment
```bash
# Using Makefile
make setup

# Or manually
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Training (Development Mode)
```bash
# Using Makefile
make train 


```
**Option 1: Metaflow Pipeline (Recommended)**
```bash
# Or directly
python flow/credit_risk_flow.py run --data_path data/credit_risk_data_enhanced.csv
```
**Option 2: Jupyter Notebook**
```bash
jupyter notebook notebooks/modeling.ipynb
```

### 3. Run Production Scoring
```bash
python scripts/score_production.py \
    --config config/config.yaml \
    --input data/credit_risk_data_enhanced.csv \
    --output outputs/scoring/scores.parquet
```

### 4. Run Tests
```bash
make test
```

---

## Environment Configuration

The pipeline supports three environments: `dev`, `staging`, `prod`

### Setting Environment
```bash
# Development (default) - uses local filesystem
export ENV=dev
make train

# Staging - uses S3 staging bucket
export ENV=staging
make train

# Production - uses S3 prod bucket, writes to Redshift
export ENV=prod
make train
```

### Environment-Specific Behavior

| Component | Dev | Staging | Prod |
|-----------|-----|---------|------|
| Model Storage | Local `outputs/models/` | S3 `s3://bestegg-ml-staging/` | S3 `s3://bestegg-ml-prod/` |
| Scoring Output | Local Parquet | S3 Parquet | Redshift table |
| Logging | DEBUG | INFO | WARNING |
| Monitoring | Disabled | Enabled | Enabled + Alerts |

### Configuration File

All settings are in `config/config.yaml`:
```yaml
storage:
  dev:
    type: "local"
    model_dir: "outputs/models"
    
  prod:
    type: "s3"
    model_dir: "s3://bestegg-ml-prod/credit-risk/models"

scoring:
  output_format:
    dev: "parquet"
    prod: "redshift"
    
  redshift:
    prod:
      enabled: true
      schema: "ml_scores"
      table: "credit_risk_scores"
```

---

## Makefile Commands
```bash
make setup    # Create venv and install dependencies
make train    # Train model (uses ENV variable)
make score    # Run production scoring
make test     # Run unit tests
make lint     # Run code linting
make clean    # Remove cache files
```

---

## Training Pipeline (Metaflow)

### Pipeline Steps
```
start --> preprocess --> feature_engineering --> split_data --> train_models --> evaluate --> end
```

### Running the Training Pipeline
```bash
# Basic run (dev mode)
python flow/credit_risk_flow.py run --data_path data/credit_risk_data_enhanced.csv

# With custom parameters
python flow/credit_risk_flow.py run \
    --train_vintage_end 202212 \
    --test_vintage_start 202401
```

---

## Production Scoring Pipeline

### How Production Scoring Works

1. Load model artifacts from storage (local or S3)
2. Transform raw application data (handle special values, engineer features)
3. Generate probability scores using trained model
4. Apply decision thresholds
5. Output results to configured destination (Parquet or Redshift)

### Running Production Scoring
```bash
# Development mode
python scripts/score_production.py \
    --config config/config.yaml \
    --input new_applications.csv \
    --output scores.parquet

# Production mode (writes to Redshift)
ENV=prod python scripts/score_production.py \
    --config config/config.yaml \
    --input s3://bucket/applications.parquet \
    --output outputs/scoring/scores.parquet
```

### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| loan_id | string | Original loan identifier |
| pd_score | float | Probability of default (0-1) |
| risk_decile | int | Risk decile (1=lowest, 10=highest) |
| credit_decision | string | APPROVE / DECLINE / REVIEW |
| scored_at | timestamp | When scoring occurred |
| model_version | string | Model version used |

### Decision Logic
```yaml
decision_thresholds:
    approve_below: 0.03    # PD < 3% -> Auto-approve
    decline_above: 0.15    # PD > 15% -> Auto-decline
    # 3% <= PD <= 15% -> Manual review
```

---

## Key Findings

### Data Quality Issues Identified

| Issue | Column | Special Value | Action |
|-------|--------|---------------|--------|
| Missing FICO | fico_score | 99999 | Created missing indicator + median imputation |
| Missing Income | income | -1 | Created missing indicator + median imputation |
| Missing Inquiries | inquiries_last_6m | 99 | Created missing indicator + median imputation |

Key Insight: Missingness is INFORMATIVE - borrowers with missing data have higher default rates.

### Data Leakage Detected

| Feature | Problem | Action |
|---------|---------|--------|
| days_past_due_current | IS the target (AUC=1.0) | Removed |
| total_payments_to_date | Post-origination data | Removed |
| months_on_book | Would be 0 at application | Removed |

### Model Performance

| Model | Train AUC | Test AUC | Test KS | Overfit Gap |
|-------|-----------|----------|---------|-------------|
| Logistic Regression | 0.6879 | 0.6618 | 0.2667 | 0.0261 |
| XGBoost | 0.7704 | 0.6866 | 0.2792 | 0.0838 |

Recommendation: XGBoost for production deployment.

### Feature Importance (Top 10)

1. fico_score
2. fico_score_missing
3. debt_to_income
4. debt_burden_score
5. inquiries_last_6m_missing
6. income_missing
7. utilization_rate
8. income
9. num_open_trades
10. loan_to_income

---

## Running Tests
```bash
# Run all tests
make test

# Run specific test
pytest tests/test_modules.py -v
```

---

## Evaluation Criteria Addressed

| Criteria | How Addressed |
|----------|---------------|
| Technical Approach | XGBoost + Logistic Regression, proper validation |
| Data Understanding | Special values, leakage detection, temporal analysis |
| Feature Engineering | 20 meaningful features, missing indicators, composites |
| Model Performance | AUC, KS, calibration, PR curves |
| Code Quality | Modular src/, docstrings, type hints |
| Communication | Visualizations, clear explanations |
| Production Readiness | Metaflow pipeline, config-driven, scoring pipeline |

### Bonus Section

| Criteria | How Addressed |
|----------|---------------|
| Engineering Best Practices | Separate modules, unit tests, Makefile |
| Workflow Management | Metaflow with parameters, artifact storage |
| Environment Support | Dev/Staging/Prod configs, S3/Redshift integration |

---

## Recommendations for Improvement

1. More Data Sources
   - Bureau tradelines (payment history)
   - Credit history length
   - Alternative data (bank transactions)

2. Model Enhancements
   - Hyperparameter tuning with Optuna
   - Ensemble methods
   - Calibration adjustment

3. Monitoring
   - Population Stability Index (PSI)
   - Performance decay alerts
   - Feature drift detection

---

## Author

Radhakrishna
Best Egg Take-Home Assignment
December 2025