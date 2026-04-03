# MLOps Implementation Guide

## Overview
This guide explains the MLOps implementation for the Student Performance Prediction project.

## Key Components

### 1. **Experiment Tracking (MLflow)**

MLflow tracks all model experiments, hyperparameters, and metrics.

#### Setup MLflow Server Locally:
```bash
mlflow ui --backend-store-uri ./mlruns
```

Then visit `http://localhost:5000` to view your experiments.

#### How It Works:
- Each model training run is logged to MLflow
- Metrics, parameters, and models are tracked
- Easy comparison across different experiments
- Model registry for version management

### 2. **Data Versioning (DVC)**

DVC tracks data and model files without storing them in Git.

#### Initialize DVC:
```bash
dvc init
dvc config core.autostage true
```

#### Track Data Files:
```bash
dvc add data/data.csv
git add data/data.csv.dvc .gitignore
```

### 3. **Configuration Management**

Configuration is controlled via `configs/config.yaml`:
- Data paths and parameters
- Model hyperparameters
- Evaluation metrics
- MLflow settings

Modify this file to change training behavior without changing code.

### 4. **Project Structure**

```
project/
├── src/                    # Source code
│   ├── data/              # Data loading and processing
│   ├── models/            # Model utilities
│   ├── features/          # Feature engineering
│   └── utils/             # Configuration, logging, MLflow integration
├── pipelines/             # Training and evaluation pipelines
├── tests/                 # Unit tests
├── configs/               # Configuration files
├── notebooks/             # Jupyter notebooks (exploratory)
├── outputs/               # Models and figures
├── .github/workflows/     # CI/CD pipelines
└── requirements.txt       # Dependencies
```

## Workflow

### Step 1: Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Copy .env.example to .env and configure
cp .env.example .env
```

### Step 2: Prepare Data

```bash
python src/data/prepare.py
```

### Step 3: Build Features

```bash
python src/features/build_features.py
```

### Step 4: Train Models

```bash
# Start MLflow server (in a separate terminal)
mlflow ui --backend-store-uri ./mlruns

# Run training pipeline
python pipelines/train_pipeline.py
```

### Step 5: Evaluate Models

```bash
python pipelines/evaluate_pipeline.py
```

### Step 6: View Results

Visit MLflow UI at `http://localhost:5000` to:
- Compare model runs
- View metrics and plots
- Download best model

## Testing

Run all tests:
```bash
pytest tests/ -v --cov=src
```

Run specific test:
```bash
pytest tests/test_data_models.py::test_data_loader_load -v
```

## Deployment

The model is deployed via the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

## CI/CD Pipeline

GitHub Actions automatically:
1. Runs tests on every push/PR
2. Trains models on main branch
3. Checks code quality (flake8, black, pylint)
4. Uploads artifacts

Configure by editing `.github/workflows/mlops-pipeline.yml`

## Monitoring & Logging

All runs are logged to:
- `logs/` directory (local file logs)
- MLflow (metrics and models)
- Console output

## Best Practices

✅ **Do:**
- Track all experiments in MLflow
- Use configuration files (don't hardcode)
- Write tests for critical functions
- Version control code and configurations
- Document changes in commits

❌ **Don't:**
- Hardcode hyperparameters in code
- Store large data files in Git (use DVC)
- Skip testing before pushing
- Train without logging metrics
- Use different configs locally vs. production

## Troubleshooting

### MLflow server not starting
```bash
# Check if port 5000 is in use
lsof -i :5000  # Linux/Mac
netstat -ano | findstr :5000  # Windows
```

### Models not saving
- Check `outputs/models/` directory exists
- Verify disk space
- Check file write permissions

### Tests failing
- Update test data in `tests/test_data_models.py`
- Install test dependencies: `pip install pytest pytest-cov`
- Run with verbose flag: `pytest -v`

## Next Steps

- **Phase 2**: Implement automated data pipelines
- **Phase 3**: Add comprehensive testing suite
- **Phase 4**: Setup production deployment with monitoring

For questions, refer to MLflow docs: https://mlflow.org/
