# MLOps Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Your Project
Edit `configs/config.yaml` and update:
- `features.target`: Your target column name (e.g., "Grade", "Final_Score")
- `data.raw_path`: Path to your CSV file
- Model parameters if needed

### 3. Start MLflow Dashboard
In a **new terminal**:
```bash
mlflow ui --backend-store-uri ./mlruns
```
Then open: http://localhost:5000

### 4. Run Data Preparation
```bash
python src/data/prepare.py
```

### 5. Build Features
```bash
python src/features/build_features.py
```

### 6. Train Your First Model
```bash
python pipelines/train_pipeline.py
```

### 7. View Results
- Check the MLflow dashboard at http://localhost:5000
- See logs in the `logs/` directory
- Models saved in `outputs/models/`

---

## 📊 Check Your Results

After training, you should see:
- ✅ Metrics logged to MLflow
- ✅ Model saved to `outputs/models/best_model.joblib`
- ✅ Logs in `logs/` directory
- ✅ Metrics JSON in `metrics/` directory

---

## 🧪 Run Tests
```bash
pytest tests/ -v
```

---

## 📝 Common Tasks

### Change Model Hyperparameters
Edit `configs/config.yaml`:
```yaml
model:
  models_to_train:
    - name: "random_forest"
      params:
        n_estimators: 200  # Change this
        max_depth: 15      # And this
```

### View Experiment History
Go to: http://localhost:5000
- Filter by date or metric
- Compare different runs
- Download best model

### Evaluate Model
```bash
python pipelines/evaluate_pipeline.py
```

### Deploy with Streamlit
```bash
streamlit run app/streamlit_app.py
```

---

## 🆘 Troubleshooting

**'Column not found' error**
- Open `configs/config.yaml`
- Set correct target column name under `features.target`

**MLflow dashboard not loading**
- Kill other processes on port 5000: `lsof -i :5000` (Mac) or `netstat -ano | findstr :5000` (Windows)

**Tests failing**
```bash
pytest --tb=short  # See more details
pytest tests/test_data_models.py -v  # Run specific test
```

---

## 📚 Next Steps

1. ✅ **Phase 1 Complete**: Foundational setup
2. ⏳ **Phase 2**: Automate data pipelines
3. ⏳ **Phase 3**: Add monitoring
4. ⏳ **Phase 4**: Deploy to production

For detailed info, see [MLOPS_GUIDE.md](MLOPS_GUIDE.md)
