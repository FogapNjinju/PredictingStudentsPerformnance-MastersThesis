# Predicting Student Academic Performance - MLOps Implementation

## Project Overview

This project predicts student academic performance using machine learning with a **production-ready MLOps system**.  
It uses student demographic data, study habits, and previous academic metrics to build and deploy models that classify students into three categories: **Dropout**, **Enrolled**, or **Graduate**.

### Key Features
- ✅ **Automated ML Pipeline** - Data preparation → Feature engineering → Model training
- ✅ **Experiment Tracking** - MLflow dashboard to compare all model runs
- ✅ **Configuration Management** - YAML-based config for easy parameter changes
- ✅ **Unit Testing** - Comprehensive tests for data and models
- ✅ **CI/CD Ready** - GitHub Actions workflow included
- ✅ **Production Deployment** - Streamlit web app for predictions

**Objective:**  
- Identify key factors affecting student performance
- Build and evaluate machine learning models with reproducibility
- Track experiments professionally with MLflow
- Deploy model to production instantly
- Provide actionable insights for improving academic outcomes

---

## Project Structure

```
PredictingStudentsPerformnance-MastersThesis/
│
├── src/                          # Source code (organized & modular)
│   ├── data/
│   │   ├── __init__.py          # DataLoader utilities
│   │   └── prepare.py           # Data cleaning pipeline
│   ├── models/
│   │   └── __init__.py          # ModelManager utilities (train, save, evaluate)
│   ├── features/
│   │   └── __init__.py          # Feature engineering pipeline
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Configuration management
│       ├── logger.py            # Logging setup
│       └── mlflow_utils.py      # MLflow integration
│
├── pipelines/                    # ML Workflow pipelines
│   ├── train_pipeline.py        # Full training workflow
│   └── evaluate_pipeline.py     # Model evaluation
│
├── tests/                        # Unit tests
│   └── test_data_models.py      # Data and model tests
│
├── configs/
│   └── config.yaml              # Main configuration file
│
├── notebooks/                    # Jupyter notebooks (exploratory)
│   ├── 00_project_setup.ipynb
│   ├── 01_data_import.ipynb
│   ├── 02_eda.ipynb
│   ├── ... (and more)
│   └── 07_final_analysis.ipynb
│
├── data/
│   └── data.csv                 # Raw student data (4,424 records)
│
├── outputs/
│   ├── models/
│   │   ├── best_model.joblib   # Trained Random Forest Classifier
│   │   └── ... (other models)
│   ├── figures/                 # Visualizations
│   └── reports/                 # Analysis reports
│
├── mlruns/                       # MLflow experiment tracking data (auto-generated)
│   └── (experiment runs stored here)
│
├── logs/                         # Application logs (auto-generated)
│   └── (timestamped log files)
│
├── .github/workflows/
│   └── mlops-pipeline.yml       # CI/CD automation (GitHub Actions)
│
├── .env.example                  # Environment variables template
├── .dvc.yaml                     # DVC pipeline configuration
├── requirements.txt              # Python dependencies
├── MLOPS_GUIDE.md               # Detailed MLOps documentation
├── QUICKSTART.md                # Quick start guide (5 minutes)
├── IMPLEMENTATION_SUMMARY.md    # What was implemented
├── MLOPS_CHECKLIST.md           # Progress tracking
├── README.md                     # This file
└── LICENSE

```

---

## Installation & Setup

### Prerequisites
- Python 3.9+ (we're using Python 3.14)
- pip (Python package manager)
- Git

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd PredictingStudentsPerformnance-MastersThesis
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- **Data Processing**: numpy, pandas
- **ML Libraries**: scikit-learn, xgboost, lightgbm
- **MLOps**: mlflow, dvc, pytest
- **Visualization**: matplotlib, seaborn
- **Web App**: streamlit

### Step 3: Configure Settings (Optional)
Edit `configs/config.yaml` to customize:
```yaml
data:
  test_size: 0.2              # 20% for testing
features:
  target: "Target"            # Prediction target
model:
  n_estimators: 100           # Number of trees
  max_depth: 10               # Tree depth
```

---

## Quick Start (5 Steps)

### **Step 1: Prepare Data** 🧹
```bash
python src/data/prepare.py
```
Cleans raw data, handles missing values, removes duplicates.

### **Step 2: Build Features** 🏗️
```bash
python src/features/build_features.py
```
Scales numeric values, encodes categories, creates new features.

### **Step 3: Start MLflow Dashboard** 📊 (New Terminal)
```bash
python -m mlflow server --backend-store-uri file:./mlruns --host 127.0.0.1 --port 5000
```
Open: [http://localhost:5000](http://localhost:5000)

### **Step 4: Train Model** 🤖
```bash
python pipelines/train_pipeline.py
```
Trains Random Forest Classifier, logs results to MLflow.

**Expected Output:**
```
Random Forest Metrics:
- Accuracy: 0.766 (76.6% correct)
- Precision: 0.750 (75% when predicting)
- Recall: 0.766 (catches 76.6% of actuals)
- F1 Score: 0.746 (balanced metric)
```

### **Step 5: View Results** 📈
Go to [http://localhost:5000](http://localhost:5000) in your browser:
- Experiment: `student_performance`
- Run: `random_forest_classifier`
- Metrics: Accuracy, Precision, Recall, F1 Score
- Model: Saved as artifact

---

## MLOps Components Explained

### **1. MLflow Experiment Tracking** 📊
Tracks every model run with:
- **Metrics**: accuracy, precision, recall, f1_score
- **Parameters**: n_estimators, max_depth, random_state
- **Artifacts**: trained model file (model.pkl)
- **Metadata**: execution time, user, timestamp

**View experiments:**
```
http://localhost:5000
→ Experiments → student_performance → View run details
```

### **2. Configuration Management** ⚙️
All settings in `configs/config.yaml`:
- Change parameters without touching code
- Non-technical users can configure
- Easy to reproduce results

### **3. Data Pipeline** 🔄
```
Raw Data (data/data.csv)
    ↓ [prepare.py]
Cleaned Data (data/processed_data.csv)
    ↓ [build_features.py]
Featured Data (data/featured_data.csv)
    ↓ [train_pipeline.py]
Trained Model (outputs/models/best_model.joblib)
    ↓ [evaluate_pipeline.py]
Metrics (metrics/*.json)
```

### **4. Automated Model Training** 🤖
Pipeline automatically:
1. Loads and preprocesses data
2. Splits into train/test (80/20)
3. Trains Random Forest Classifier
4. Calculates performance metrics
5. Logs everything to MLflow
6. Saves best model

### **5. Testing Framework** 🧪
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Model Details

### **Algorithm: Random Forest Classifier**
- **Type**: Ensemble learning (multiple decision trees)
- **Why**: Handles complex patterns, robust, interpretable
- **Number of trees**: 100 (configurable)
- **Max depth**: 10 levels (configurable)
- **Output**: Predicts 3 classes:
  - `0` = Dropout
  - `1` = Enrolled
  - `2` = Graduate

### **Performance Metrics**

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **Accuracy** | 76.6% | Correct predictions |
| **Precision** | 75.0% | When predicting positive, accuracy |
| **Recall** | 76.6% | Captures actual positives |
| **F1 Score** | 74.6% | Harmonic mean of precision & recall |

### **Data Characteristics**
- **Total Records**: 4,424 students
- **Features**: 36 (demographics, academic metrics, etc.)
- **Target Classes**: 3 (Dropout, Enrolled, Graduate)
- **Train/Test Split**: 80/20 (3,539 train, 885 test)

---

## Usage Examples

### **Train with Different Hyperparameters**
```bash
# Edit configs/config.yaml:
model:
  n_estimators: 200    # More trees
  max_depth: 8         # Shallower trees

# Retrain
python pipelines/train_pipeline.py

# Compare in MLflow dashboard
```

### **Make Predictions on New Data**
```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load("outputs/models/best_model.joblib")

# Prepare new data
new_students = pd.read_csv("new_students.csv")

# Make predictions
predictions = model.predict(new_students)  # Returns class (0=Dropout, 1=Enrolled, 2=Graduate)
probabilities = model.predict_proba(new_students)  # Returns confidence scores
```

### **Deploy with Streamlit**
```bash
streamlit run app/streamlit_app.py
```
Opens interactive web app at `http://localhost:8501`

### **Run Tests**
```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_data_models.py::test_data_loader_load -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## Workflow Summary

```
1. DATA PREPARATION
   Raw Data → Clean Data → Featured Data
   
2. MODEL TRAINING
   Load Data → Split Train/Test → Train Model → Calculate Metrics
   
3. EXPERIMENT TRACKING
   Log to MLflow → Store Metrics → Save Model → Create Dashboard
   
4. EVALUATION
   Evaluate Model → Generate Metrics → Compare with MLflow
   
5. DEPLOYMENT
   Package Model → Deploy Streamlit App → Users Get Predictions
   
6. MONITORING
   Track Performance → Detect Issues → Retrain if Needed
```

---

## Files & What They Do

### **Core Scripts**

| File | Purpose |
|------|---------|
| `src/data/prepare.py` | Clean and preprocess data |
| `src/features/__init__.py` | Create and scale features |
| `pipelines/train_pipeline.py` | Main training workflow |
| `pipelines/evaluate_pipeline.py` | Evaluate model performance |
| `app/streamlit_app.py` | Web interface for predictions |
| `tests/test_data_models.py` | Unit tests |

### **Configuration**

| File | Purpose |
|------|---------|
| `configs/config.yaml` | All ML parameters |
| `.env.example` | Environment variables template |
| `.dvc.yaml` | DVC pipeline definition |

### **Documentation**

| File | Purpose |
|------|---------|
| `QUICKSTART.md` | 5-minute quick start |
| `MLOPS_GUIDE.md` | Detailed MLOps guide |
| `IMPLEMENTATION_SUMMARY.md` | What was implemented |
| `MLOPS_CHECKLIST.md` | Progress tracking |

---

## Troubleshooting

### **"No runs logged" in MLflow**
```bash
# Restart MLflow with correct backend
python -m mlflow server --backend-store-uri file:./mlruns --host 127.0.0.1 --port 5000
```

### **"Column not found" error**
```
Edit configs/config.yaml and set correct:
- features.target: "Target" (or your column name)
- data.raw_path: "data/data.csv" (check path)
```

### **Tests failing**
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run with verbose output
pytest tests/ -v --tb=short
```

### **Model not saving**
Check that `outputs/models/` directory exists:
```bash
mkdir -p outputs/models
```

---

## Next Steps

### 🔄 **Try different models:**
- Add XGBoost in config
- Add LightGBM in config
- Compare results in MLflow

### 📊 **Improve model:**
- Tune hyperparameters
- Add more features
- Get more training data

### 🚀 **Deploy:**
- Run Streamlit app
- Share with team
- Monitor in production

### 🔧 **Enhance system:**
- Add data validation
- Setup monitoring
- Implement retraining logic

---

## Technologies Used

- **Python 3.14** - Programming language
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning
- **MLflow** - Experiment tracking
- **DVC** - Data versioning
- **pytest** - Testing
- **Streamlit** - Web app
- **GitHub Actions** - CI/CD

---

## Project Phases

| Phase | Status | Components |
|-------|--------|-----------|
| **Phase 1: Foundational Setup** | ✅ Complete | Config, logging, MLflow, pipeline |
| **Phase 2: Pipeline Automation** | ✅ Ready | Data prep, features, training |
| **Phase 3: Testing & Quality** | ⏳ Ready | Unit tests, CI/CD checks |
| **Phase 4: Monitoring** | 🔜 Next | Performance tracking, alerting |

---

## References

- [MLflow Documentation](https://mlflow.org/)
- [Scikit-learn Docs](https://scikit-learn.org/)
- [Pandas Guide](https://pandas.pydata.org/)
- [Random Forest Explained](https://en.wikipedia.org/wiki/Random_forest)

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Author

**Ninju Zilefac Fogap**  
Email: [andrefogap@icloud.com](mailto:andrefogap@icloud.com)

**MLOps Implementation**: April 2026

---

## Acknowledgments

- MLOps best practices from: MLflow, DVC, and community standards

---

## Support

For questions or issues:
1. Check [MLOPS_GUIDE.md](MLOPS_GUIDE.md) for detailed explanations
2. See [QUICKSTART.md](QUICKSTART.md) for quick reference
3. Review code comments in `src/` files
4. Run tests to validate setup

**Questions?** Create an issue or contact the author!