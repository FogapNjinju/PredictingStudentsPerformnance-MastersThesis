# Predicting Student Academic Performance Using Machine Learning

## Project Overview
This project aims to predict student academic performance using machine learning techniques.  
It uses student demographic data, study habits, attendance, and previous grades to build predictive models.  

**Objective:**  
- Identify key factors affecting student performance.  
- Build and evaluate machine learning models to predict final grades.  
- Provide actionable insights for improving academic outcomes.

## Project Structure
```
StudentPerformanceML/
│
├── notebooks/          # Jupyter notebooks for each stage
├── data/               # Raw and processed datasets
├── output/
│   ├── models/         # Saved trained models
│   ├── figures/        # Plots and visualizations
│   └── reports/        # Evaluation metrics and summary tables
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .gitignore          # Files to ignore in Git
```

## Notebooks
1. `00_project_setup.ipynb` – Environment setup and project overview  
2. `01_data_import.ipynb` – Load and inspect datasets  
3. `02_data_preprocessing.ipynb` – Data cleaning and preprocessing  
4. `03_eda.ipynb` – Exploratory Data Analysis (EDA)  
5. `04_feature_engineering.ipynb` – Feature creation and selection  
6. `05_model_training.ipynb` – Train ML models  
7. `06_model_evaluation.ipynb` – Evaluate model performance  
8. `07_final_analysis.ipynb` – Summarize insights and results  

## Dependencies
All Python dependencies are listed in `requirements.txt`. Example:
```
numpy
pandas
scikit-learn
matplotlib
seaborn
xgboost
```
Install via:
```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
```bash
git clone <repository-url>
cd StudentPerformanceML
```
2. Open notebooks in `notebooks/` in Jupyter or VSCode.  
3. Follow the numbered sequence for reproducibility (`00_` → `07_`).  

## Notes
- Keep raw datasets untouched; use preprocessed versions for modeling.  
- Save all plots in `output/figures/` and model artifacts in `output/models/`.  
- Relative paths are used in notebooks to ensure portability.  

## Author
Ninju Zilefac Fogap – [andrefogap@icloud.com]