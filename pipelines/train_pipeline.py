"""Main training pipeline using MLOps best practices"""

import sys
from pathlib import Path
import yaml
import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import logger
from src.utils.config import config
from src.utils.mlflow_utils import mlflow_logger
from src.data import DataLoader
from src.models import ModelManager

class TrainingPipeline:
    """End-to-end training pipeline"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize pipeline with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info("Training Pipeline initialized")
    
    def run(self):
        """Run the complete training pipeline"""
        try:
            logger.info("Starting training pipeline")
            
            # Step 1: Load data
            logger.info("Step 1: Loading data")
            df = DataLoader.load_data(self.config['data']['raw_path'])
            
            # Step 2: Get feature info
            feature_info = DataLoader.get_feature_info(df)
            logger.info(f"Features: {feature_info}")
            
            # Step 3: Split data
            logger.info("Step 2: Splitting data")
            target = self.config['features']['target']
            
            if target not in df.columns:
                logger.warning(f"Target column '{target}' not found. Using last column.")
                target = df.columns[-1]
            
            # Encode target if categorical
            df_copy = df.copy()
            if df[target].dtype == 'object':
                le = LabelEncoder()
                df_copy[target] = le.fit_transform(df[target])
                logger.info(f"Encoded target classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")
            
            # Drop non-numeric columns except target
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
            if target in numeric_cols:
                numeric_cols.remove(target)
            
            X = df_copy[numeric_cols]
            y = df_copy[target]
            
            # Split data
            X_train, X_test, y_train, y_test = DataLoader.split_data(
                pd.concat([X, y], axis=1),
                target_column=target,
                test_size=self.config['data']['test_size'],
                random_state=self.config['data']['random_state']
            )
            
            # Step 4: Train model with MLflow
            logger.info("Step 3: Training models with MLflow")
            
            model_results = {}
            
            # Train Random Forest Classifier
            mlflow_logger.start_run(
                run_name="random_forest_classifier",
                tags={"model_type": "RandomForestClassifier", "pipeline": "main"}
            )
            
            rf_model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(
                    n_estimators=self.config['model']['models_to_train'][0]['params']['n_estimators'],
                    max_depth=self.config['model']['models_to_train'][0]['params']['max_depth'],
                    random_state=self.config['model']['models_to_train'][0]['params']['random_state'],
                    n_jobs=-1
                ))
            ])
            
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)
            
            # Calculate classification metrics
            metrics_rf = {
                "accuracy": accuracy_score(y_test, y_pred_rf),
                "precision": precision_score(y_test, y_pred_rf, zero_division=0, average='weighted'),
                "recall": recall_score(y_test, y_pred_rf, zero_division=0, average='weighted'),
                "f1_score": f1_score(y_test, y_pred_rf, zero_division=0, average='weighted')
            }
            
            mlflow_logger.log_params(self.config['model']['models_to_train'][0]['params'])
            mlflow_logger.log_metrics(metrics_rf)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(rf_model, "model")
            model_results['random_forest'] = metrics_rf
            
            logger.info(f"Random Forest Metrics: {metrics_rf}")
            mlflow_logger.end_run()
            
            # Train XGBoost Classifier
            mlflow_logger.start_run(
                run_name="xgboost_classifier",
                tags={"model_type": "XGBoostClassifier", "pipeline": "main"}
            )
            
            xgb_model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', xgb.XGBClassifier(
                    n_estimators=self.config['model']['models_to_train'][1]['params']['n_estimators'],
                    max_depth=self.config['model']['models_to_train'][1]['params']['max_depth'],
                    learning_rate=self.config['model']['models_to_train'][1]['params']['learning_rate'],
                    random_state=42,
                    verbosity=0,
                    use_label_encoder=False,
                    eval_metric='mlogloss'
                ))
            ])
            
            xgb_model.fit(X_train, y_train)
            y_pred_xgb = xgb_model.predict(X_test)
            
            metrics_xgb = {
                "accuracy": accuracy_score(y_test, y_pred_xgb),
                "precision": precision_score(y_test, y_pred_xgb, zero_division=0, average='weighted'),
                "recall": recall_score(y_test, y_pred_xgb, zero_division=0, average='weighted'),
                "f1_score": f1_score(y_test, y_pred_xgb, zero_division=0, average='weighted')
            }
            
            mlflow_logger.log_params(self.config['model']['models_to_train'][1]['params'])
            mlflow_logger.log_metrics(metrics_xgb)
            mlflow.sklearn.log_model(xgb_model, "model")
            model_results['xgboost'] = metrics_xgb
            
            logger.info(f"XGBoost Metrics: {metrics_xgb}")
            mlflow_logger.end_run()
            
            # Train LightGBM Classifier
            mlflow_logger.start_run(
                run_name="lightgbm_classifier",
                tags={"model_type": "LightGBMClassifier", "pipeline": "main"}
            )
            
            lgb_model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', lgb.LGBMClassifier(
                    n_estimators=self.config['model']['models_to_train'][2]['params']['n_estimators'],
                    max_depth=self.config['model']['models_to_train'][2]['params']['max_depth'],
                    learning_rate=self.config['model']['models_to_train'][2]['params']['learning_rate'],
                    random_state=42,
                    verbose=-1
                ))
            ])
            
            lgb_model.fit(X_train, y_train)
            y_pred_lgb = lgb_model.predict(X_test)
            
            metrics_lgb = {
                "accuracy": accuracy_score(y_test, y_pred_lgb),
                "precision": precision_score(y_test, y_pred_lgb, zero_division=0, average='weighted'),
                "recall": recall_score(y_test, y_pred_lgb, zero_division=0, average='weighted'),
                "f1_score": f1_score(y_test, y_pred_lgb, zero_division=0, average='weighted')
            }
            
            mlflow_logger.log_params(self.config['model']['models_to_train'][2]['params'])
            mlflow_logger.log_metrics(metrics_lgb)
            mlflow.sklearn.log_model(lgb_model, "model")
            model_results['lightgbm'] = metrics_lgb
            
            logger.info(f"LightGBM Metrics: {metrics_lgb}")
            mlflow_logger.end_run()
            
            # Step 5: Select and save best model
            logger.info("Step 4: Selecting best model")
            best_model_name = self._compare_models(model_results)
            
            # Get the best model
            if best_model_name == 'random_forest':
                best_model = rf_model
            elif best_model_name == 'xgboost':
                best_model = xgb_model
            else:
                best_model = lgb_model
            
            # Save best model
            output_path = self.config['paths']['models_dir']
            ModelManager.save_model(best_model, f"{output_path}/best_model.joblib")
            
            logger.info("Training pipeline completed successfully!")
            logger.info(f"All models trained: RF={model_results['random_forest']['f1_score']:.4f}, "
                       f"XGB={model_results['xgboost']['f1_score']:.4f}, "
                       f"LGB={model_results['lightgbm']['f1_score']:.4f}")
            
            return best_model, model_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _compare_models(self, results):
        """Compare models and return best one"""
        best_model = max(results, key=lambda x: results[x]['f1_score'])
        logger.info(f"Best model: {best_model} (F1: {results[best_model]['f1_score']:.4f})")
        return best_model

if __name__ == "__main__":
    # Start MLflow server (optional for local testing)
    # You can run: mlflow ui --backend-store-uri ./mlruns
    
    pipeline = TrainingPipeline()
    best_model, results = pipeline.run()
