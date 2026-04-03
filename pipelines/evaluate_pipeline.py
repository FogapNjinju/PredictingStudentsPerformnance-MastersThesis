"""Model evaluation pipeline"""

import sys
from pathlib import Path
import yaml
import json
import pandas as pd
import mlflow

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import logger
from src.utils.mlflow_utils import mlflow_logger
from src.data import DataLoader
from src.models import ModelManager

class EvaluationPipeline:
    """Evaluate trained models"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def run(self):
        """Run evaluation pipeline"""
        try:
            logger.info("Starting model evaluation")
            
            # Load best model
            model_path = f"{self.config['paths']['models_dir']}/best_model.joblib"
            model = ModelManager.load_model(model_path)
            
            # Load featured data
            df = DataLoader.load_data(self.config['data'].get('featured_data_path', 
                                                              'data/featured_data.csv'))
            
            # Prepare data
            target = self.config['features']['target']
            if target not in df.columns:
                target = df.columns[-1]
            
            X = df.drop(columns=[target])
            y = df[target]
            
            # Get numeric columns
            numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
            X_numeric = X[numeric_cols]
            
            # Make predictions
            y_pred = model.predict(X_numeric)
            
            # Evaluate
            metrics = ModelManager.evaluate_model(y, y_pred)
            
            logger.info(f"Evaluation metrics: {metrics}")
            
            # Save metrics
            metrics_dir = Path("metrics")
            metrics_dir.mkdir(exist_ok=True)
            with open(metrics_dir / "eval_metrics.json", 'w') as f:
                json.dump(metrics, f)
            
            # Log to MLflow
            mlflow_logger.start_run(run_name="model_evaluation")
            mlflow_logger.log_metrics(metrics)
            mlflow_logger.end_run()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise

if __name__ == "__main__":
    eval_pipeline = EvaluationPipeline()
    metrics = eval_pipeline.run()
    logger.info("Evaluation pipeline completed successfully!")
