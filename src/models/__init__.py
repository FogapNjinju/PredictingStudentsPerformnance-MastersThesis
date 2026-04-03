"""Model training and evaluation utilities"""

import joblib
from pathlib import Path
from typing import Any, Dict, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from src.utils.logger import logger

class ModelManager:
    """Utility class for model management"""
    
    @staticmethod
    def save_model(model: Any, model_path: str) -> None:
        """
        Save model to disk
        
        Args:
            model: Trained model object
            model_path: Path to save the model
        """
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
    
    @staticmethod
    def load_model(model_path: str) -> Any:
        """
        Load model from disk
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model object
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        return model
    
    @staticmethod
    def evaluate_model(y_true, y_pred) -> Dict[str, float]:
        """
        Evaluate model predictions
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2_score": r2_score(y_true, y_pred),
        }
        
        # Add MAPE if no negative values
        if (y_true > 0).all():
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics["mape"] = mape
        
        return metrics
    
    @staticmethod
    def compare_models(results: Dict[str, Dict[str, float]], metric: str = "rmse") -> str:
        """
        Compare models and return the best one
        
        Args:
            results: Dictionary of model results
            metric: Metric to use for comparison
            
        Returns:
            Name of the best model
        """
        best_model = min(results, key=lambda x: results[x][metric])
        logger.info(f"Best model: {best_model} ({metric}: {results[best_model][metric]:.4f})")
        
        return best_model
