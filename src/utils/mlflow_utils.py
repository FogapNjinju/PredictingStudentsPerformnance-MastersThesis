"""Experiment tracking and MLflow integration"""

import mlflow
from typing import Dict, Any, Optional
import json
from pathlib import Path

class MLflowLogger:
    """Wrapper for MLflow tracking"""
    
    def __init__(self, experiment_name: str = "student_performance"):
        """
        Initialize MLflow logger
        
        Args:
            experiment_name: Name of the MLflow experiment
        """
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
    
    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        """Start a new MLflow run"""
        mlflow.start_run(run_name=run_name)
        if tags:
            mlflow.set_tags(tags)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_artifact(self, artifact_path: str):
        """Log an artifact (file or directory)"""
        mlflow.log_artifact(artifact_path)
    
    def log_model(self, model, artifact_path: str, **kwargs):
        """Log a model"""
        mlflow.sklearn.log_model(model, artifact_path, **kwargs)
    
    def end_run(self):
        """End current run"""
        mlflow.end_run()
    
    @staticmethod
    def get_best_run(metric_name: str = "rmse", ascending: bool = True):
        """Get best run for a metric"""
        experiment = mlflow.get_experiment_by_name("student_performance")
        if not experiment:
            return None
        
        exp_id = experiment.experiment_id
        runs = mlflow.search_runs(experiment_ids=[exp_id])
        
        if runs.empty:
            return None
        
        if ascending:
            best_run = runs.loc[runs[f"metrics.{metric_name}"].idxmin()]
        else:
            best_run = runs.loc[runs[f"metrics.{metric_name}"].idxmax()]
        
        return best_run


# Global MLflow logger
mlflow_logger = MLflowLogger()
