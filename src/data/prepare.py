"""Data preparation script"""

import sys
from pathlib import Path
import yaml
import json
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import logger
from src.data import DataLoader

class DataPreparation:
    """Prepare and clean data"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def run(self):
        """Execute data preparation"""
        try:
            logger.info("Starting data preparation")
            
            # Load raw data
            df = DataLoader.load_data(self.config['data']['raw_path'])
            
            # Log initial info
            logger.info(f"Original data shape: {df.shape}")
            logger.info(f"Missing values:\n{df.isnull().sum()}")
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Remove duplicates
            df = self._remove_duplicates(df)
            
            # Basic data cleaning
            df = self._clean_data(df)
            
            # Save processed data
            output_path = self.config['data']['processed_path']
            DataLoader.save_data(df, output_path)
            
            # Log metrics
            metrics = {
                "final_shape": list(df.shape),
                "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
                "categorical_columns": len(df.select_dtypes(include=['object']).columns)
            }
            
            logger.info(f"Data preparation complete: {metrics}")
            
            # Save metrics
            metrics_dir = Path("metrics")
            metrics_dir.mkdir(exist_ok=True)
            with open(metrics_dir / "prepare_metrics.json", 'w') as f:
                json.dump(metrics, f)
            
            return df
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        logger.info("Handling missing values")
        
        # For numeric columns: fill with mean
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mean(), inplace=True)
        
        # For categorical columns: fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        logger.info(f"Missing values handled. Remaining: {df.isnull().sum().sum()}")
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning"""
        logger.info("Cleaning data")
        
        # Remove rows where all values are NaN
        df = df.dropna(how='all')
        
        # Strip whitespace from object columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
        
        return df

if __name__ == "__main__":
    prep = DataPreparation()
    prepared_data = prep.run()
    logger.info("Data preparation pipeline completed successfully!")
