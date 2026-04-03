"""Data loading and processing utilities"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from src.utils.logger import logger

class DataLoader:
    """Utility class for loading and basic data operations"""
    
    @staticmethod
    def load_data(data_path: str, delimiter: str = None) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            data_path: Path to the CSV file
            delimiter: Delimiter used in CSV (auto-detect if None)
            
        Returns:
            Loaded data as DataFrame
        """
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        logger.info(f"Loading data from {data_path}")
        
        # Auto-detect delimiter if not specified
        if delimiter is None:
            with open(data_path, 'r') as f:
                first_line = f.readline()
                if ';' in first_line:
                    delimiter = ';'
                elif '\t' in first_line:
                    delimiter = '\t'
                else:
                    delimiter = ','
        
        df = pd.read_csv(data_path, delimiter=delimiter)
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns (delimiter: '{delimiter}')")
        
        return df
    
    @staticmethod
    def save_data(df: pd.DataFrame, output_path: str) -> None:
        """
        Save data to CSV file
        
        Args:
            df: DataFrame to save
            output_path: Path to save the CSV file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_file, index=False)
        logger.info(f"Data saved to {output_path}")
    
    @staticmethod
    def split_data(
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Data split: Train {X_train.shape[0]}, Test {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def get_feature_info(df: pd.DataFrame) -> dict:
        """
        Get information about features in the DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with feature information
        """
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        return {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "total_features": len(numeric_features) + len(categorical_features)
        }
