"""Feature engineering script"""

import sys
from pathlib import Path
import yaml
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import logger
from src.data import DataLoader

class FeatureEngineering:
    """Build and engineer features"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def run(self):
        """Execute feature engineering"""
        try:
            logger.info("Starting feature engineering")
            
            # Load processed data
            df = DataLoader.load_data(self.config['data']['processed_path'])
            
            logger.info(f"Input shape: {df.shape}")
            
            # Create new features (example)
            df = self._create_features(df)
            
            # Handle categorical variables (example)
            df = self._encode_categorical(df)
            
            # Feature scaling
            df = self._scale_features(df)
            
            # Select features for modeling
            target = self.config['features']['target']
            drop_cols = self.config['features']['drop_columns']
            
            # Get feature info
            feature_info = DataLoader.get_feature_info(df)
            logger.info(f"Features created: {feature_info}")
            
            # Save featured data
            output_path = self.config['data'].get('featured_data_path', 
                                                   'data/featured_data.csv')
            DataLoader.save_data(df, output_path)
            
            # Save metrics
            metrics = {
                "total_features": feature_info['total_features'],
                "numeric_features": len(feature_info['numeric_features']),
                "categorical_features": len(feature_info['categorical_features']),
                "final_shape": list(df.shape)
            }
            
            logger.info(f"Feature engineering complete: {metrics}")
            
            metrics_dir = Path("metrics")
            metrics_dir.mkdir(exist_ok=True)
            with open(metrics_dir / "features_metrics.json", 'w') as f:
                json.dump(metrics, f)
            
            return df
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features"""
        logger.info("Creating new features")
        
        # Example: Create interaction features or polynomials
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column if it exists
        target = self.config['features']['target']
        if target in numeric_cols:
            numeric_cols.remove(target)
        
        # Add example polynomial features
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            logger.info(f"Created interaction feature: {col1}_x_{col2}")
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        logger.info("Encoding categorical variables")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column if categorical
        target = self.config['features']['target']
        if target in categorical_cols:
            categorical_cols.remove(target)
        
        # One-hot encode
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            logger.info(f"Encoded {len(categorical_cols)} categorical columns")
        
        return df
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric features"""
        logger.info("Scaling features")
        
        target = self.config['features']['target']
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col != target]
        
        if numeric_cols:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            logger.info(f"Scaled {len(numeric_cols)} numeric features")
        
        return df

if __name__ == "__main__":
    fe = FeatureEngineering()
    featured_data = fe.run()
    logger.info("Feature engineering pipeline completed successfully!")
