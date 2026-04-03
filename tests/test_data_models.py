"""Test suite for data utilities"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataLoader
from src.models import ModelManager

# Fixtures
@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'Age': np.random.randint(18, 60, 100),
        'Score1': np.random.randint(0, 100, 100),
        'Score2': np.random.randint(0, 100, 100),
        'Grade': np.random.randint(0, 100, 100)
    })

# Data tests
def test_data_loader_load(sample_data, tmp_path):
    """Test data loading"""
    # Save sample data
    data_path = tmp_path / "test_data.csv"
    sample_data.to_csv(data_path, index=False)
    
    # Load and verify
    loaded_data = DataLoader.load_data(str(data_path))
    assert loaded_data.shape == sample_data.shape
    pd.testing.assert_frame_equal(loaded_data, sample_data)

def test_data_loader_split(sample_data):
    """Test data splitting"""
    X_train, X_test, y_train, y_test = DataLoader.split_data(
        sample_data,
        target_column='Grade',
        test_size=0.2,
        random_state=42
    )
    
    assert len(X_train) + len(X_test) == len(sample_data)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

def test_feature_info(sample_data):
    """Test feature information extraction"""
    info = DataLoader.get_feature_info(sample_data)
    
    assert 'numeric_features' in info
    assert 'categorical_features' in info
    assert len(info['numeric_features']) == 4

# Model tests
def test_model_save_load(tmp_path):
    """Test model saving and loading"""
    from sklearn.tree import DecisionTreeRegressor
    
    model = DecisionTreeRegressor()
    model_path = tmp_path / "test_model.joblib"
    
    # Save model
    ModelManager.save_model(model, str(model_path))
    assert model_path.exists()
    
    # Load model
    loaded_model = ModelManager.load_model(str(model_path))
    assert isinstance(loaded_model, DecisionTreeRegressor)

def test_model_evaluation():
    """Test model evaluation metrics"""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    
    metrics = ModelManager.evaluate_model(y_true, y_pred)
    
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2_score' in metrics
    assert all(isinstance(v, (int, float)) for v in metrics.values())

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
