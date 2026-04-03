"""Configuration management for the ML pipeline"""

import yaml
from pathlib import Path
from typing import Any, Dict

class ConfigManager:
    """Manages configuration loading and access"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize ConfigManager
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default=None) -> Any:
        """Get configuration value by dot-separated key"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default
    
    def __getitem__(self, key: str):
        """Get configuration value using dictionary syntax"""
        return self.get(key)


# Global config instance
config = ConfigManager()
