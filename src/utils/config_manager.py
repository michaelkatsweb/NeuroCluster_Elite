#!/usr/bin/env python3
"""
File: config_manager.py
Path: NeuroCluster-Elite/src/utils/config_manager.py
Description: Configuration management system for NeuroCluster Elite

This module provides comprehensive configuration management for the NeuroCluster Elite
trading platform, including environment-based configuration, secrets management,
validation, and dynamic configuration updates.

Features:
- Environment-based configuration (dev/staging/prod)
- Secure secrets management with encryption
- Configuration validation and type checking
- Hot reloading of configuration changes
- Default configuration management
- Configuration versioning and backup

Author: Your Name
Created: 2025-06-28
Version: 1.0.0
License: MIT
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import secrets
from cryptography.fernet import Fernet
from pydantic import BaseSettings, validator, Field
import threading
import time
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION SCHEMAS ====================

class AlgorithmConfig(BaseSettings):
    """NeuroCluster algorithm configuration"""
    
    # Core algorithm parameters (proven values)
    similarity_threshold: float = Field(0.75, ge=0.5, le=1.0)
    learning_rate: float = Field(0.14, ge=0.01, le=0.5)
    decay_rate: float = Field(0.02, ge=0.001, le=0.1)
    max_clusters: int = Field(12, ge=5, le=20)
    
    # Advanced features
    vectorization_enabled: bool = True
    drift_detection: bool = True
    adaptive_learning: bool = True
    health_monitoring: bool = True
    
    # Performance tuning
    feature_vector_size: int = Field(12, ge=8, le=20)
    outlier_threshold: float = Field(2.5, ge=1.0, le=5.0)
    memory_limit_mb: int = Field(256, ge=64, le=2048)
    
    @validator('similarity_threshold')
    def validate_similarity_threshold(cls, v):
        if not 0.5 <= v <= 1.0:
            raise ValueError('Similarity threshold must be between 0.5 and 1.0')
        return v

class TradingConfig(BaseSettings):
    """Trading engine configuration"""
    
    # Portfolio settings
    initial_capital: float = Field(100000.0, ge=1000.0)
    max_positions: int = Field(20, ge=1, le=100)
    max_position_size: float = Field(0.10, ge=0.01, le=0.5)
    max_portfolio_risk: float = Field(0.02, ge=0.001, le=0.1)
    
    # Risk management
    stop_loss_pct: float = Field(0.05, ge=0.01, le=0.2)
    take_profit_pct: float = Field(0.15, ge=0.05, le=1.0)
    kelly_fraction_limit: float = Field(0.25, ge=0.01, le=0.5)
    
    # Trading modes
    paper_trading: bool = True
    auto_trading: bool = False
    strategy_adaptation: bool = True
    
    # Asset allocation limits
    max_stock_allocation: float = Field(0.70, ge=0.1, le=1.0)
    max_crypto_allocation: float = Field(0.30, ge=0.0, le=0.5)
    max_forex_allocation: float = Field(0.20, ge=0.0, le=0.5)
    max_commodity_allocation: float = Field(0.15, ge=0.0, le=0.3)

class DataConfig(BaseSettings):
    """Data management configuration"""
    
    # Cache settings
    cache_enabled: bool = True
    cache_ttl_seconds: int = Field(30, ge=5, le=300)
    max_cache_size_mb: int = Field(512, ge=64, le=2048)
    
    # Data sources
    yahoo_finance_enabled: bool = True
    alpha_vantage_enabled: bool = True
    coingecko_enabled: bool = True
    polygon_enabled: bool = False
    
    # Update intervals
    market_data_interval: int = Field(30, ge=5, le=300)
    regime_detection_interval: int = Field(10, ge=5, le=60)
    portfolio_update_interval: int = Field(60, ge=30, le=300)
    
    # Data validation
    price_validation_enabled: bool = True
    volume_validation_enabled: bool = True
    outlier_detection_enabled: bool = True

class SecurityConfig(BaseSettings):
    """Security configuration"""
    
    # Authentication
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_expiry_hours: int = Field(24, ge=1, le=168)
    max_login_attempts: int = Field(5, ge=1, le=10)
    
    # API security
    rate_limit_enabled: bool = True
    max_requests_per_minute: int = Field(100, ge=10, le=1000)
    api_key_rotation_days: int = Field(90, ge=30, le=365)
    
    # Encryption
    encryption_enabled: bool = True
    encryption_key: Optional[str] = None
    
    # Network security
    allowed_hosts: List[str] = Field(default_factory=lambda: ["*"])
    cors_enabled: bool = True
    https_only: bool = False

class NotificationConfig(BaseSettings):
    """Notification configuration"""
    
    # Email notifications
    email_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    
    # Discord notifications
    discord_enabled: bool = False
    discord_bot_token: Optional[str] = None
    discord_channel_id: Optional[str] = None
    
    # Telegram notifications
    telegram_enabled: bool = False
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    
    # Voice notifications
    voice_enabled: bool = True
    voice_language: str = "en-US"

@dataclass
class ApplicationConfig:
    """Complete application configuration"""
    
    # Environment
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    
    # Component configurations
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    
    # API keys (encrypted)
    api_keys: Dict[str, str] = field(default_factory=dict)
    
    # Feature flags
    features: Dict[str, bool] = field(default_factory=lambda: {
        "voice_control": True,
        "sentiment_analysis": True,
        "news_integration": True,
        "social_sentiment": True,
        "pattern_recognition": True,
        "advanced_charting": True,
        "mobile_api": True,
        "webhook_support": True
    })

# ==================== CONFIGURATION MANAGER ====================

class ConfigManager:
    """Comprehensive configuration management system"""
    
    def __init__(self, config_dir: str = "config", environment: str = None):
        self.config_dir = Path(config_dir)
        self.environment = environment or os.getenv("NEUROCLUSTER_ENV", "development")
        self.config_cache = {}
        self.encryption_manager = None
        self.file_watchers = {}
        self.config_lock = threading.RLock()
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize encryption
        self._initialize_encryption()
        
        # Load configuration
        self._load_configuration()
        
        logger.info(f"ConfigManager initialized for environment: {self.environment}")
    
    def _initialize_encryption(self):
        """Initialize encryption for sensitive configuration data"""
        
        encryption_key_file = self.config_dir / ".encryption_key"
        
        if encryption_key_file.exists():
            with open(encryption_key_file, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(encryption_key_file, 'wb') as f:
                f.write(key)
            # Secure the key file
            os.chmod(encryption_key_file, 0o600)
        
        self.encryption_manager = Fernet(key)
        logger.info("Encryption manager initialized")
    
    def _load_configuration(self):
        """Load configuration from various sources"""
        
        with self.config_lock:
            # 1. Load default configuration
            default_config = self._load_default_config()
            
            # 2. Load environment-specific configuration
            env_config = self._load_environment_config()
            
            # 3. Load user configuration
            user_config = self._load_user_config()
            
            # 4. Load environment variables
            env_vars = self._load_environment_variables()
            
            # 5. Merge configurations (priority: env_vars > user > env > default)
            merged_config = self._merge_configs([
                default_config,
                env_config,
                user_config,
                env_vars
            ])
            
            # 6. Validate configuration
            validated_config = self._validate_configuration(merged_config)
            
            # 7. Cache configuration
            self.config_cache = validated_config
            
            logger.info("Configuration loaded and validated successfully")
    
    def _load_default_config(self) -> Dict:
        """Load default configuration"""
        
        default_config_file = self.config_dir / "default.yaml"
        
        if default_config_file.exists():
            return self._load_yaml_file(default_config_file)
        else:
            # Create default configuration
            default_config = asdict(ApplicationConfig())
            self._save_yaml_file(default_config_file, default_config)
            return default_config
    
    def _load_environment_config(self) -> Dict:
        """Load environment-specific configuration"""
        
        env_config_file = self.config_dir / f"{self.environment}.yaml"
        
        if env_config_file.exists():
            return self._load_yaml_file(env_config_file)
        
        return {}
    
    def _load_user_config(self) -> Dict:
        """Load user-specific configuration"""
        
        user_config_file = self.config_dir / "user.yaml"
        
        if user_config_file.exists():
            return self._load_yaml_file(user_config_file)
        
        return {}
    
    def _load_environment_variables(self) -> Dict:
        """Load configuration from environment variables"""
        
        env_config = {}
        
        # Algorithm settings
        if os.getenv("NCS_SIMILARITY_THRESHOLD"):
            env_config.setdefault("algorithm", {})["similarity_threshold"] = float(
                os.getenv("NCS_SIMILARITY_THRESHOLD")
            )
        
        if os.getenv("NCS_LEARNING_RATE"):
            env_config.setdefault("algorithm", {})["learning_rate"] = float(
                os.getenv("NCS_LEARNING_RATE")
            )
        
        # Trading settings
        if os.getenv("INITIAL_CAPITAL"):
            env_config.setdefault("trading", {})["initial_capital"] = float(
                os.getenv("INITIAL_CAPITAL")
            )
        
        if os.getenv("PAPER_TRADING"):
            env_config.setdefault("trading", {})["paper_trading"] = (
                os.getenv("PAPER_TRADING").lower() == "true"
            )
        
        # API keys
        api_keys = {}
        for key, env_var in {
            "alpha_vantage": "ALPHA_VANTAGE_API_KEY",
            "polygon": "POLYGON_API_KEY",
            "binance": "BINANCE_API_KEY",
            "coinbase": "COINBASE_API_KEY",
            "twitter": "TWITTER_API_KEY",
            "news_api": "NEWS_API_KEY"
        }.items():
            if os.getenv(env_var):
                api_keys[key] = os.getenv(env_var)
        
        if api_keys:
            env_config["api_keys"] = api_keys
        
        return env_config
    
    def _load_yaml_file(self, file_path: Path) -> Dict:
        """Load YAML configuration file"""
        
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading YAML file {file_path}: {e}")
            return {}
    
    def _save_yaml_file(self, file_path: Path, data: Dict):
        """Save configuration to YAML file"""
        
        try:
            with open(file_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving YAML file {file_path}: {e}")
    
    def _merge_configs(self, configs: List[Dict]) -> Dict:
        """Merge multiple configuration dictionaries"""
        
        merged = {}
        
        for config in configs:
            if config:
                merged = self._deep_merge(merged, config)
        
        return merged
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_configuration(self, config: Dict) -> Dict:
        """Validate configuration using Pydantic models"""
        
        try:
            # Validate algorithm configuration
            if "algorithm" in config:
                algorithm_config = AlgorithmConfig(**config["algorithm"])
                config["algorithm"] = algorithm_config.dict()
            
            # Validate trading configuration
            if "trading" in config:
                trading_config = TradingConfig(**config["trading"])
                config["trading"] = trading_config.dict()
            
            # Validate data configuration
            if "data" in config:
                data_config = DataConfig(**config["data"])
                config["data"] = data_config.dict()
            
            # Validate security configuration
            if "security" in config:
                security_config = SecurityConfig(**config["security"])
                config["security"] = security_config.dict()
            
            # Validate notification configuration
            if "notifications" in config:
                notification_config = NotificationConfig(**config["notifications"])
                config["notifications"] = notification_config.dict()
            
            return config
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            raise ValueError(f"Invalid configuration: {e}")
    
    def get_config(self, section: str = None) -> Union[Dict, Any]:
        """Get configuration or configuration section"""
        
        with self.config_lock:
            if section:
                return self.config_cache.get(section, {})
            return self.config_cache.copy()
    
    def set_config(self, key: str, value: Any, section: str = None):
        """Set configuration value"""
        
        with self.config_lock:
            if section:
                if section not in self.config_cache:
                    self.config_cache[section] = {}
                self.config_cache[section][key] = value
            else:
                self.config_cache[key] = value
            
            # Save to user configuration file
            self._save_user_config()
            
            logger.info(f"Configuration updated: {section}.{key} = {value}" if section else f"{key} = {value}")
    
    def _save_user_config(self):
        """Save current configuration to user config file"""
        
        user_config_file = self.config_dir / "user.yaml"
        self._save_yaml_file(user_config_file, self.config_cache)
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt sensitive configuration value"""
        
        if self.encryption_manager:
            return self.encryption_manager.encrypt(value.encode()).decode()
        return value
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt sensitive configuration value"""
        
        if self.encryption_manager:
            try:
                return self.encryption_manager.decrypt(encrypted_value.encode()).decode()
            except Exception as e:
                logger.error(f"Failed to decrypt value: {e}")
                return encrypted_value
        return encrypted_value
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get decrypted API key for service"""
        
        api_keys = self.get_config("api_keys")
        encrypted_key = api_keys.get(service)
        
        if encrypted_key:
            return self.decrypt_value(encrypted_key)
        
        return None
    
    def set_api_key(self, service: str, api_key: str):
        """Set encrypted API key for service"""
        
        encrypted_key = self.encrypt_value(api_key)
        
        with self.config_lock:
            if "api_keys" not in self.config_cache:
                self.config_cache["api_keys"] = {}
            self.config_cache["api_keys"][service] = encrypted_key
        
        self._save_user_config()
        logger.info(f"API key set for service: {service}")
    
    def reload_configuration(self):
        """Reload configuration from files"""
        
        logger.info("Reloading configuration...")
        self._load_configuration()
    
    def backup_configuration(self) -> str:
        """Create backup of current configuration"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.config_dir / f"backup_{timestamp}.yaml"
        
        self._save_yaml_file(backup_file, self.config_cache)
        
        logger.info(f"Configuration backed up to: {backup_file}")
        return str(backup_file)
    
    def restore_configuration(self, backup_file: str):
        """Restore configuration from backup"""
        
        backup_path = Path(backup_file)
        
        if backup_path.exists():
            restored_config = self._load_yaml_file(backup_path)
            
            with self.config_lock:
                self.config_cache = restored_config
            
            self._save_user_config()
            logger.info(f"Configuration restored from: {backup_file}")
        else:
            raise FileNotFoundError(f"Backup file not found: {backup_file}")
    
    def get_config_hash(self) -> str:
        """Get hash of current configuration for change detection"""
        
        config_str = json.dumps(self.config_cache, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def export_configuration(self, file_path: str, include_secrets: bool = False):
        """Export configuration to file"""
        
        export_config = self.config_cache.copy()
        
        if not include_secrets:
            # Remove sensitive data
            if "api_keys" in export_config:
                export_config["api_keys"] = {k: "***REDACTED***" for k in export_config["api_keys"]}
            if "security" in export_config:
                for key in ["secret_key", "encryption_key"]:
                    if key in export_config["security"]:
                        export_config["security"][key] = "***REDACTED***"
        
        export_path = Path(file_path)
        
        if export_path.suffix.lower() == '.json':
            with open(export_path, 'w') as f:
                json.dump(export_config, f, indent=2)
        else:
            self._save_yaml_file(export_path, export_config)
        
        logger.info(f"Configuration exported to: {file_path}")
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate all configured API keys"""
        
        api_keys = self.get_config("api_keys")
        validation_results = {}
        
        for service, encrypted_key in api_keys.items():
            try:
                decrypted_key = self.decrypt_value(encrypted_key)
                # Basic validation (non-empty, reasonable length)
                is_valid = bool(decrypted_key and len(decrypted_key) > 10)
                validation_results[service] = is_valid
                
                if not is_valid:
                    logger.warning(f"Invalid API key for service: {service}")
                    
            except Exception as e:
                logger.error(f"Error validating API key for {service}: {e}")
                validation_results[service] = False
        
        return validation_results
    
    def __str__(self) -> str:
        """String representation of configuration manager"""
        
        return f"ConfigManager(environment={self.environment}, config_dir={self.config_dir})"

# ==================== CONFIGURATION UTILITIES ====================

def create_sample_config() -> Dict:
    """Create sample configuration for new installations"""
    
    sample_config = {
        "environment": "development",
        "debug": True,
        "log_level": "INFO",
        
        "algorithm": {
            "similarity_threshold": 0.75,
            "learning_rate": 0.14,
            "decay_rate": 0.02,
            "max_clusters": 12,
            "vectorization_enabled": True
        },
        
        "trading": {
            "initial_capital": 100000.0,
            "max_positions": 20,
            "max_position_size": 0.10,
            "paper_trading": True
        },
        
        "data": {
            "cache_enabled": True,
            "cache_ttl_seconds": 30,
            "market_data_interval": 30
        },
        
        "security": {
            "rate_limit_enabled": True,
            "max_requests_per_minute": 100
        },
        
        "notifications": {
            "email_enabled": False,
            "discord_enabled": False,
            "voice_enabled": True
        },
        
        "features": {
            "voice_control": True,
            "sentiment_analysis": True,
            "news_integration": True
        }
    }
    
    return sample_config

def validate_environment() -> bool:
    """Validate environment setup"""
    
    required_dirs = ["config", "logs", "data", "cache"]
    
    for directory in required_dirs:
        Path(directory).mkdir(exist_ok=True)
    
    return True

# ==================== MAIN FUNCTION ====================

if __name__ == "__main__":
    # Test configuration manager
    print("🧪 Testing NeuroCluster Elite Configuration Manager")
    
    # Initialize configuration manager
    config_manager = ConfigManager()
    
    # Test configuration access
    algorithm_config = config_manager.get_config("algorithm")
    print(f"Algorithm config: {algorithm_config}")
    
    # Test API key management
    config_manager.set_api_key("test_service", "test_api_key_12345")
    retrieved_key = config_manager.get_api_key("test_service")
    print(f"API key test: {'✅ PASS' if retrieved_key == 'test_api_key_12345' else '❌ FAIL'}")
    
    # Test configuration validation
    validation_results = config_manager.validate_api_keys()
    print(f"API key validation: {validation_results}")
    
    # Test backup/restore
    backup_file = config_manager.backup_configuration()
    print(f"Configuration backed up to: {backup_file}")
    
    print("✅ Configuration manager test completed!")