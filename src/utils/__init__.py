#!/usr/bin/env python3
"""
File: __init__.py
Path: NeuroCluster-Elite/src/utils/__init__.py
Description: Utilities package initialization

This module initializes utility components including configuration management,
logging, security, database operations, caching, and helper functions.

Author: Your Name
Created: 2025-06-29
Version: 1.0.0
License: MIT
"""

# Import main utility components
try:
    from .config_manager import ConfigManager, AlgorithmConfig, TradingConfig, DataConfig
    from .logger import (
        get_enhanced_logger, setup_logging, EnhancedLogger, 
        LogCategory, performance_monitor, log_audit
    )
    from .security import (
        SecurityManager, UserRole, SecurityEventType, 
        APIKey, User, RateLimiter, SecurityEvent
    )
    from .helpers import (
        format_currency, format_percentage, format_number,
        calculate_sharpe_ratio, calculate_max_drawdown, calculate_sortino_ratio,
        print_banner, print_table, validate_email
    )
    from .database import (
        DatabaseManager, QueryBuilder, TransactionManager,
        DatabaseConfig, ConnectionPool, MigrationManager
    )
    from .cache import (
        CacheManager, CacheConfig, CacheStrategy, 
        RedisCacheBackend, MemoryCacheBackend, HybridCacheBackend
    )
    
    __all__ = [
        # Configuration
        'ConfigManager',
        'AlgorithmConfig',
        'TradingConfig', 
        'DataConfig',
        
        # Logging
        'get_enhanced_logger',
        'setup_logging',
        'EnhancedLogger',
        'LogCategory',
        'performance_monitor',
        'log_audit',
        
        # Security
        'SecurityManager',
        'UserRole',
        'SecurityEventType',
        'APIKey',
        'User',
        'RateLimiter',
        'SecurityEvent',
        
        # Helpers
        'format_currency',
        'format_percentage',
        'format_number',
        'calculate_sharpe_ratio',
        'calculate_max_drawdown',
        'calculate_sortino_ratio',
        'print_banner',
        'print_table',
        'validate_email',
        
        # Database
        'DatabaseManager',
        'QueryBuilder',
        'TransactionManager',
        'DatabaseConfig',
        'ConnectionPool',
        'MigrationManager',
        
        # Cache
        'CacheManager',
        'CacheConfig',
        'CacheStrategy',
        'RedisCacheBackend',
        'MemoryCacheBackend',
        'HybridCacheBackend'
    ]
    
except ImportError as e:
    import warnings
    warnings.warn(f"Some utility components could not be imported: {e}")
    __all__ = []

# Utility module constants
SUPPORTED_LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_CONFIG_DIR = 'config'
DEFAULT_LOG_DIR = 'logs'
DEFAULT_DATA_DIR = 'data'

# Security constants
MIN_PASSWORD_LENGTH = 8
DEFAULT_JWT_EXPIRY_HOURS = 24
DEFAULT_RATE_LIMIT_PER_MINUTE = 60

# Database constants
SUPPORTED_DATABASES = ['sqlite', 'postgresql', 'mysql', 'mongodb']
DEFAULT_CONNECTION_POOL_SIZE = 10
DEFAULT_QUERY_TIMEOUT = 30

# Cache constants
SUPPORTED_CACHE_BACKENDS = ['memory', 'redis', 'memcached', 'hybrid']
DEFAULT_CACHE_TTL = 300  # 5 minutes
DEFAULT_MAX_CACHE_SIZE = 1000

def get_utils_info():
    """Get utilities module information"""
    return {
        'components': len(__all__),
        'log_levels': len(SUPPORTED_LOG_LEVELS),
        'default_dirs': [DEFAULT_CONFIG_DIR, DEFAULT_LOG_DIR, DEFAULT_DATA_DIR],
        'security_features': ['JWT', '2FA', 'Rate Limiting', 'Encryption'],
        'min_password_length': MIN_PASSWORD_LENGTH,
        'supported_databases': SUPPORTED_DATABASES,
        'supported_cache_backends': SUPPORTED_CACHE_BACKENDS,
        'cache_features': ['TTL', 'LRU', 'Compression', 'Encryption', 'Clustering']
    }

def validate_environment():
    """Validate environment setup for NeuroCluster Elite"""
    
    import os
    from pathlib import Path
    
    validation_results = {
        'directories': {},
        'permissions': {},
        'dependencies': {},
        'configuration': {}
    }
    
    # Check required directories
    required_dirs = [DEFAULT_CONFIG_DIR, DEFAULT_LOG_DIR, DEFAULT_DATA_DIR, 'cache', 'exports']
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        validation_results['directories'][dir_name] = {
            'exists': dir_path.exists(),
            'writable': dir_path.exists() and os.access(dir_path, os.W_OK),
            'readable': dir_path.exists() and os.access(dir_path, os.R_OK)
        }
        
        # Create directory if it doesn't exist
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                validation_results['directories'][dir_name]['created'] = True
            except Exception as e:
                validation_results['directories'][dir_name]['error'] = str(e)
    
    # Check Python dependencies
    required_packages = [
        'numpy', 'pandas', 'asyncio', 'aiohttp', 'redis', 
        'sqlite3', 'psycopg2', 'cryptography', 'pydantic'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            validation_results['dependencies'][package] = True
        except ImportError:
            validation_results['dependencies'][package] = False
    
    return validation_results

def initialize_infrastructure(config: dict = None):
    """Initialize NeuroCluster Elite infrastructure"""
    
    config = config or {}
    
    # Validate environment
    validation = validate_environment()
    
    # Initialize logging
    log_config = config.get('logging', {})
    setup_logging(
        level=log_config.get('level', 'INFO'),
        log_file=log_config.get('file', 'logs/neurocluster.log'),
        enable_json=log_config.get('json_format', True)
    )
    
    # Initialize database if configured
    db_config = config.get('database', {})
    if db_config.get('enabled', True):
        try:
            db_manager = DatabaseManager(db_config)
            db_manager.initialize()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Database initialization failed: {e}")
    
    # Initialize cache if configured
    cache_config = config.get('cache', {})
    if cache_config.get('enabled', True):
        try:
            cache_manager = CacheManager(cache_config)
            cache_manager.initialize()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Cache initialization failed: {e}")
    
    return validation

# Version information
__version__ = "1.0.0"
__author__ = "NeuroCluster Elite Team"
__license__ = "MIT"