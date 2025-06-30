#!/usr/bin/env python3
"""
File: __init__.py
Path: NeuroCluster-Elite/src/data/__init__.py
Description: Data management package initialization

This module initializes the data management components including multi-asset
data fetching, caching, validation, and provider integrations.

Author: Your Name
Created: 2025-06-28
Version: 1.0.0
License: MIT
"""

# Import main data components
try:
    from .multi_asset_manager import (
        MultiAssetDataManager,
        DataCache,
        DataProvider,
        YahooFinanceProvider,
        CryptoProvider,
        ForexProvider
    )
    
    # Import when available
    # from .stock_data import StockDataFetcher
    # from .crypto_data import CryptoDataFetcher
    # from .forex_data import ForexDataFetcher
    # from .commodity_data import CommodityDataFetcher
    # from .data_validator import DataValidator
    
    __all__ = [
        'MultiAssetDataManager',
        'DataCache',
        'DataProvider',
        'YahooFinanceProvider',
        'CryptoProvider',
        'ForexProvider',
        # 'StockDataFetcher',
        # 'CryptoDataFetcher',
        # 'ForexDataFetcher',
        # 'CommodityDataFetcher',
        # 'DataValidator'
    ]
    
except ImportError as e:
    import warnings
    warnings.warn(f"Some data components could not be imported: {e}")
    __all__ = []

# Data module constants
SUPPORTED_PROVIDERS = ['yahoo_finance', 'alpha_vantage', 'coingecko', 'polygon']
DEFAULT_CACHE_TTL = 30  # seconds
MAX_CACHE_SIZE = 1000   # items
DEFAULT_RATE_LIMIT = 60 # requests per minute

# Supported asset types
SUPPORTED_ASSETS = [
    'STOCK', 'ETF', 'CRYPTO', 'FOREX', 'COMMODITY', 
    'INDEX', 'BOND', 'OPTION', 'FUTURE'
]

def get_data_info():
    """Get data module information"""
    return {
        'supported_providers': len(SUPPORTED_PROVIDERS),
        'supported_assets': len(SUPPORTED_ASSETS),
        'default_cache_ttl': f"{DEFAULT_CACHE_TTL}s",
        'max_cache_size': MAX_CACHE_SIZE,
        'rate_limit': f"{DEFAULT_RATE_LIMIT}/min"
    }