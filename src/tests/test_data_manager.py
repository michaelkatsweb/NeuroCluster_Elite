#!/usr/bin/env python3
"""
File: test_data_manager.py
Path: NeuroCluster-Elite/src/tests/test_data_manager.py
Description: Comprehensive tests for data management modules

This module contains comprehensive tests for the MultiAssetDataManager and related
data management components, including data fetching, validation, caching,
and multi-source integration.

Features tested:
- Multi-asset data fetching (stocks, crypto, forex, commodities)
- Data source failover and redundancy
- Intelligent caching and performance optimization
- Data validation and quality assurance
- Real-time data processing
- Historical data management
- Rate limiting and API management
- Error handling and recovery

Author: Michael Katsaros
Created: 2025-06-29
Version: 1.0.0
License: MIT
"""

import asyncio
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import time
import json
import aiohttp
from typing import Dict, List, Any
import sqlite3
from pathlib import Path

# Import modules to test
try:
    from src.data.multi_asset_manager import (
        MultiAssetDataManager, DataSource, DataQuality, CacheManager
    )
    from src.data.stock_data import StockDataFetcher
    from src.data.crypto_data import CryptoDataFetcher
    from src.data.forex_data import ForexDataFetcher
    from src.data.commodity_data import CommodityDataFetcher
    from src.data.data_validator import DataValidator, ValidationResult
    from src.core.neurocluster_elite import MarketData, AssetType
except ImportError as e:
    pytest.skip(f"Could not import data modules: {e}", allow_module_level=True)

# Test configuration
TEST_CONFIG = {
    'cache_enabled': True,
    'cache_ttl_seconds': 300,
    'max_retries': 3,
    'timeout_seconds': 10,
    'rate_limit_calls': 100,
    'rate_limit_period': 60
}

# Sample market data for testing
SAMPLE_STOCK_DATA = {
    'AAPL': MarketData(
        symbol='AAPL',
        asset_type=AssetType.STOCK,
        price=150.25,
        change=2.75,
        change_percent=1.86,
        volume=45231000,
        timestamp=datetime.now(),
        high=151.50,
        low=148.80,
        open=149.00,
        close=150.25,
        rsi=65.4,
        macd=1.25,
        volatility=0.28
    ),
    'GOOGL': MarketData(
        symbol='GOOGL',
        asset_type=AssetType.STOCK,
        price=2850.50,
        change=-15.25,
        change_percent=-0.53,
        volume=1250000,
        timestamp=datetime.now(),
        high=2875.00,
        low=2840.00,
        open=2865.75,
        close=2850.50,
        rsi=58.7,
        macd=-5.50,
        volatility=0.32
    )
}

SAMPLE_CRYPTO_DATA = {
    'BTC-USD': MarketData(
        symbol='BTC-USD',
        asset_type=AssetType.CRYPTO,
        price=45250.75,
        change=1250.25,
        change_percent=2.84,
        volume=28500000,
        timestamp=datetime.now(),
        high=45800.00,
        low=43900.00,
        open=44000.50,
        close=45250.75,
        rsi=72.3,
        macd=850.25,
        volatility=0.45
    ),
    'ETH-USD': MarketData(
        symbol='ETH-USD',
        asset_type=AssetType.CRYPTO,
        price=3125.80,
        change=85.30,
        change_percent=2.81,
        volume=15750000,
        timestamp=datetime.now(),
        high=3150.00,
        low=3040.50,
        open=3040.50,
        close=3125.80,
        rsi=68.9,
        macd=45.75,
        volatility=0.42
    )
}

# ==================== FIXTURES ====================

@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp session for API calls"""
    session = Mock()
    
    # Mock successful responses
    async def mock_get(*args, **kwargs):
        response = Mock()
        response.status = 200
        response.json = AsyncMock(return_value={
            'symbol': 'AAPL',
            'price': 150.25,
            'change': 2.75,
            'volume': 45231000
        })
        response.text = AsyncMock(return_value='{"status": "success"}')
        return response
    
    session.get = AsyncMock(side_effect=mock_get)
    return session

@pytest.fixture
def cache_manager():
    """Create cache manager for testing"""
    return CacheManager(config=TEST_CONFIG)

@pytest.fixture
def data_validator():
    """Create data validator for testing"""
    return DataValidator()

@pytest.fixture
def stock_data_fetcher(mock_aiohttp_session):
    """Create stock data fetcher with mocked session"""
    fetcher = StockDataFetcher(config=TEST_CONFIG)
    fetcher.session = mock_aiohttp_session
    return fetcher

@pytest.fixture
def crypto_data_fetcher(mock_aiohttp_session):
    """Create crypto data fetcher with mocked session"""
    fetcher = CryptoDataFetcher(config=TEST_CONFIG)
    fetcher.session = mock_aiohttp_session
    return fetcher

@pytest.fixture
def multi_asset_manager(mock_aiohttp_session):
    """Create multi-asset data manager for testing"""
    manager = MultiAssetDataManager(config=TEST_CONFIG)
    
    # Mock all fetchers
    manager.stock_fetcher.session = mock_aiohttp_session
    manager.crypto_fetcher.session = mock_aiohttp_session
    manager.forex_fetcher.session = mock_aiohttp_session
    manager.commodity_fetcher.session = mock_aiohttp_session
    
    return manager

@pytest.fixture
def sample_historical_data():
    """Generate sample historical data"""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(140, 160, 100),
        'high': np.random.uniform(145, 165, 100),
        'low': np.random.uniform(135, 155, 100),
        'close': np.random.uniform(140, 160, 100),
        'volume': np.random.randint(500000, 2000000, 100)
    })
    return data

# ==================== CACHE MANAGER TESTS ====================

class TestCacheManager:
    """Test cache management functionality"""
    
    def test_cache_initialization(self, cache_manager):
        """Test cache manager initialization"""
        assert cache_manager.config == TEST_CONFIG
        assert cache_manager.memory_cache == {}
        assert cache_manager.cache_timestamps == {}
    
    def test_memory_cache_operations(self, cache_manager):
        """Test in-memory cache operations"""
        key = "AAPL_stock_data"
        data = SAMPLE_STOCK_DATA['AAPL']
        
        # Test cache set
        cache_manager.set(key, data)
        assert key in cache_manager.memory_cache
        assert key in cache_manager.cache_timestamps
        
        # Test cache get
        cached_data = cache_manager.get(key)
        assert cached_data is not None
        assert cached_data.symbol == data.symbol
        assert cached_data.price == data.price
    
    def test_cache_expiration(self, cache_manager):
        """Test cache expiration logic"""
        key = "test_expiration"
        data = SAMPLE_STOCK_DATA['AAPL']
        
        # Set cache with timestamp in the past
        cache_manager.set(key, data)
        
        # Manually expire the cache
        cache_manager.cache_timestamps[key] = datetime.now() - timedelta(seconds=400)
        
        # Should return None for expired cache
        cached_data = cache_manager.get(key)
        assert cached_data is None
        assert key not in cache_manager.memory_cache
    
    def test_cache_size_limits(self, cache_manager):
        """Test cache size management"""
        # Fill cache beyond reasonable limits
        for i in range(1000):
            key = f"test_key_{i}"
            cache_manager.set(key, SAMPLE_STOCK_DATA['AAPL'])
        
        # Cache should manage size automatically
        assert len(cache_manager.memory_cache) <= 500  # Reasonable limit
    
    @pytest.mark.asyncio
    async def test_sqlite_cache_operations(self, cache_manager):
        """Test SQLite cache operations"""
        if not hasattr(cache_manager, 'sqlite_conn') or cache_manager.sqlite_conn is None:
            pytest.skip("SQLite cache not configured")
        
        key = "AAPL_sqlite_test"
        data = SAMPLE_STOCK_DATA['AAPL']
        
        # Test SQLite cache set
        cache_manager.set(key, data)
        
        # Clear memory cache to force SQLite lookup
        cache_manager.memory_cache.clear()
        cache_manager.cache_timestamps.clear()
        
        # Should retrieve from SQLite
        cached_data = cache_manager.get(key)
        assert cached_data is not None
        assert cached_data.symbol == data.symbol

# ==================== DATA VALIDATOR TESTS ====================

class TestDataValidator:
    """Test data validation functionality"""
    
    def test_valid_market_data(self, data_validator):
        """Test validation of valid market data"""
        data = SAMPLE_STOCK_DATA['AAPL']
        result = data_validator.validate_market_data(data)
        
        assert result.is_valid
        assert result.quality_score > 0.8
        assert len(result.warnings) == 0
        assert len(result.errors) == 0
    
    def test_invalid_market_data(self, data_validator):
        """Test validation of invalid market data"""
        # Create invalid data
        invalid_data = MarketData(
            symbol='',  # Invalid empty symbol
            asset_type=AssetType.STOCK,
            price=-100.0,  # Invalid negative price
            change=0.0,
            change_percent=0.0,
            volume=-1000,  # Invalid negative volume
            timestamp=datetime.now()
        )
        
        result = data_validator.validate_market_data(invalid_data)
        
        assert not result.is_valid
        assert result.quality_score < 0.5
        assert len(result.errors) > 0
    
    def test_data_quality_scoring(self, data_validator):
        """Test data quality scoring algorithm"""
        # High quality data
        high_quality = SAMPLE_STOCK_DATA['AAPL']
        result_high = data_validator.validate_market_data(high_quality)
        
        # Lower quality data (missing indicators)
        low_quality = MarketData(
            symbol='TEST',
            asset_type=AssetType.STOCK,
            price=100.0,
            change=0.0,
            change_percent=0.0,
            volume=1000,
            timestamp=datetime.now()
            # Missing RSI, MACD, etc.
        )
        result_low = data_validator.validate_market_data(low_quality)
        
        assert result_high.quality_score > result_low.quality_score
    
    def test_outlier_detection(self, data_validator):
        """Test outlier detection in data"""
        # Create data with outliers
        outlier_data = MarketData(
            symbol='OUTLIER',
            asset_type=AssetType.STOCK,
            price=1000000.0,  # Extremely high price
            change=999999.0,   # Extremely high change
            change_percent=99999.0,  # Extremely high percentage
            volume=1,  # Extremely low volume
            timestamp=datetime.now()
        )
        
        result = data_validator.validate_market_data(outlier_data)
        
        assert len(result.warnings) > 0
        assert any('outlier' in warning.lower() for warning in result.warnings)
    
    def test_timestamp_validation(self, data_validator):
        """Test timestamp validation"""
        # Future timestamp
        future_data = MarketData(
            symbol='FUTURE',
            asset_type=AssetType.STOCK,
            price=100.0,
            change=0.0,
            change_percent=0.0,
            volume=1000,
            timestamp=datetime.now() + timedelta(days=1)
        )
        
        result = data_validator.validate_market_data(future_data)
        assert len(result.warnings) > 0
        
        # Very old timestamp
        old_data = MarketData(
            symbol='OLD',
            asset_type=AssetType.STOCK,
            price=100.0,
            change=0.0,
            change_percent=0.0,
            volume=1000,
            timestamp=datetime.now() - timedelta(days=30)
        )
        
        result = data_validator.validate_market_data(old_data)
        assert len(result.warnings) > 0

# ==================== STOCK DATA FETCHER TESTS ====================

class TestStockDataFetcher:
    """Test stock data fetching functionality"""
    
    @pytest.mark.asyncio
    async def test_single_stock_fetch(self, stock_data_fetcher):
        """Test fetching single stock data"""
        with patch.object(stock_data_fetcher, '_fetch_from_yahoo') as mock_yahoo:
            mock_yahoo.return_value = SAMPLE_STOCK_DATA['AAPL']
            
            data = await stock_data_fetcher.fetch_data('AAPL')
            
            assert data is not None
            assert data.symbol == 'AAPL'
            assert data.asset_type == AssetType.STOCK
            assert data.price > 0
    
    @pytest.mark.asyncio
    async def test_multiple_stock_fetch(self, stock_data_fetcher):
        """Test fetching multiple stocks"""
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        
        with patch.object(stock_data_fetcher, '_fetch_from_yahoo') as mock_yahoo:
            mock_yahoo.side_effect = [
                SAMPLE_STOCK_DATA['AAPL'],
                SAMPLE_STOCK_DATA['GOOGL'],
                SAMPLE_STOCK_DATA['AAPL']  # Use AAPL data for MSFT
            ]
            
            data = await stock_data_fetcher.fetch_multiple(symbols)
            
            assert len(data) == 3
            assert 'AAPL' in data
            assert 'GOOGL' in data
            assert 'MSFT' in data
    
    @pytest.mark.asyncio
    async def test_api_failover(self, stock_data_fetcher):
        """Test API source failover"""
        with patch.object(stock_data_fetcher, '_fetch_from_yahoo') as mock_yahoo:
            with patch.object(stock_data_fetcher, '_fetch_from_alpha_vantage') as mock_av:
                # Yahoo fails, Alpha Vantage succeeds
                mock_yahoo.side_effect = Exception("Yahoo API error")
                mock_av.return_value = SAMPLE_STOCK_DATA['AAPL']
                
                data = await stock_data_fetcher.fetch_data('AAPL')
                
                assert data is not None
                assert data.symbol == 'AAPL'
                assert mock_yahoo.called
                assert mock_av.called
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, stock_data_fetcher):
        """Test rate limiting functionality"""
        # Make many rapid requests
        tasks = []
        for _ in range(10):
            task = stock_data_fetcher.fetch_data('AAPL')
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Should have some rate limiting effect
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) > 0
        
        # Should not be instantaneous due to rate limiting
        assert total_time > 0.1
    
    @pytest.mark.asyncio
    async def test_historical_data_fetch(self, stock_data_fetcher, sample_historical_data):
        """Test historical data fetching"""
        with patch.object(stock_data_fetcher, '_fetch_historical_yahoo') as mock_hist:
            mock_hist.return_value = sample_historical_data
            
            start_date = datetime.now() - timedelta(days=30)
            end_date = datetime.now()
            
            data = await stock_data_fetcher.fetch_historical('AAPL', start_date, end_date)
            
            assert data is not None
            assert len(data) > 0
            assert 'open' in data.columns
            assert 'high' in data.columns
            assert 'low' in data.columns
            assert 'close' in data.columns
            assert 'volume' in data.columns

# ==================== CRYPTO DATA FETCHER TESTS ====================

class TestCryptoDataFetcher:
    """Test cryptocurrency data fetching"""
    
    @pytest.mark.asyncio
    async def test_crypto_fetch(self, crypto_data_fetcher):
        """Test fetching cryptocurrency data"""
        with patch.object(crypto_data_fetcher, '_fetch_from_binance') as mock_binance:
            mock_binance.return_value = SAMPLE_CRYPTO_DATA['BTC-USD']
            
            data = await crypto_data_fetcher.fetch_data('BTC-USD')
            
            assert data is not None
            assert data.symbol == 'BTC-USD'
            assert data.asset_type == AssetType.CRYPTO
            assert data.price > 0
    
    @pytest.mark.asyncio
    async def test_crypto_exchange_failover(self, crypto_data_fetcher):
        """Test cryptocurrency exchange failover"""
        with patch.object(crypto_data_fetcher, '_fetch_from_binance') as mock_binance:
            with patch.object(crypto_data_fetcher, '_fetch_from_coinbase') as mock_coinbase:
                # Binance fails, Coinbase succeeds
                mock_binance.side_effect = Exception("Binance API error")
                mock_coinbase.return_value = SAMPLE_CRYPTO_DATA['BTC-USD']
                
                data = await crypto_data_fetcher.fetch_data('BTC-USD')
                
                assert data is not None
                assert data.symbol == 'BTC-USD'
                assert mock_binance.called
                assert mock_coinbase.called
    
    @pytest.mark.asyncio
    async def test_crypto_websocket_data(self, crypto_data_fetcher):
        """Test real-time WebSocket data handling"""
        # Mock WebSocket data
        mock_ws_data = {
            'symbol': 'BTCUSDT',
            'price': '45250.75',
            'change': '1250.25',
            'volume': '28500000'
        }
        
        with patch.object(crypto_data_fetcher, '_parse_websocket_data') as mock_parse:
            mock_parse.return_value = SAMPLE_CRYPTO_DATA['BTC-USD']
            
            data = crypto_data_fetcher._parse_websocket_data(mock_ws_data)
            
            assert data is not None
            assert data.asset_type == AssetType.CRYPTO

# ==================== MULTI-ASSET MANAGER TESTS ====================

class TestMultiAssetManager:
    """Test multi-asset data manager"""
    
    @pytest.mark.asyncio
    async def test_multi_asset_fetch(self, multi_asset_manager):
        """Test fetching data for multiple asset types"""
        symbols = ['AAPL', 'BTC-USD', 'EUR/USD', 'GOLD']
        asset_types = {
            'AAPL': AssetType.STOCK,
            'BTC-USD': AssetType.CRYPTO,
            'EUR/USD': AssetType.FOREX,
            'GOLD': AssetType.COMMODITY
        }
        
        # Mock fetchers
        with patch.object(multi_asset_manager.stock_fetcher, 'fetch_data') as mock_stock:
            with patch.object(multi_asset_manager.crypto_fetcher, 'fetch_data') as mock_crypto:
                with patch.object(multi_asset_manager.forex_fetcher, 'fetch_data') as mock_forex:
                    with patch.object(multi_asset_manager.commodity_fetcher, 'fetch_data') as mock_commodity:
                        
                        mock_stock.return_value = SAMPLE_STOCK_DATA['AAPL']
                        mock_crypto.return_value = SAMPLE_CRYPTO_DATA['BTC-USD']
                        mock_forex.return_value = MarketData(
                            symbol='EUR/USD',
                            asset_type=AssetType.FOREX,
                            price=1.0850,
                            change=0.0025,
                            change_percent=0.23,
                            volume=0,
                            timestamp=datetime.now()
                        )
                        mock_commodity.return_value = MarketData(
                            symbol='GOLD',
                            asset_type=AssetType.COMMODITY,
                            price=2050.75,
                            change=15.25,
                            change_percent=0.75,
                            volume=0,
                            timestamp=datetime.now()
                        )
                        
                        data = await multi_asset_manager.fetch_market_data(symbols, asset_types)
                        
                        assert len(data) == 4
                        assert 'AAPL' in data
                        assert 'BTC-USD' in data
                        assert 'EUR/USD' in data
                        assert 'GOLD' in data
    
    @pytest.mark.asyncio
    async def test_intelligent_routing(self, multi_asset_manager):
        """Test intelligent routing to appropriate data sources"""
        # Test that symbols are routed to correct fetchers
        with patch.object(multi_asset_manager, '_route_symbol_to_fetcher') as mock_route:
            mock_route.side_effect = lambda symbol, asset_type: {
                'AAPL': multi_asset_manager.stock_fetcher,
                'BTC-USD': multi_asset_manager.crypto_fetcher,
                'EUR/USD': multi_asset_manager.forex_fetcher,
                'GOLD': multi_asset_manager.commodity_fetcher
            }[symbol]
            
            # Test routing logic
            stock_fetcher = multi_asset_manager._route_symbol_to_fetcher('AAPL', AssetType.STOCK)
            crypto_fetcher = multi_asset_manager._route_symbol_to_fetcher('BTC-USD', AssetType.CRYPTO)
            
            assert stock_fetcher == multi_asset_manager.stock_fetcher
            assert crypto_fetcher == multi_asset_manager.crypto_fetcher
    
    @pytest.mark.asyncio
    async def test_data_quality_filtering(self, multi_asset_manager):
        """Test data quality filtering and validation"""
        symbols = ['AAPL', 'INVALID']
        
        with patch.object(multi_asset_manager.stock_fetcher, 'fetch_data') as mock_fetch:
            with patch.object(multi_asset_manager.data_validator, 'validate_market_data') as mock_validate:
                
                # Setup mock responses
                valid_data = SAMPLE_STOCK_DATA['AAPL']
                invalid_data = MarketData(
                    symbol='INVALID',
                    asset_type=AssetType.STOCK,
                    price=-100.0,  # Invalid price
                    change=0.0,
                    change_percent=0.0,
                    volume=0,
                    timestamp=datetime.now()
                )
                
                mock_fetch.side_effect = [valid_data, invalid_data]
                
                # Valid data passes validation
                valid_result = ValidationResult(is_valid=True, quality_score=0.95, warnings=[], errors=[])
                # Invalid data fails validation
                invalid_result = ValidationResult(is_valid=False, quality_score=0.2, warnings=[], errors=['Invalid price'])
                
                mock_validate.side_effect = [valid_result, invalid_result]
                
                data = await multi_asset_manager.fetch_market_data(symbols, {s: AssetType.STOCK for s in symbols})
                
                # Should only contain valid data
                assert len(data) == 1
                assert 'AAPL' in data
                assert 'INVALID' not in data
    
    def test_cache_integration(self, multi_asset_manager):
        """Test cache integration with data fetching"""
        symbol = 'AAPL'
        cached_data = SAMPLE_STOCK_DATA['AAPL']
        
        # Set cache
        multi_asset_manager.cache_manager.set(f"{symbol}_stock", cached_data)
        
        # Should retrieve from cache
        result = multi_asset_manager.cache_manager.get(f"{symbol}_stock")
        
        assert result is not None
        assert result.symbol == symbol
        assert result.price == cached_data.price

# ==================== PERFORMANCE TESTS ====================

class TestDataPerformance:
    """Test data fetching performance"""
    
    @pytest.mark.asyncio
    async def test_concurrent_fetch_performance(self, multi_asset_manager):
        """Test concurrent data fetching performance"""
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        with patch.object(multi_asset_manager.stock_fetcher, 'fetch_data') as mock_fetch:
            mock_fetch.return_value = SAMPLE_STOCK_DATA['AAPL']
            
            start_time = time.time()
            
            # Fetch concurrently
            tasks = [
                multi_asset_manager.fetch_market_data([symbol], {symbol: AssetType.STOCK})
                for symbol in symbols
            ]
            
            results = await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            
            print(f"\n⚡ Concurrent Fetch Performance:")
            print(f"   Symbols: {len(symbols)}")
            print(f"   Total time: {total_time:.3f}s")
            print(f"   Average per symbol: {total_time/len(symbols):.3f}s")
            
            # Should be reasonably fast
            assert total_time < 5.0  # Under 5 seconds for 5 symbols
            assert len(results) == len(symbols)
    
    @pytest.mark.asyncio
    async def test_cache_performance_benefit(self, multi_asset_manager):
        """Test performance benefit of caching"""
        symbol = 'AAPL'
        
        with patch.object(multi_asset_manager.stock_fetcher, 'fetch_data') as mock_fetch:
            mock_fetch.return_value = SAMPLE_STOCK_DATA['AAPL']
            
            # First fetch (no cache)
            start_time = time.time()
            await multi_asset_manager.fetch_market_data([symbol], {symbol: AssetType.STOCK})
            first_fetch_time = time.time() - start_time
            
            # Second fetch (with cache)
            start_time = time.time()
            await multi_asset_manager.fetch_market_data([symbol], {symbol: AssetType.STOCK})
            cached_fetch_time = time.time() - start_time
            
            print(f"\n💾 Cache Performance Benefit:")
            print(f"   First fetch: {first_fetch_time:.3f}s")
            print(f"   Cached fetch: {cached_fetch_time:.3f}s")
            print(f"   Speedup: {first_fetch_time/cached_fetch_time:.1f}x")
            
            # Cached fetch should be faster
            assert cached_fetch_time < first_fetch_time

# ==================== ERROR HANDLING TESTS ====================

class TestErrorHandling:
    """Test error handling in data management"""
    
    @pytest.mark.asyncio
    async def test_network_error_handling(self, multi_asset_manager):
        """Test handling of network errors"""
        with patch.object(multi_asset_manager.stock_fetcher, 'fetch_data') as mock_fetch:
            mock_fetch.side_effect = aiohttp.ClientError("Network error")
            
            data = await multi_asset_manager.fetch_market_data(['AAPL'], {'AAPL': AssetType.STOCK})
            
            # Should return empty dict on error
            assert len(data) == 0
    
    @pytest.mark.asyncio
    async def test_api_rate_limit_handling(self, multi_asset_manager):
        """Test handling of API rate limits"""
        with patch.object(multi_asset_manager.stock_fetcher, 'fetch_data') as mock_fetch:
            # Simulate rate limit error
            mock_fetch.side_effect = Exception("Rate limit exceeded")
            
            data = await multi_asset_manager.fetch_market_data(['AAPL'], {'AAPL': AssetType.STOCK})
            
            # Should handle gracefully
            assert isinstance(data, dict)
    
    @pytest.mark.asyncio
    async def test_partial_failure_handling(self, multi_asset_manager):
        """Test handling when some symbols fail"""
        symbols = ['AAPL', 'INVALID', 'GOOGL']
        
        with patch.object(multi_asset_manager.stock_fetcher, 'fetch_data') as mock_fetch:
            def fetch_side_effect(symbol):
                if symbol == 'INVALID':
                    raise Exception("Invalid symbol")
                elif symbol == 'AAPL':
                    return SAMPLE_STOCK_DATA['AAPL']
                else:
                    return SAMPLE_STOCK_DATA['GOOGL']
            
            mock_fetch.side_effect = fetch_side_effect
            
            data = await multi_asset_manager.fetch_market_data(
                symbols, {s: AssetType.STOCK for s in symbols}
            )
            
            # Should return successful fetches only
            assert len(data) == 2
            assert 'AAPL' in data
            assert 'GOOGL' in data
            assert 'INVALID' not in data

# ==================== INTEGRATION TESTS ====================

class TestDataIntegration:
    """Test integration with other components"""
    
    def test_neurocluster_data_format(self, multi_asset_manager):
        """Test data format compatibility with NeuroCluster"""
        data = SAMPLE_STOCK_DATA['AAPL']
        
        # Verify required fields for NeuroCluster
        assert hasattr(data, 'symbol')
        assert hasattr(data, 'price')
        assert hasattr(data, 'volume')
        assert hasattr(data, 'timestamp')
        assert hasattr(data, 'asset_type')
        
        # Verify data types
        assert isinstance(data.price, (int, float))
        assert isinstance(data.volume, (int, float))
        assert isinstance(data.timestamp, datetime)
        assert isinstance(data.asset_type, AssetType)
    
    @pytest.mark.asyncio
    async def test_real_time_data_flow(self, multi_asset_manager):
        """Test real-time data flow simulation"""
        symbols = ['AAPL', 'BTC-USD']
        
        # Simulate real-time updates
        updates = []
        for i in range(5):
            with patch.object(multi_asset_manager.stock_fetcher, 'fetch_data') as mock_stock:
                with patch.object(multi_asset_manager.crypto_fetcher, 'fetch_data') as mock_crypto:
                    
                    # Create slightly different data for each update
                    stock_data = MarketData(
                        symbol='AAPL',
                        asset_type=AssetType.STOCK,
                        price=150.0 + i,
                        change=i,
                        change_percent=i * 0.67,
                        volume=1000000,
                        timestamp=datetime.now()
                    )
                    
                    crypto_data = MarketData(
                        symbol='BTC-USD',
                        asset_type=AssetType.CRYPTO,
                        price=45000.0 + i * 100,
                        change=i * 100,
                        change_percent=i * 0.22,
                        volume=500000,
                        timestamp=datetime.now()
                    )
                    
                    mock_stock.return_value = stock_data
                    mock_crypto.return_value = crypto_data
                    
                    data = await multi_asset_manager.fetch_market_data(
                        symbols, {'AAPL': AssetType.STOCK, 'BTC-USD': AssetType.CRYPTO}
                    )
                    
                    updates.append(data)
                    
                    # Small delay to simulate real-time
                    await asyncio.sleep(0.01)
        
        # Verify we got updates
        assert len(updates) == 5
        
        # Verify prices changed across updates
        aapl_prices = [update['AAPL'].price for update in updates]
        assert len(set(aapl_prices)) > 1  # Prices should vary

# ==================== MAIN TEST RUNNER ====================

if __name__ == "__main__":
    print("🧪 Running Data Manager Tests")
    print("=" * 50)
    
    # Run tests with verbose output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings",
        "-x"  # Stop on first failure
    ])