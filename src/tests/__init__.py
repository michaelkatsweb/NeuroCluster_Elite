#!/usr/bin/env python3
"""
File: __init__.py
Path: NeuroCluster-Elite/src/tests/__init__.py
Description: Test package initialization for NeuroCluster Elite

This module initializes the test package and provides utilities for comprehensive
testing of the NeuroCluster Elite trading platform including unit tests,
integration tests, and performance benchmarks.

Features:
- Test configuration and fixtures
- Mock data generators
- Test utilities and helpers
- Performance benchmarking tools
- Integration test setup
- Continuous integration support

Author: Your Name
Created: 2025-06-30
Version: 1.0.0
License: MIT
"""

import os
import sys
import asyncio
import logging
import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock, patch
import json

# Add src to path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Configure test logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise during tests
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test logger
logger = logging.getLogger(__name__)

# ==================== TEST CONFIGURATION ====================

@dataclass
class TestConfig:
    """Test configuration settings"""
    
    # Test data settings
    test_data_dir: str = "test_data"
    temp_dir: Optional[str] = None
    cleanup_after_tests: bool = True
    
    # Mock settings
    use_mock_data: bool = True
    mock_api_calls: bool = True
    mock_file_io: bool = False
    
    # Performance settings
    benchmark_enabled: bool = True
    performance_threshold: float = 1.0  # seconds
    memory_threshold: int = 100  # MB
    
    # Integration test settings
    run_integration_tests: bool = False
    live_api_testing: bool = False
    test_with_real_data: bool = False
    
    # Paper trading settings
    paper_trading_initial_capital: float = 100000.0
    paper_trading_symbols: List[str] = None
    
    def __post_init__(self):
        if self.paper_trading_symbols is None:
            self.paper_trading_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'BTC/USD', 'ETH/USD']

# Global test configuration
TEST_CONFIG = TestConfig()

# ==================== TEST FIXTURES ====================

class TestDataGenerator:
    """Generate mock data for testing"""
    
    @staticmethod
    def generate_market_data(symbol: str = "AAPL", 
                           start_date: datetime = None,
                           end_date: datetime = None,
                           frequency: str = "1min") -> pd.DataFrame:
        """Generate realistic market data"""
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        # Generate time series
        if frequency == "1min":
            periods = int((end_date - start_date).total_seconds() / 60)
            dates = pd.date_range(start=start_date, end=end_date, periods=periods)
        elif frequency == "1h":
            periods = int((end_date - start_date).total_seconds() / 3600)
            dates = pd.date_range(start=start_date, end=end_date, periods=periods)
        else:  # daily
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic price data using geometric brownian motion
        np.random.seed(42)  # For reproducible tests
        initial_price = 150.0 if symbol == "AAPL" else 100.0
        
        # Parameters for realistic price movement
        mu = 0.0001  # drift
        sigma = 0.02  # volatility
        
        # Generate price series
        dt = 1.0 / 252 / (24 * 60) if frequency == "1min" else 1.0 / 252  # time step
        dW = np.random.normal(0, np.sqrt(dt), len(dates))  # Brownian motion
        
        # Geometric Brownian Motion
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * dW
        prices = initial_price * np.exp(np.cumsum(log_returns))
        
        # Generate OHLCV data
        high_noise = np.random.uniform(1.001, 1.01, len(prices))
        low_noise = np.random.uniform(0.99, 0.999, len(prices))
        
        data = {
            'timestamp': dates,
            'open': np.roll(prices, 1),
            'high': prices * high_noise,
            'low': prices * low_noise,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, len(prices))
        }
        
        data['open'][0] = initial_price  # Fix first open price
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    @staticmethod
    def generate_portfolio_data(symbols: List[str] = None, 
                              initial_capital: float = 100000.0) -> Dict[str, Any]:
        """Generate mock portfolio data"""
        
        if symbols is None:
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        
        positions = {}
        total_value = 0
        
        for symbol in symbols:
            quantity = np.random.randint(10, 100)
            price = np.random.uniform(100, 300)
            value = quantity * price
            
            positions[symbol] = {
                'symbol': symbol,
                'quantity': quantity,
                'avg_cost': price * np.random.uniform(0.9, 1.1),
                'current_price': price,
                'market_value': value,
                'unrealized_pnl': value * np.random.uniform(-0.1, 0.1),
                'realized_pnl': np.random.uniform(-1000, 1000)
            }
            
            total_value += value
        
        cash_balance = initial_capital - total_value * 0.8
        
        return {
            'positions': positions,
            'cash_balance': cash_balance,
            'total_value': total_value + cash_balance,
            'daily_pnl': (total_value + cash_balance) * np.random.uniform(-0.05, 0.05),
            'total_pnl': (total_value + cash_balance) - initial_capital
        }
    
    @staticmethod
    def generate_trading_signals(count: int = 10) -> List[Dict[str, Any]]:
        """Generate mock trading signals"""
        
        signals = []
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'BTC/USD', 'ETH/USD']
        signal_types = ['BUY', 'SELL', 'HOLD']
        strategies = ['momentum', 'mean_reversion', 'breakout', 'neural_cluster']
        
        for i in range(count):
            signal = {
                'id': f"signal_{i:03d}",
                'symbol': np.random.choice(symbols),
                'signal_type': np.random.choice(signal_types),
                'strategy_name': np.random.choice(strategies),
                'confidence': np.random.uniform(60, 95),
                'entry_price': np.random.uniform(100, 300),
                'stop_loss': np.random.uniform(90, 110),
                'take_profit': np.random.uniform(290, 350),
                'reasoning': f"Test signal {i} generated by mock strategy",
                'timestamp': datetime.now() - timedelta(minutes=np.random.randint(0, 1440))
            }
            signals.append(signal)
        
        return signals

# ==================== TEST UTILITIES ====================

class AsyncTestCase:
    """Base class for async test cases"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def teardown_method(self):
        """Cleanup after each test method"""
        self.loop.close()
    
    def run_async(self, coro):
        """Run async function in test"""
        return self.loop.run_until_complete(coro)

class MockDataProvider:
    """Provides mock data for testing"""
    
    def __init__(self):
        self.market_data_cache = {}
        self.portfolio_cache = None
        self.signals_cache = []
    
    def get_market_data(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Get cached or generate market data"""
        cache_key = f"{symbol}_{kwargs.get('frequency', '1min')}"
        
        if cache_key not in self.market_data_cache:
            self.market_data_cache[cache_key] = TestDataGenerator.generate_market_data(
                symbol=symbol, **kwargs
            )
        
        return self.market_data_cache[cache_key].copy()
    
    def get_portfolio_data(self, **kwargs) -> Dict[str, Any]:
        """Get cached or generate portfolio data"""
        if self.portfolio_cache is None:
            self.portfolio_cache = TestDataGenerator.generate_portfolio_data(**kwargs)
        
        return self.portfolio_cache.copy()
    
    def get_trading_signals(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get cached or generate trading signals"""
        if len(self.signals_cache) < count:
            self.signals_cache = TestDataGenerator.generate_trading_signals(count)
        
        return self.signals_cache[:count].copy()

# Global mock data provider
MOCK_DATA_PROVIDER = MockDataProvider()

# ==================== TEST DECORATORS ====================

def requires_integration(func):
    """Decorator to skip tests that require integration setup"""
    return pytest.mark.skipif(
        not TEST_CONFIG.run_integration_tests,
        reason="Integration tests disabled"
    )(func)

def requires_live_api(func):
    """Decorator to skip tests that require live API access"""
    return pytest.mark.skipif(
        not TEST_CONFIG.live_api_testing,
        reason="Live API testing disabled"
    )(func)

def benchmark(threshold: float = None):
    """Decorator to benchmark test performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            result = func(*args, **kwargs)
            
            elapsed = time.time() - start_time
            max_time = threshold or TEST_CONFIG.performance_threshold
            
            if elapsed > max_time:
                logger.warning(f"Test {func.__name__} took {elapsed:.2f}s (threshold: {max_time}s)")
            
            return result
        return wrapper
    return decorator

# ==================== PYTEST FIXTURES ====================

@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration"""
    return TEST_CONFIG

@pytest.fixture(scope="session")
def temp_test_dir():
    """Create temporary directory for tests"""
    temp_dir = tempfile.mkdtemp(prefix="neurocluster_test_")
    yield temp_dir
    if TEST_CONFIG.cleanup_after_tests:
        shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(scope="function")
def mock_data_provider():
    """Provide mock data provider"""
    return MockDataProvider()

@pytest.fixture(scope="function")
def sample_market_data():
    """Provide sample market data"""
    return TestDataGenerator.generate_market_data()

@pytest.fixture(scope="function")
def sample_portfolio():
    """Provide sample portfolio data"""
    return TestDataGenerator.generate_portfolio_data()

@pytest.fixture(scope="function")
def sample_signals():
    """Provide sample trading signals"""
    return TestDataGenerator.generate_trading_signals()

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# ==================== MOCK CONFIGURATIONS ====================

class MockIntegrationConfig:
    """Mock integration configuration for testing"""
    
    @staticmethod
    def get_paper_trading_config():
        """Get paper trading configuration"""
        return {
            'name': 'paper_trading',
            'enabled': True,
            'paper_trading': True,
            'initial_capital': TEST_CONFIG.paper_trading_initial_capital
        }
    
    @staticmethod
    def get_mock_broker_config():
        """Get mock broker configuration"""
        return {
            'name': 'mock_broker',
            'enabled': True,
            'paper_trading': True,
            'api_key': 'test_key',
            'secret_key': 'test_secret'
        }
    
    @staticmethod
    def get_mock_exchange_config():
        """Get mock exchange configuration"""
        return {
            'name': 'mock_exchange',
            'enabled': True,
            'sandbox_mode': True,
            'api_key': 'test_key',
            'secret_key': 'test_secret'
        }

# ==================== TEST ASSERTIONS ====================

class CustomAssertions:
    """Custom assertions for trading tests"""
    
    @staticmethod
    def assert_valid_price(price: float, min_price: float = 0.01):
        """Assert price is valid"""
        assert isinstance(price, (int, float)), f"Price must be numeric, got {type(price)}"
        assert price >= min_price, f"Price {price} must be >= {min_price}"
    
    @staticmethod
    def assert_valid_quantity(quantity: float, min_quantity: float = 0):
        """Assert quantity is valid"""
        assert isinstance(quantity, (int, float)), f"Quantity must be numeric, got {type(quantity)}"
        assert quantity >= min_quantity, f"Quantity {quantity} must be >= {min_quantity}"
    
    @staticmethod
    def assert_valid_percentage(percentage: float, min_pct: float = -100, max_pct: float = 1000):
        """Assert percentage is valid"""
        assert isinstance(percentage, (int, float)), f"Percentage must be numeric, got {type(percentage)}"
        assert min_pct <= percentage <= max_pct, f"Percentage {percentage} must be between {min_pct} and {max_pct}"
    
    @staticmethod
    def assert_trading_signal_valid(signal: Dict[str, Any]):
        """Assert trading signal is valid"""
        required_fields = ['symbol', 'signal_type', 'confidence', 'timestamp']
        for field in required_fields:
            assert field in signal, f"Signal missing required field: {field}"
        
        assert signal['signal_type'] in ['BUY', 'SELL', 'HOLD'], f"Invalid signal type: {signal['signal_type']}"
        assert 0 <= signal['confidence'] <= 100, f"Invalid confidence: {signal['confidence']}"
        assert isinstance(signal['timestamp'], datetime), f"Invalid timestamp type: {type(signal['timestamp'])}"
    
    @staticmethod
    def assert_portfolio_valid(portfolio: Dict[str, Any]):
        """Assert portfolio data is valid"""
        required_fields = ['positions', 'cash_balance', 'total_value']
        for field in required_fields:
            assert field in portfolio, f"Portfolio missing required field: {field}"
        
        assert isinstance(portfolio['positions'], dict), "Positions must be a dictionary"
        assert portfolio['cash_balance'] >= 0, f"Cash balance cannot be negative: {portfolio['cash_balance']}"
        assert portfolio['total_value'] > 0, f"Total value must be positive: {portfolio['total_value']}"

# ==================== PERFORMANCE MONITORING ====================

class PerformanceMonitor:
    """Monitor test performance"""
    
    def __init__(self):
        self.test_times = {}
        self.memory_usage = {}
    
    def start_test(self, test_name: str):
        """Start monitoring a test"""
        import time
        import psutil
        
        self.test_times[test_name] = {
            'start': time.time(),
            'memory_start': psutil.Process().memory_info().rss / 1024 / 1024  # MB
        }
    
    def end_test(self, test_name: str):
        """End monitoring a test"""
        import time
        import psutil
        
        if test_name not in self.test_times:
            return
        
        end_time = time.time()
        memory_end = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        test_data = self.test_times[test_name]
        duration = end_time - test_data['start']
        memory_delta = memory_end - test_data['memory_start']
        
        self.test_times[test_name].update({
            'duration': duration,
            'memory_delta': memory_delta,
            'passed_threshold': duration <= TEST_CONFIG.performance_threshold
        })
        
        return {
            'duration': duration,
            'memory_delta': memory_delta,
            'within_threshold': duration <= TEST_CONFIG.performance_threshold
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.test_times:
            return {'total_tests': 0}
        
        durations = [data.get('duration', 0) for data in self.test_times.values() if 'duration' in data]
        memory_deltas = [data.get('memory_delta', 0) for data in self.test_times.values() if 'memory_delta' in data]
        
        return {
            'total_tests': len(durations),
            'avg_duration': np.mean(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'avg_memory_delta': np.mean(memory_deltas) if memory_deltas else 0,
            'max_memory_delta': max(memory_deltas) if memory_deltas else 0,
            'tests_over_threshold': sum(1 for d in durations if d > TEST_CONFIG.performance_threshold)
        }

# Global performance monitor
PERFORMANCE_MONITOR = PerformanceMonitor()

# ==================== TEST RUNNER UTILITIES ====================

def run_all_tests(verbose: bool = True, coverage: bool = True):
    """Run all tests with optional coverage"""
    import subprocess
    import sys
    
    cmd = [sys.executable, "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    # Add test directory
    cmd.append("src/tests/")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if verbose:
            print("STDOUT:")
            print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return False

def setup_test_environment():
    """Setup test environment"""
    
    # Create test directories
    test_dirs = [
        TEST_CONFIG.test_data_dir,
        "logs",
        "temp"
    ]
    
    for dir_name in test_dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    # Set environment variables for testing
    os.environ['NEUROCLUSTER_TESTING'] = 'true'
    os.environ['NEUROCLUSTER_LOG_LEVEL'] = 'WARNING'
    
    logger.info("✅ Test environment setup complete")

def cleanup_test_environment():
    """Cleanup test environment"""
    
    if TEST_CONFIG.cleanup_after_tests:
        # Remove test directories
        test_dirs = [
            TEST_CONFIG.test_data_dir,
            "temp"
        ]
        
        for dir_name in test_dirs:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name, ignore_errors=True)
    
    # Clean environment variables
    test_env_vars = [
        'NEUROCLUSTER_TESTING',
        'NEUROCLUSTER_LOG_LEVEL'
    ]
    
    for var in test_env_vars:
        if var in os.environ:
            del os.environ[var]
    
    logger.info("✅ Test environment cleanup complete")

# ==================== EXPORTS ====================

__all__ = [
    # Configuration
    'TestConfig',
    'TEST_CONFIG',
    
    # Data generation
    'TestDataGenerator',
    'MockDataProvider',
    'MOCK_DATA_PROVIDER',
    
    # Test utilities
    'AsyncTestCase',
    'MockIntegrationConfig',
    'CustomAssertions',
    
    # Decorators
    'requires_integration',
    'requires_live_api',
    'benchmark',
    
    # Performance monitoring
    'PerformanceMonitor',
    'PERFORMANCE_MONITOR',
    
    # Test runner
    'run_all_tests',
    'setup_test_environment',
    'cleanup_test_environment'
]

# Auto-setup on import
if os.getenv('NEUROCLUSTER_TESTING') != 'true':
    setup_test_environment()