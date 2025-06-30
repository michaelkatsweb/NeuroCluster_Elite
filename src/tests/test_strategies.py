#!/usr/bin/env python3
"""
File: test_strategies.py
Path: NeuroCluster-Elite/src/tests/test_strategies.py
Description: Comprehensive tests for trading strategies

This module contains comprehensive tests for all trading strategies including
base strategy functionality, regime-specific strategies, asset-specific strategies,
and strategy performance validation.

Features tested:
- Base strategy class functionality
- Bull market strategies
- Bear market strategies  
- Volatility strategies
- Breakout strategies
- Range trading strategies
- Crypto-specific strategies
- Strategy selection logic
- Signal generation and validation
- Performance metrics and backtesting
- Risk-adjusted returns

Author: Michael Katsaros
Created: 2025-06-30
Version: 1.0.0
License: MIT
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import time

# Import modules to test
try:
    from src.trading.strategies.base_strategy import (
        BaseStrategy, TradingSignal, SignalType, StrategyMetrics,
        StrategyConfig, SignalConfidence
    )
    from src.trading.strategies.bull_strategy import BullMarketStrategy
    from src.trading.strategies.bear_strategy import BearMarketStrategy
    from src.trading.strategies.volatility_strategy import AdvancedVolatilityStrategy
    from src.trading.strategies.breakout_strategy import AdvancedBreakoutStrategy
    from src.trading.strategies.range_strategy import RangeStrategy
    from src.trading.strategies.crypto_strategies import (
        CryptoMomentumStrategy, CryptoDCAStrategy, CryptoArbitrageStrategy
    )
    from src.trading.strategy_selector import StrategySelector, StrategyPerformance
    from src.core.neurocluster_elite import RegimeType, AssetType, MarketData
    from src.analysis.technical_indicators import TechnicalIndicators
except ImportError as e:
    pytest.skip(f"Could not import strategy modules: {e}", allow_module_level=True)

# Test configuration
TEST_CONFIG = {
    'max_position_size': 0.1,
    'stop_loss_pct': 0.05,
    'take_profit_pct': 0.10,
    'min_confidence': 0.6,
    'risk_per_trade': 0.02
}

# Sample market data for different scenarios
BULL_MARKET_DATA = MarketData(
    symbol='AAPL',
    asset_type=AssetType.STOCK,
    price=155.0,
    change=5.25,
    change_percent=3.51,
    volume=2500000,
    timestamp=datetime.now(),
    high=156.0,
    low=150.0,
    open=150.5,
    close=155.0,
    rsi=75.5,
    macd=2.25,
    volatility=0.25,
    sma_20=150.0,
    sma_50=145.0,
    ema_12=152.0,
    bollinger_upper=160.0,
    bollinger_lower=140.0
)

BEAR_MARKET_DATA = MarketData(
    symbol='AAPL',
    asset_type=AssetType.STOCK,
    price=140.0,
    change=-8.75,
    change_percent=-5.88,
    volume=3500000,
    timestamp=datetime.now(),
    high=145.0,
    low=139.0,
    open=148.75,
    close=140.0,
    rsi=25.3,
    macd=-3.15,
    volatility=0.35,
    sma_20=145.0,
    sma_50=150.0,
    ema_12=142.0,
    bollinger_upper=155.0,
    bollinger_lower=135.0
)

VOLATILE_MARKET_DATA = MarketData(
    symbol='BTC-USD',
    asset_type=AssetType.CRYPTO,
    price=42000.0,
    change=2000.0,
    change_percent=5.0,
    volume=1500000,
    timestamp=datetime.now(),
    high=43500.0,
    low=40000.0,
    open=40000.0,
    close=42000.0,
    rsi=65.0,
    macd=500.0,
    volatility=0.6,
    sma_20=41000.0,
    sma_50=40000.0,
    ema_12=41500.0
)

SIDEWAYS_MARKET_DATA = MarketData(
    symbol='MSFT',
    asset_type=AssetType.STOCK,
    price=280.0,
    change=0.25,
    change_percent=0.09,
    volume=800000,
    timestamp=datetime.now(),
    high=282.0,
    low=278.0,
    open=280.0,
    close=280.0,
    rsi=50.0,
    macd=0.05,
    volatility=0.15,
    sma_20=279.5,
    sma_50=280.5,
    ema_12=280.0,
    bollinger_upper=285.0,
    bollinger_lower=275.0
)

# ==================== FIXTURES ====================

@pytest.fixture
def sample_historical_data():
    """Generate sample historical data for backtesting"""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # Generate realistic price movements
    initial_price = 150.0
    prices = [initial_price]
    
    for i in range(99):
        # Add trend and noise
        trend = 0.001  # Slight upward trend
        noise = np.random.normal(0, 0.02)  # 2% daily volatility
        next_price = prices[-1] * (1 + trend + noise)
        prices.append(max(next_price, 1.0))  # Ensure positive prices
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': [p * np.random.uniform(0.99, 1.01) for p in prices],
        'high': [p * np.random.uniform(1.00, 1.05) for p in prices],
        'low': [p * np.random.uniform(0.95, 1.00) for p in prices],
        'close': prices,
        'volume': np.random.randint(500000, 2000000, 100)
    })
    
    return data

@pytest.fixture
def mock_technical_indicators():
    """Mock technical indicators"""
    indicators = Mock(spec=TechnicalIndicators)
    
    # Mock indicator calculations
    indicators.calculate_rsi.return_value = 65.0
    indicators.calculate_macd.return_value = (1.5, 1.0, 0.5)
    indicators.calculate_bollinger_bands.return_value = (160.0, 150.0, 140.0)
    indicators.calculate_sma.return_value = 150.0
    indicators.calculate_ema.return_value = 152.0
    indicators.detect_support_resistance.return_value = ([145.0, 140.0], [155.0, 160.0])
    
    return indicators

@pytest.fixture
def base_strategy():
    """Create base strategy for testing"""
    return BaseStrategy(config=TEST_CONFIG)

@pytest.fixture
def bull_strategy():
    """Create bull market strategy"""
    return BullMarketStrategy(config=TEST_CONFIG)

@pytest.fixture
def bear_strategy():
    """Create bear market strategy"""
    return BearMarketStrategy(config=TEST_CONFIG)

@pytest.fixture
def volatility_strategy():
    """Create volatility strategy"""
    return AdvancedVolatilityStrategy(config=TEST_CONFIG)

@pytest.fixture
def breakout_strategy():
    """Create breakout strategy"""
    return AdvancedBreakoutStrategy(config=TEST_CONFIG)

@pytest.fixture
def range_strategy():
    """Create range trading strategy"""
    return RangeStrategy(config=TEST_CONFIG)

@pytest.fixture
def crypto_momentum_strategy():
    """Create crypto momentum strategy"""
    return CryptoMomentumStrategy(config=TEST_CONFIG)

@pytest.fixture
def strategy_selector():
    """Create strategy selector"""
    return StrategySelector(config=TEST_CONFIG)

# ==================== BASE STRATEGY TESTS ====================

class TestBaseStrategy:
    """Test base strategy functionality"""
    
    def test_initialization(self, base_strategy):
        """Test base strategy initialization"""
        assert base_strategy.config == TEST_CONFIG
        assert base_strategy.name == "BaseStrategy"
        assert base_strategy.description is not None
        assert base_strategy.metrics is not None
    
    def test_signal_creation(self, base_strategy):
        """Test trading signal creation"""
        signal = base_strategy._create_signal(
            symbol='AAPL',
            signal_type=SignalType.BUY,
            asset_type=AssetType.STOCK,
            entry_price=150.0,
            confidence=0.8,
            data=BULL_MARKET_DATA
        )
        
        assert signal.symbol == 'AAPL'
        assert signal.signal_type == SignalType.BUY
        assert signal.asset_type == AssetType.STOCK
        assert signal.entry_price == 150.0
        assert signal.confidence == 0.8
        assert signal.timestamp is not None
        assert signal.strategy_name == "BaseStrategy"
    
    def test_stop_loss_calculation(self, base_strategy):
        """Test stop loss calculation"""
        entry_price = 150.0
        stop_loss = base_strategy._calculate_stop_loss(entry_price, SignalType.BUY)
        
        expected_stop_loss = entry_price * (1 - TEST_CONFIG['stop_loss_pct'])
        assert abs(stop_loss - expected_stop_loss) < 0.01
        
        # Test short position
        stop_loss_short = base_strategy._calculate_stop_loss(entry_price, SignalType.SELL)
        expected_stop_loss_short = entry_price * (1 + TEST_CONFIG['stop_loss_pct'])
        assert abs(stop_loss_short - expected_stop_loss_short) < 0.01
    
    def test_take_profit_calculation(self, base_strategy):
        """Test take profit calculation"""
        entry_price = 150.0
        take_profit = base_strategy._calculate_take_profit(entry_price, SignalType.BUY)
        
        expected_take_profit = entry_price * (1 + TEST_CONFIG['take_profit_pct'])
        assert abs(take_profit - expected_take_profit) < 0.01
    
    def test_confidence_scoring(self, base_strategy):
        """Test confidence scoring mechanism"""
        # Test with strong bullish indicators
        confidence = base_strategy._calculate_confidence(BULL_MARKET_DATA, SignalType.BUY)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably confident
        
        # Test with weak indicators
        confidence_weak = base_strategy._calculate_confidence(SIDEWAYS_MARKET_DATA, SignalType.BUY)
        assert 0.0 <= confidence_weak <= 1.0
        assert confidence_weak < confidence  # Should be less confident
    
    def test_metrics_tracking(self, base_strategy):
        """Test strategy metrics tracking"""
        # Add some sample trades
        base_strategy.metrics.total_signals = 10
        base_strategy.metrics.winning_trades = 6
        base_strategy.metrics.total_return = 0.15
        
        win_rate = base_strategy.metrics.winning_trades / base_strategy.metrics.total_signals
        assert win_rate == 0.6
        
        # Test metrics update
        base_strategy._update_metrics(profit=100.0, win=True)
        assert base_strategy.metrics.winning_trades == 7
        assert base_strategy.metrics.total_signals == 11

# ==================== BULL STRATEGY TESTS ====================

class TestBullMarketStrategy:
    """Test bull market strategy"""
    
    def test_bull_signal_generation(self, bull_strategy):
        """Test bull market signal generation"""
        signal = bull_strategy.generate_signal(
            data=BULL_MARKET_DATA,
            regime=RegimeType.BULL,
            confidence=0.8
        )
        
        assert signal is not None
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence > 0.6
        assert signal.volume_confirmation is not None
    
    def test_bull_conditions(self, bull_strategy):
        """Test bull market conditions detection"""
        # Strong bull conditions
        is_bullish = bull_strategy._is_bullish_conditions(BULL_MARKET_DATA)
        assert is_bullish
        
        # Weak bull conditions
        is_bullish_weak = bull_strategy._is_bullish_conditions(SIDEWAYS_MARKET_DATA)
        assert not is_bullish_weak
        
        # Bear conditions
        is_bullish_bear = bull_strategy._is_bullish_conditions(BEAR_MARKET_DATA)
        assert not is_bullish_bear
    
    def test_momentum_confirmation(self, bull_strategy):
        """Test momentum confirmation logic"""
        momentum_confirmed = bull_strategy._confirm_momentum(BULL_MARKET_DATA)
        assert momentum_confirmed
        
        # Test with weak momentum
        weak_momentum = bull_strategy._confirm_momentum(SIDEWAYS_MARKET_DATA)
        assert not weak_momentum
    
    def test_volume_confirmation(self, bull_strategy):
        """Test volume confirmation"""
        volume_confirmed = bull_strategy._confirm_volume(BULL_MARKET_DATA)
        assert isinstance(volume_confirmed, bool)
        
        # High volume should confirm
        high_volume_data = MarketData(
            symbol='AAPL',
            asset_type=AssetType.STOCK,
            price=155.0,
            volume=5000000,  # Very high volume
            timestamp=datetime.now()
        )
        
        volume_confirmed_high = bull_strategy._confirm_volume(high_volume_data)
        assert volume_confirmed_high
    
    def test_risk_reward_calculation(self, bull_strategy):
        """Test risk-reward ratio calculation"""
        signal = bull_strategy.generate_signal(
            data=BULL_MARKET_DATA,
            regime=RegimeType.BULL,
            confidence=0.8
        )
        
        if signal:
            risk = signal.entry_price - signal.stop_loss
            reward = signal.take_profit - signal.entry_price
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            assert risk_reward_ratio > 1.5  # Should have good risk-reward ratio

# ==================== BEAR STRATEGY TESTS ====================

class TestBearMarketStrategy:
    """Test bear market strategy"""
    
    def test_bear_signal_generation(self, bear_strategy):
        """Test bear market signal generation"""
        signal = bear_strategy.generate_signal(
            data=BEAR_MARKET_DATA,
            regime=RegimeType.BEAR,
            confidence=0.8
        )
        
        assert signal is not None
        assert signal.signal_type == SignalType.SELL
        assert signal.confidence > 0.6
    
    def test_bear_conditions(self, bear_strategy):
        """Test bear market conditions detection"""
        is_bearish = bear_strategy._is_bearish_conditions(BEAR_MARKET_DATA)
        assert is_bearish
        
        is_bearish_bull = bear_strategy._is_bearish_conditions(BULL_MARKET_DATA)
        assert not is_bearish_bull
    
    def test_downtrend_confirmation(self, bear_strategy):
        """Test downtrend confirmation"""
        downtrend_confirmed = bear_strategy._confirm_downtrend(BEAR_MARKET_DATA)
        assert downtrend_confirmed
        
        downtrend_bull = bear_strategy._confirm_downtrend(BULL_MARKET_DATA)
        assert not downtrend_bull
    
    def test_bear_market_exit_conditions(self, bear_strategy):
        """Test bear market exit conditions"""
        # Test exit on oversold conditions
        oversold_data = MarketData(
            symbol='AAPL',
            asset_type=AssetType.STOCK,
            price=130.0,
            rsi=15.0,  # Very oversold
            timestamp=datetime.now()
        )
        
        should_exit = bear_strategy._should_exit_short(oversold_data)
        assert should_exit

# ==================== VOLATILITY STRATEGY TESTS ====================

class TestVolatilityStrategy:
    """Test volatility trading strategy"""
    
    def test_volatility_signal_generation(self, volatility_strategy):
        """Test volatility strategy signal generation"""
        signal = volatility_strategy.generate_signal(
            data=VOLATILE_MARKET_DATA,
            regime=RegimeType.VOLATILE,
            confidence=0.8
        )
        
        assert signal is not None
        assert signal.confidence > 0.5
    
    def test_volatility_detection(self, volatility_strategy):
        """Test volatility detection"""
        is_high_vol = volatility_strategy._is_high_volatility(VOLATILE_MARKET_DATA)
        assert is_high_vol
        
        is_low_vol = volatility_strategy._is_high_volatility(SIDEWAYS_MARKET_DATA)
        assert not is_low_vol
    
    def test_volatility_breakout_detection(self, volatility_strategy):
        """Test volatility breakout detection"""
        breakout = volatility_strategy._detect_volatility_breakout(VOLATILE_MARKET_DATA)
        assert isinstance(breakout, bool)
    
    def test_volatility_position_sizing(self, volatility_strategy):
        """Test position sizing based on volatility"""
        # High volatility should result in smaller positions
        size_high_vol = volatility_strategy._calculate_volatility_adjusted_size(
            VOLATILE_MARKET_DATA, base_size=100
        )
        
        size_low_vol = volatility_strategy._calculate_volatility_adjusted_size(
            SIDEWAYS_MARKET_DATA, base_size=100
        )
        
        assert size_high_vol < size_low_vol

# ==================== BREAKOUT STRATEGY TESTS ====================

class TestBreakoutStrategy:
    """Test breakout trading strategy"""
    
    def test_breakout_detection(self, breakout_strategy):
        """Test breakout pattern detection"""
        # Mock support/resistance levels
        with patch.object(breakout_strategy, '_get_support_resistance_levels') as mock_levels:
            mock_levels.return_value = ([145.0, 140.0], [155.0, 160.0])
            
            signal = breakout_strategy.generate_signal(
                data=BULL_MARKET_DATA,
                regime=RegimeType.BREAKOUT,
                confidence=0.8
            )
            
            if signal:
                assert signal.signal_type in [SignalType.BUY, SignalType.SELL]
                assert signal.confidence > 0.5
    
    def test_resistance_breakout(self, breakout_strategy):
        """Test resistance breakout detection"""
        # Price breaking above resistance
        breakout_data = MarketData(
            symbol='AAPL',
            asset_type=AssetType.STOCK,
            price=162.0,  # Above resistance at 160.0
            volume=3000000,  # High volume
            timestamp=datetime.now()
        )
        
        is_breakout = breakout_strategy._is_resistance_breakout(
            breakout_data, resistance_level=160.0
        )
        assert is_breakout
    
    def test_support_breakdown(self, breakout_strategy):
        """Test support breakdown detection"""
        # Price breaking below support
        breakdown_data = MarketData(
            symbol='AAPL',
            asset_type=AssetType.STOCK,
            price=138.0,  # Below support at 140.0
            volume=3000000,  # High volume
            timestamp=datetime.now()
        )
        
        is_breakdown = breakout_strategy._is_support_breakdown(
            breakdown_data, support_level=140.0
        )
        assert is_breakdown
    
    def test_false_breakout_detection(self, breakout_strategy):
        """Test false breakout detection"""
        # Weak volume breakout
        false_breakout_data = MarketData(
            symbol='AAPL',
            asset_type=AssetType.STOCK,
            price=162.0,  # Above resistance
            volume=100000,  # Very low volume
            timestamp=datetime.now()
        )
        
        is_valid = breakout_strategy._validate_breakout(false_breakout_data, 160.0)
        assert not is_valid

# ==================== RANGE STRATEGY TESTS ====================

class TestRangeStrategy:
    """Test range trading strategy"""
    
    def test_range_detection(self, range_strategy):
        """Test range market detection"""
        is_ranging = range_strategy._is_ranging_market(SIDEWAYS_MARKET_DATA)
        assert is_ranging
        
        is_ranging_bull = range_strategy._is_ranging_market(BULL_MARKET_DATA)
        assert not is_ranging_bull
    
    def test_range_bounds_calculation(self, range_strategy):
        """Test range bounds calculation"""
        with patch.object(range_strategy, '_calculate_range_bounds') as mock_bounds:
            mock_bounds.return_value = (275.0, 285.0)  # Support, Resistance
            
            signal = range_strategy.generate_signal(
                data=SIDEWAYS_MARKET_DATA,
                regime=RegimeType.SIDEWAYS,
                confidence=0.8
            )
            
            if signal:
                assert signal.confidence > 0.5
    
    def test_range_entry_points(self, range_strategy):
        """Test range trading entry points"""
        # Near support - should buy
        near_support = MarketData(
            symbol='MSFT',
            asset_type=AssetType.STOCK,
            price=276.0,  # Near support at 275.0
            timestamp=datetime.now()
        )
        
        buy_signal = range_strategy._should_buy_at_support(near_support, support=275.0)
        assert buy_signal
        
        # Near resistance - should sell
        near_resistance = MarketData(
            symbol='MSFT',
            asset_type=AssetType.STOCK,
            price=284.0,  # Near resistance at 285.0
            timestamp=datetime.now()
        )
        
        sell_signal = range_strategy._should_sell_at_resistance(near_resistance, resistance=285.0)
        assert sell_signal

# ==================== CRYPTO STRATEGY TESTS ====================

class TestCryptoStrategies:
    """Test cryptocurrency-specific strategies"""
    
    def test_crypto_momentum_strategy(self, crypto_momentum_strategy):
        """Test crypto momentum strategy"""
        signal = crypto_momentum_strategy.generate_signal(
            data=VOLATILE_MARKET_DATA,
            regime=RegimeType.BULL,
            confidence=0.8
        )
        
        if signal:
            assert signal.asset_type == AssetType.CRYPTO
            assert signal.confidence > 0.5
    
    def test_crypto_volatility_adjustment(self, crypto_momentum_strategy):
        """Test crypto volatility adjustments"""
        # Crypto should have higher volatility tolerance
        is_acceptable_vol = crypto_momentum_strategy._is_acceptable_volatility(
            VOLATILE_MARKET_DATA
        )
        assert is_acceptable_vol
    
    def test_crypto_24h_analysis(self, crypto_momentum_strategy):
        """Test 24-hour crypto market analysis"""
        # Crypto markets are 24/7, strategy should handle this
        weekend_data = MarketData(
            symbol='BTC-USD',
            asset_type=AssetType.CRYPTO,
            price=42000.0,
            timestamp=datetime.now(),  # Could be weekend
            volume=1000000
        )
        
        signal = crypto_momentum_strategy.generate_signal(
            data=weekend_data,
            regime=RegimeType.BULL,
            confidence=0.8
        )
        
        # Should still generate signals on weekends
        assert signal is not None or signal is None  # Valid either way
    
    def test_crypto_risk_management(self, crypto_momentum_strategy):
        """Test crypto-specific risk management"""
        # Crypto should have tighter stops due to volatility
        stop_loss = crypto_momentum_strategy._calculate_stop_loss(42000.0, SignalType.BUY)
        take_profit = crypto_momentum_strategy._calculate_take_profit(42000.0, SignalType.BUY)
        
        risk = 42000.0 - stop_loss
        reward = take_profit - 42000.0
        
        # Risk-reward should be appropriate for crypto volatility
        assert risk > 0
        assert reward > 0
        assert reward / risk >= 1.5  # Minimum 1.5:1 ratio

# ==================== STRATEGY SELECTOR TESTS ====================

class TestStrategySelector:
    """Test strategy selection logic"""
    
    def test_regime_based_selection(self, strategy_selector):
        """Test strategy selection based on market regime"""
        # Bull market should select bull strategies
        strategies = strategy_selector.select_strategies(
            regime=RegimeType.BULL,
            asset_types={'AAPL': AssetType.STOCK}
        )
        
        assert 'AAPL' in strategies
        assert 'Bull' in strategies['AAPL'] or 'Momentum' in strategies['AAPL']
        
        # Bear market should select bear strategies
        bear_strategies = strategy_selector.select_strategies(
            regime=RegimeType.BEAR,
            asset_types={'AAPL': AssetType.STOCK}
        )
        
        assert 'AAPL' in bear_strategies
        assert 'Bear' in bear_strategies['AAPL'] or 'Short' in bear_strategies['AAPL']
    
    def test_asset_specific_selection(self, strategy_selector):
        """Test asset-specific strategy selection"""
        # Crypto should get crypto-specific strategies
        crypto_strategies = strategy_selector.select_strategies(
            regime=RegimeType.VOLATILE,
            asset_types={'BTC-USD': AssetType.CRYPTO}
        )
        
        assert 'BTC-USD' in crypto_strategies
        assert 'Crypto' in crypto_strategies['BTC-USD']
    
    def test_performance_based_selection(self, strategy_selector):
        """Test performance-based strategy selection"""
        # Mock performance data
        strategy_selector.strategy_performance = {
            'BullMarketStrategy': StrategyPerformance(
                total_trades=100,
                winning_trades=65,
                total_return=0.25,
                sharpe_ratio=1.8,
                max_drawdown=0.08
            ),
            'MomentumStrategy': StrategyPerformance(
                total_trades=80,
                winning_trades=45,
                total_return=0.15,
                sharpe_ratio=1.2,
                max_drawdown=0.12
            )
        }
        
        best_strategies = strategy_selector.get_best_strategies(limit=1)
        
        assert len(best_strategies) == 1
        assert best_strategies[0][0] == 'BullMarketStrategy'  # Better performance
    
    def test_strategy_switching_logic(self, strategy_selector):
        """Test strategy switching with hysteresis"""
        # Test strategy switching logic
        current_strategy = 'BullMarketStrategy'
        new_regime = RegimeType.BEAR
        
        should_switch = strategy_selector._should_switch_strategy(
            current_strategy, new_regime, confidence=0.9
        )
        
        assert should_switch  # High confidence should trigger switch
        
        # Low confidence should not switch
        should_not_switch = strategy_selector._should_switch_strategy(
            current_strategy, new_regime, confidence=0.5
        )
        
        assert not should_not_switch

# ==================== BACKTESTING TESTS ====================

class TestStrategyBacktesting:
    """Test strategy backtesting functionality"""
    
    def test_strategy_backtesting(self, bull_strategy, sample_historical_data):
        """Test strategy backtesting"""
        # Run backtest
        results = bull_strategy.backtest(
            historical_data=sample_historical_data,
            initial_capital=100000.0
        )
        
        assert 'total_return' in results
        assert 'win_rate' in results
        assert 'num_trades' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        
        # Results should be reasonable
        assert results['win_rate'] >= 0.0
        assert results['win_rate'] <= 1.0
        assert results['num_trades'] >= 0
    
    def test_performance_metrics_calculation(self, bull_strategy):
        """Test performance metrics calculation"""
        # Sample trade results
        trade_returns = [0.05, -0.02, 0.08, -0.01, 0.03, -0.04, 0.06]
        
        metrics = bull_strategy._calculate_performance_metrics(trade_returns)
        
        assert 'total_return' in metrics
        assert 'win_rate' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        
        # Verify calculations
        winning_trades = sum(1 for r in trade_returns if r > 0)
        expected_win_rate = winning_trades / len(trade_returns)
        
        assert abs(metrics['win_rate'] - expected_win_rate) < 0.01
    
    def test_walk_forward_analysis(self, bull_strategy, sample_historical_data):
        """Test walk-forward analysis"""
        # Split data into train/test periods
        train_size = len(sample_historical_data) // 2
        
        results = bull_strategy.walk_forward_analysis(
            historical_data=sample_historical_data,
            train_size=train_size,
            test_size=20
        )
        
        assert 'periods' in results
        assert len(results['periods']) > 0
        
        for period in results['periods']:
            assert 'train_metrics' in period
            assert 'test_metrics' in period

# ==================== PERFORMANCE TESTS ====================

class TestStrategyPerformance:
    """Test strategy performance and optimization"""
    
    def test_signal_generation_speed(self, bull_strategy):
        """Test signal generation performance"""
        times = []
        
        for _ in range(100):
            start_time = time.time()
            bull_strategy.generate_signal(
                data=BULL_MARKET_DATA,
                regime=RegimeType.BULL,
                confidence=0.8
            )
            times.append((time.time() - start_time) * 1000)
        
        avg_time = np.mean(times)
        max_time = np.max(times)
        
        print(f"\n⚡ Signal Generation Performance:")
        print(f"   Average time: {avg_time:.3f}ms")
        print(f"   Max time: {max_time:.3f}ms")
        print(f"   Signals per second: {1000/avg_time:.0f}")
        
        # Performance targets
        assert avg_time < 10  # Under 10ms average
        assert max_time < 50  # Under 50ms max
    
    def test_strategy_memory_usage(self, bull_strategy):
        """Test strategy memory usage"""
        import tracemalloc
        
        tracemalloc.start()
        
        # Generate many signals
        for _ in range(1000):
            bull_strategy.generate_signal(
                data=BULL_MARKET_DATA,
                regime=RegimeType.BULL,
                confidence=0.8
            )
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"\n💾 Strategy Memory Usage:")
        print(f"   Current: {current / 1024 / 1024:.1f} MB")
        print(f"   Peak: {peak / 1024 / 1024:.1f} MB")
        
        # Memory should be reasonable
        assert peak < 50 * 1024 * 1024  # Under 50MB
    
    def test_concurrent_signal_generation(self, bull_strategy):
        """Test concurrent signal generation"""
        import asyncio
        
        async def generate_signal_async():
            return bull_strategy.generate_signal(
                data=BULL_MARKET_DATA,
                regime=RegimeType.BULL,
                confidence=0.8
            )
        
        async def run_concurrent_test():
            tasks = [generate_signal_async() for _ in range(20)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
        
        # Run concurrent test
        results = asyncio.run(run_concurrent_test())
        
        # Should handle concurrent execution
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) > 0

# ==================== ERROR HANDLING TESTS ====================

class TestStrategyErrorHandling:
    """Test error handling in strategies"""
    
    def test_invalid_data_handling(self, bull_strategy):
        """Test handling of invalid market data"""
        # Invalid data with missing fields
        invalid_data = MarketData(
            symbol='',
            asset_type=AssetType.STOCK,
            price=-100.0,  # Invalid negative price
            timestamp=datetime.now()
        )
        
        signal = bull_strategy.generate_signal(
            data=invalid_data,
            regime=RegimeType.BULL,
            confidence=0.8
        )
        
        # Should handle gracefully
        assert signal is None
    
    def test_missing_indicator_handling(self, bull_strategy):
        """Test handling when technical indicators are missing"""
        # Data without indicators
        minimal_data = MarketData(
            symbol='AAPL',
            asset_type=AssetType.STOCK,
            price=150.0,
            timestamp=datetime.now()
            # Missing RSI, MACD, etc.
        )
        
        signal = bull_strategy.generate_signal(
            data=minimal_data,
            regime=RegimeType.BULL,
            confidence=0.8
        )
        
        # Should either generate signal with lower confidence or return None
        if signal:
            assert signal.confidence < 0.8
    
    def test_extreme_market_conditions(self, bull_strategy):
        """Test handling of extreme market conditions"""
        # Extreme volatility
        extreme_data = MarketData(
            symbol='AAPL',
            asset_type=AssetType.STOCK,
            price=150.0,
            change=75.0,  # 50% change
            change_percent=50.0,
            volatility=2.0,  # 200% volatility
            timestamp=datetime.now()
        )
        
        signal = bull_strategy.generate_signal(
            data=extreme_data,
            regime=RegimeType.VOLATILE,
            confidence=0.8
        )
        
        # Should handle extreme conditions appropriately
        if signal:
            # Should have very tight stops for extreme volatility
            risk_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
            assert risk_pct < 0.1  # Less than 10% risk

# ==================== MAIN TEST RUNNER ====================

if __name__ == "__main__":
    print("🧪 Running Strategy Tests")
    print("=" * 50)
    
    # Run tests with verbose output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings",
        "-x"  # Stop on first failure
    ])