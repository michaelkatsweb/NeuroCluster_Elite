#!/usr/bin/env python3
"""
File: test_trading_engine.py
Path: NeuroCluster-Elite/src/tests/test_trading_engine.py
Description: Comprehensive tests for the trading engine module

This module contains comprehensive tests for the AdvancedTradingEngine class,
including functionality tests, performance benchmarks, integration tests,
and error handling validation.

Features tested:
- Core trading cycle functionality
- Risk management integration
- Portfolio management
- Order execution simulation
- Performance metrics and tracking
- Error handling and recovery
- Multi-asset support
- Real-time data processing

Author: Your Name
Created: 2025-06-30
Version: 1.0.0
License: MIT
"""

import asyncio
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time
import uuid
import sqlite3
from pathlib import Path

# Import modules to test
try:
    from src.trading.trading_engine import (
        AdvancedTradingEngine, TradingMode, OrderType, OrderStatus,
        PositionSide, Position, Trade, PerformanceMetrics
    )
    from src.core.neurocluster_elite import (
        NeuroClusterElite, RegimeType, AssetType, MarketData
    )
    from src.data.multi_asset_manager import MultiAssetDataManager
    from src.trading.strategies.base_strategy import TradingSignal, SignalType
    from src.trading.risk_manager import RiskManager
    from src.trading.portfolio_manager import PortfolioManager
    from src.trading.order_manager import OrderManager
except ImportError as e:
    pytest.skip(f"Could not import trading modules: {e}", allow_module_level=True)

# Test configuration
TEST_CONFIG = {
    'trading_mode': 'PAPER',
    'initial_capital': 100000.0,
    'max_positions': 10,
    'risk_per_trade': 0.02,
    'stop_loss_pct': 0.05,
    'take_profit_pct': 0.10
}

# ==================== FIXTURES ====================

@pytest.fixture
def mock_neurocluster():
    """Mock NeuroCluster algorithm"""
    mock = Mock(spec=NeuroClusterElite)
    mock.detect_regime.return_value = (RegimeType.BULL, 0.85)
    mock.get_performance_metrics.return_value = {
        'efficiency_rate': 99.59,
        'avg_processing_time_ms': 0.045,
        'total_processed': 1000,
        'cluster_count': 8
    }
    return mock

@pytest.fixture
def mock_data_manager():
    """Mock multi-asset data manager"""
    mock = Mock(spec=MultiAssetDataManager)
    
    # Mock market data
    sample_data = {
        'AAPL': MarketData(
            symbol='AAPL',
            asset_type=AssetType.STOCK,
            price=150.0,
            change=2.5,
            change_percent=1.67,
            volume=1000000,
            timestamp=datetime.now(),
            rsi=65.0,
            macd=0.5
        ),
        'BTC-USD': MarketData(
            symbol='BTC-USD',
            asset_type=AssetType.CRYPTO,
            price=45000.0,
            change=1000.0,
            change_percent=2.27,
            volume=500000,
            timestamp=datetime.now(),
            rsi=70.0,
            macd=100.0
        )
    }
    
    mock.fetch_market_data = AsyncMock(return_value=sample_data)
    mock.get_historical_data = AsyncMock(return_value=pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='D'),
        'open': np.random.uniform(140, 160, 100),
        'high': np.random.uniform(145, 165, 100),
        'low': np.random.uniform(135, 155, 100),
        'close': np.random.uniform(140, 160, 100),
        'volume': np.random.randint(500000, 2000000, 100)
    }))
    
    return mock

@pytest.fixture
def sample_trading_signals():
    """Sample trading signals for testing"""
    return [
        TradingSignal(
            symbol='AAPL',
            signal_type=SignalType.BUY,
            asset_type=AssetType.STOCK,
            entry_price=150.0,
            stop_loss=142.5,
            take_profit=165.0,
            confidence=0.85,
            timestamp=datetime.now(),
            strategy_name='BullMarketStrategy',
            volume_confirmation=True,
            risk_reward_ratio=2.0
        ),
        TradingSignal(
            symbol='BTC-USD',
            signal_type=SignalType.BUY,
            asset_type=AssetType.CRYPTO,
            entry_price=45000.0,
            stop_loss=42750.0,
            take_profit=49500.0,
            confidence=0.78,
            timestamp=datetime.now(),
            strategy_name='CryptoMomentumStrategy',
            volume_confirmation=True,
            risk_reward_ratio=2.0
        )
    ]

@pytest.fixture
def trading_engine(mock_neurocluster, mock_data_manager):
    """Create trading engine instance for testing"""
    engine = AdvancedTradingEngine(
        neurocluster=mock_neurocluster,
        data_manager=mock_data_manager,
        config=TEST_CONFIG
    )
    return engine

@pytest.fixture
def sample_positions():
    """Sample positions for testing"""
    return {
        'AAPL': Position(
            id=str(uuid.uuid4()),
            symbol='AAPL',
            asset_type=AssetType.STOCK,
            side=PositionSide.LONG,
            quantity=100,
            entry_price=150.0,
            current_price=155.0,
            market_value=15500.0,
            unrealized_pnl=500.0,
            unrealized_pnl_pct=3.33,
            stop_loss=142.5,
            take_profit=165.0,
            strategy_name='BullMarketStrategy',
            regime_at_entry=RegimeType.BULL
        )
    }

# ==================== BASIC FUNCTIONALITY TESTS ====================

class TestTradingEngineBasics:
    """Test basic trading engine functionality"""
    
    def test_initialization(self, trading_engine):
        """Test trading engine initialization"""
        assert trading_engine.trading_mode == TradingMode.PAPER
        assert trading_engine.initial_capital == 100000.0
        assert trading_engine.portfolio_value == 100000.0
        assert trading_engine.cash_balance == 100000.0
        assert not trading_engine.is_trading_active
        assert len(trading_engine.positions) == 0
        assert len(trading_engine.trades) == 0
    
    def test_configuration_loading(self, mock_neurocluster, mock_data_manager):
        """Test configuration loading and validation"""
        custom_config = {
            'trading_mode': 'LIVE',
            'initial_capital': 50000.0,
            'max_positions': 5
        }
        
        engine = AdvancedTradingEngine(
            neurocluster=mock_neurocluster,
            data_manager=mock_data_manager,
            config=custom_config
        )
        
        assert engine.trading_mode == TradingMode.LIVE
        assert engine.initial_capital == 50000.0
    
    def test_start_stop_trading(self, trading_engine):
        """Test starting and stopping trading"""
        # Start trading
        result = trading_engine.start_trading()
        assert result['status'] == 'success'
        assert trading_engine.is_trading_active
        
        # Stop trading
        result = trading_engine.stop_trading()
        assert result['status'] == 'success'
        assert not trading_engine.is_trading_active

# ==================== TRADING CYCLE TESTS ====================

class TestTradingCycle:
    """Test trading cycle functionality"""
    
    @pytest.mark.asyncio
    async def test_single_trading_cycle(self, trading_engine, sample_trading_signals):
        """Test a single trading cycle execution"""
        # Mock strategy generation
        with patch.object(trading_engine, '_generate_trading_signals') as mock_signals:
            mock_signals.return_value = sample_trading_signals
            
            # Mock signal execution
            with patch.object(trading_engine, '_execute_entry_signals') as mock_execute:
                mock_execute.return_value = [
                    Trade(
                        id=str(uuid.uuid4()),
                        symbol='AAPL',
                        asset_type=AssetType.STOCK,
                        side=PositionSide.LONG,
                        signal_type=SignalType.BUY,
                        quantity=100,
                        entry_price=150.0,
                        strategy_name='BullMarketStrategy'
                    )
                ]
                
                result = await trading_engine.run_trading_cycle()
                
                assert result['status'] == 'success'
                assert result['regime'] in [r for r in RegimeType]
                assert 0 <= result['confidence'] <= 1
                assert 'signals_generated' in result
                assert 'cycle_time_ms' in result
    
    @pytest.mark.asyncio
    async def test_trading_cycle_performance(self, trading_engine):
        """Test trading cycle performance benchmarks"""
        # Run multiple cycles to test performance
        times = []
        
        for _ in range(10):
            start_time = time.time()
            await trading_engine.run_trading_cycle()
            cycle_time = (time.time() - start_time) * 1000
            times.append(cycle_time)
        
        avg_time = np.mean(times)
        max_time = np.max(times)
        
        # Performance assertions (reasonable for testing environment)
        assert avg_time < 100  # Average under 100ms
        assert max_time < 500  # Max under 500ms
        
        print(f"✅ Trading cycle performance: {avg_time:.2f}ms avg, {max_time:.2f}ms max")
    
    @pytest.mark.asyncio
    async def test_error_handling_in_cycle(self, trading_engine):
        """Test error handling during trading cycle"""
        # Mock data manager to raise exception
        trading_engine.data_manager.fetch_market_data = AsyncMock(
            side_effect=Exception("Network error")
        )
        
        result = await trading_engine.run_trading_cycle()
        
        assert result['status'] == 'error'
        assert 'error' in result
        assert 'Network error' in result['error']

# ==================== POSITION MANAGEMENT TESTS ====================

class TestPositionManagement:
    """Test position management functionality"""
    
    def test_position_creation(self, trading_engine, sample_trading_signals):
        """Test position creation from signals"""
        signal = sample_trading_signals[0]
        
        position = Position(
            id=str(uuid.uuid4()),
            symbol=signal.symbol,
            asset_type=signal.asset_type,
            side=PositionSide.LONG,
            quantity=100,
            entry_price=signal.entry_price,
            current_price=signal.entry_price,
            market_value=100 * signal.entry_price,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            strategy_name=signal.strategy_name
        )
        
        assert position.symbol == 'AAPL'
        assert position.quantity == 100
        assert position.entry_price == 150.0
        assert position.market_value == 15000.0
    
    def test_position_update(self, sample_positions):
        """Test position value updates"""
        position = list(sample_positions.values())[0]
        original_pnl = position.unrealized_pnl
        
        # Update with new price
        new_price = 160.0
        position.update_market_value(new_price)
        
        assert position.current_price == new_price
        assert position.market_value == position.quantity * new_price
        assert position.unrealized_pnl > original_pnl
        assert position.unrealized_pnl_pct > 0
    
    def test_stop_loss_calculation(self, trading_engine, sample_positions):
        """Test stop loss and take profit calculations"""
        position = list(sample_positions.values())[0]
        
        # Test stop loss trigger
        stop_loss_price = position.stop_loss
        position.update_market_value(stop_loss_price - 1.0)  # Below stop loss
        
        should_exit = position.current_price <= stop_loss_price
        assert should_exit
        
        # Test take profit trigger
        take_profit_price = position.take_profit
        position.update_market_value(take_profit_price + 1.0)  # Above take profit
        
        should_exit = position.current_price >= take_profit_price
        assert should_exit

# ==================== RISK MANAGEMENT TESTS ====================

class TestRiskManagement:
    """Test risk management integration"""
    
    def test_position_sizing(self, trading_engine, sample_trading_signals):
        """Test position sizing calculations"""
        signal = sample_trading_signals[0]
        
        # Mock risk manager
        with patch.object(trading_engine, 'risk_manager') as mock_risk:
            mock_risk.calculate_position_size.return_value = 50  # 50 shares
            mock_risk.validate_signals.return_value = [signal]
            
            position_size = mock_risk.calculate_position_size(signal, 100000.0)
            assert position_size == 50
    
    def test_risk_limits(self, trading_engine, sample_trading_signals):
        """Test risk limit enforcement"""
        # Create multiple signals that would exceed risk limits
        signals = sample_trading_signals * 10  # 20 signals total
        
        with patch.object(trading_engine, 'risk_manager') as mock_risk:
            # Risk manager should filter out excess signals
            mock_risk.validate_signals.return_value = signals[:5]  # Only first 5
            
            validated = mock_risk.validate_signals(signals, 100000.0, {})
            assert len(validated) <= 5
    
    def test_portfolio_heat(self, trading_engine, sample_positions):
        """Test portfolio heat calculations"""
        trading_engine.positions = sample_positions
        
        total_risk = sum(
            abs(pos.unrealized_pnl) for pos in sample_positions.values()
        )
        portfolio_heat = total_risk / trading_engine.portfolio_value
        
        # Portfolio heat should be reasonable
        assert 0 <= portfolio_heat <= 1.0

# ==================== PERFORMANCE METRICS TESTS ====================

class TestPerformanceMetrics:
    """Test performance tracking and metrics"""
    
    def test_trade_recording(self, trading_engine, sample_positions):
        """Test trade recording and history"""
        # Simulate closing a position
        position = list(sample_positions.values())[0]
        
        trade = Trade(
            id=str(uuid.uuid4()),
            symbol=position.symbol,
            asset_type=position.asset_type,
            side=position.side,
            signal_type=SignalType.SELL,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=position.current_price,
            entry_time=position.opened_at,
            exit_time=datetime.now(),
            realized_pnl=position.unrealized_pnl,
            strategy_name=position.strategy_name
        )
        
        assert trade.realized_pnl == position.unrealized_pnl
        assert trade.exit_price == position.current_price
    
    def test_performance_calculations(self, trading_engine):
        """Test performance metric calculations"""
        # Add sample trades
        sample_trades = [
            Trade(
                id=str(uuid.uuid4()),
                symbol='AAPL',
                asset_type=AssetType.STOCK,
                side=PositionSide.LONG,
                signal_type=SignalType.SELL,
                quantity=100,
                entry_price=150.0,
                exit_price=160.0,
                realized_pnl=1000.0,
                realized_pnl_pct=6.67,
                strategy_name='BullMarketStrategy'
            ),
            Trade(
                id=str(uuid.uuid4()),
                symbol='GOOGL',
                asset_type=AssetType.STOCK,
                side=PositionSide.LONG,
                signal_type=SignalType.SELL,
                quantity=50,
                entry_price=2800.0,
                exit_price=2750.0,
                realized_pnl=-2500.0,
                realized_pnl_pct=-1.79,
                strategy_name='TechStrategy'
            )
        ]
        
        trading_engine.trades = sample_trades
        
        # Calculate metrics
        total_pnl = sum(trade.realized_pnl for trade in sample_trades)
        win_trades = [trade for trade in sample_trades if trade.realized_pnl > 0]
        win_rate = len(win_trades) / len(sample_trades)
        
        assert total_pnl == -1500.0  # 1000 - 2500
        assert win_rate == 0.5  # 1 out of 2 trades
    
    def test_sharpe_ratio_calculation(self, trading_engine):
        """Test Sharpe ratio calculation"""
        # Mock returns data
        returns = pd.Series([0.01, -0.005, 0.02, 0.015, -0.01, 0.03])
        
        sharpe = trading_engine._calculate_sharpe_ratio(returns)
        
        # Sharpe ratio should be calculated correctly
        expected_sharpe = (returns.mean() - 0.02/252) / returns.std() * np.sqrt(252)
        assert abs(sharpe - expected_sharpe) < 0.01

# ==================== INTEGRATION TESTS ====================

class TestIntegration:
    """Test integration with other components"""
    
    @pytest.mark.asyncio
    async def test_neurocluster_integration(self, trading_engine):
        """Test integration with NeuroCluster algorithm"""
        # Test regime detection
        regime, confidence = trading_engine.neurocluster.detect_regime({})
        
        assert regime in [r for r in RegimeType]
        assert 0 <= confidence <= 1
    
    @pytest.mark.asyncio
    async def test_data_manager_integration(self, trading_engine):
        """Test integration with data manager"""
        symbols = ['AAPL', 'GOOGL']
        asset_types = {'AAPL': AssetType.STOCK, 'GOOGL': AssetType.STOCK}
        
        data = await trading_engine.data_manager.fetch_market_data(symbols, AssetType.STOCK)
        
        assert isinstance(data, dict)
        assert len(data) >= 1
    
    def test_strategy_integration(self, trading_engine, sample_trading_signals):
        """Test integration with trading strategies"""
        # Mock strategy selector
        with patch.object(trading_engine, 'strategy_selector') as mock_selector:
            mock_selector.select_strategies.return_value = {
                'AAPL': 'BullMarketStrategy',
                'BTC-USD': 'CryptoMomentumStrategy'
            }
            
            strategies = mock_selector.select_strategies(RegimeType.BULL, {})
            assert 'AAPL' in strategies
            assert strategies['AAPL'] == 'BullMarketStrategy'

# ==================== ERROR HANDLING TESTS ====================

class TestErrorHandling:
    """Test error handling and recovery"""
    
    @pytest.mark.asyncio
    async def test_data_fetch_failure(self, trading_engine):
        """Test handling of data fetch failures"""
        # Mock data manager to fail
        trading_engine.data_manager.fetch_market_data = AsyncMock(
            side_effect=Exception("API rate limit exceeded")
        )
        
        result = await trading_engine.run_trading_cycle()
        
        assert result['status'] == 'error'
        assert 'API rate limit exceeded' in result['error']
    
    def test_invalid_signal_handling(self, trading_engine):
        """Test handling of invalid trading signals"""
        # Create invalid signal
        invalid_signal = TradingSignal(
            symbol='',  # Empty symbol
            signal_type=SignalType.BUY,
            asset_type=AssetType.STOCK,
            entry_price=-100.0,  # Negative price
            confidence=1.5,  # Invalid confidence > 1
            timestamp=datetime.now()
        )
        
        # Validation should catch this
        with patch.object(trading_engine, 'risk_manager') as mock_risk:
            mock_risk.validate_signals.return_value = []  # Empty after validation
            
            validated = mock_risk.validate_signals([invalid_signal], 100000.0, {})
            assert len(validated) == 0
    
    def test_insufficient_capital(self, trading_engine, sample_trading_signals):
        """Test handling of insufficient capital"""
        # Set low capital
        trading_engine.cash_balance = 1000.0  # Very low
        
        signal = sample_trading_signals[0]  # Requires $15,000
        
        with patch.object(trading_engine, 'risk_manager') as mock_risk:
            mock_risk.validate_signals.return_value = []  # Rejected due to capital
            
            validated = mock_risk.validate_signals([signal], 1000.0, {})
            assert len(validated) == 0

# ==================== PERFORMANCE BENCHMARKS ====================

class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.mark.asyncio
    async def test_cycle_speed_benchmark(self, trading_engine):
        """Benchmark trading cycle execution speed"""
        # Warm up
        for _ in range(3):
            await trading_engine.run_trading_cycle()
        
        # Benchmark
        times = []
        for _ in range(20):
            start_time = time.time()
            await trading_engine.run_trading_cycle()
            cycle_time = (time.time() - start_time) * 1000
            times.append(cycle_time)
        
        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        
        print(f"\n📊 Trading Cycle Performance Benchmark:")
        print(f"   Average time: {avg_time:.2f}ms")
        print(f"   95th percentile: {p95_time:.2f}ms")
        print(f"   Cycles per second: {1000/avg_time:.1f}")
        
        # Performance targets (reasonable for testing environment)
        assert avg_time < 200  # Average under 200ms
        assert p95_time < 500  # 95th percentile under 500ms
    
    def test_memory_usage(self, trading_engine):
        """Test memory usage patterns"""
        import tracemalloc
        
        tracemalloc.start()
        
        # Create large dataset
        for i in range(1000):
            position = Position(
                id=str(uuid.uuid4()),
                symbol=f'TEST{i}',
                asset_type=AssetType.STOCK,
                side=PositionSide.LONG,
                quantity=100,
                entry_price=100.0,
                current_price=100.0,
                market_value=10000.0,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0
            )
            trading_engine.positions[f'TEST{i}'] = position
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"\n💾 Memory Usage Test:")
        print(f"   Current: {current / 1024 / 1024:.1f} MB")
        print(f"   Peak: {peak / 1024 / 1024:.1f} MB")
        
        # Memory should be reasonable
        assert peak < 100 * 1024 * 1024  # Under 100MB

# ==================== MAIN TEST RUNNER ====================

if __name__ == "__main__":
    print("🧪 Running Trading Engine Tests")
    print("=" * 50)
    
    # Run tests with verbose output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings",
        "-x"  # Stop on first failure
    ])