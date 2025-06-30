#!/usr/bin/env python3
"""
File: __init__.py
Path: NeuroCluster-Elite/src/trading/__init__.py
Description: Trading package initialization

This module initializes the trading components including the trading engine,
risk management, portfolio tracking, and execution systems.

Author: Your Name
Created: 2025-06-28
Version: 1.0.0
License: MIT
"""

# Import main trading components
try:
    from .trading_engine import AdvancedTradingEngine, TradingMode
    from .risk_manager import RiskManager, RiskLevel, RiskMetrics, PositionRisk
    from .portfolio_manager import PortfolioManager, Position, Transaction, PortfolioSnapshot
    from .order_manager import OrderManager, Order, Fill, OrderType, OrderStatus, OrderSide
    
    # Import strategy components
    from .strategies.base_strategy import BaseStrategy, TradingSignal, SignalType, StrategyState
    from .strategies.bull_strategy import BullMarketStrategy
    from .strategies.bear_strategy import BearMarketStrategy
    
    __all__ = [
        # Trading Engine
        'AdvancedTradingEngine',
        'TradingMode',
        
        # Risk Management
        'RiskManager',
        'RiskLevel',
        'RiskMetrics',
        'PositionRisk',
        
        # Portfolio Management
        'PortfolioManager',
        'Position',
        'Transaction',
        'PortfolioSnapshot',
        
        # Order Management
        'OrderManager',
        'Order',
        'Fill',
        'OrderType',
        'OrderStatus',
        'OrderSide',
        
        # Strategy Framework
        'BaseStrategy',
        'TradingSignal',
        'SignalType',
        'StrategyState',
        'BullMarketStrategy',
        'BearMarketStrategy',
    ]
    
except ImportError as e:
    import warnings
    warnings.warn(f"Some trading components could not be imported: {e}")
    __all__ = []

# Trading module constants
DEFAULT_INITIAL_CAPITAL = 100000.0
DEFAULT_MAX_POSITION_SIZE = 0.10  # 10% of portfolio
DEFAULT_RISK_PER_TRADE = 0.02     # 2% risk per trade
DEFAULT_TRANSACTION_FEE = 0.001   # 0.1% transaction fee

# Trading modes
PAPER_TRADING = "paper"
LIVE_TRADING = "live"
BACKTEST_MODE = "backtest"

def get_trading_info():
    """Get trading module information"""
    return {
        'components': len(__all__),
        'default_capital': f"${DEFAULT_INITIAL_CAPITAL:,.0f}",
        'max_position_size': f"{DEFAULT_MAX_POSITION_SIZE:.1%}",
        'risk_per_trade': f"{DEFAULT_RISK_PER_TRADE:.1%}",
        'transaction_fee': f"{DEFAULT_TRANSACTION_FEE:.3%}"
    }