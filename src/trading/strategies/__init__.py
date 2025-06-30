#!/usr/bin/env python3
"""
File: __init__.py
Path: NeuroCluster-Elite/src/trading/strategies/__init__.py
Description: Trading strategies package initialization

This module initializes all trading strategies including the base framework
and regime-specific implementations for different market conditions.

Author: Your Name
Created: 2025-06-28
Version: 1.0.0
License: MIT
"""

# Import strategy framework
try:
    from .base_strategy import (
        BaseStrategy, TradingSignal, SignalType, StrategyState,
        StrategyMetrics, create_signal
    )
    from .bull_strategy import BullMarketStrategy
    from .bear_strategy import BearMarketStrategy
    
    # Import when available
    # from .volatility_strategy import VolatilityStrategy
    # from .breakout_strategy import BreakoutStrategy  
    # from .range_strategy import RangeTradingStrategy
    # from .crypto_strategies import CryptoMomentumStrategy, CryptoVolatilityStrategy
    
    __all__ = [
        # Base Framework
        'BaseStrategy',
        'TradingSignal', 
        'SignalType',
        'StrategyState',
        'StrategyMetrics',
        'create_signal',
        
        # Market Regime Strategies
        'BullMarketStrategy',
        'BearMarketStrategy',
        
        # Future Strategies
        # 'VolatilityStrategy',
        # 'BreakoutStrategy',
        # 'RangeTradingStrategy',
        # 'CryptoMomentumStrategy',
        # 'CryptoVolatilityStrategy'
    ]
    
except ImportError as e:
    import warnings
    warnings.warn(f"Some strategy components could not be imported: {e}")
    __all__ = []

# Strategy module constants
SUPPORTED_REGIMES = ['BULL', 'BEAR', 'SIDEWAYS', 'VOLATILE', 'BREAKOUT', 'BREAKDOWN']
SUPPORTED_SIGNAL_TYPES = ['BUY', 'SELL', 'HOLD', 'STRONG_BUY', 'STRONG_SELL']
DEFAULT_MIN_CONFIDENCE = 70.0
DEFAULT_MAX_POSITION_SIZE = 0.10
DEFAULT_RISK_PER_TRADE = 0.02

# Strategy categories
MOMENTUM_STRATEGIES = ['BullMarketStrategy', 'BreakoutStrategy']
MEAN_REVERSION_STRATEGIES = ['RangeTradingStrategy', 'VolatilityStrategy']  
DEFENSIVE_STRATEGIES = ['BearMarketStrategy', 'CapitalPreservationStrategy']
CRYPTO_STRATEGIES = ['CryptoMomentumStrategy', 'CryptoVolatilityStrategy']

def get_available_strategies():
    """Get list of available strategy classes"""
    strategies = []
    
    # Get all strategy classes from __all__
    for item_name in __all__:
        try:
            item = globals()[item_name]
            if (isinstance(item, type) and 
                issubclass(item, BaseStrategy) and 
                item != BaseStrategy):
                strategies.append(item_name)
        except (KeyError, TypeError):
            continue
    
    return strategies

def create_strategy_by_name(strategy_name: str, config: dict = None):
    """
    Create strategy instance by name
    
    Args:
        strategy_name: Name of strategy class
        config: Strategy configuration
        
    Returns:
        Strategy instance or None if not found
    """
    try:
        strategy_class = globals().get(strategy_name)
        if strategy_class and issubclass(strategy_class, BaseStrategy):
            return strategy_class(config)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error creating strategy {strategy_name}: {e}")
    
    return None

def get_strategies_info():
    """Get strategies module information"""
    return {
        'available_strategies': len(get_available_strategies()),
        'supported_regimes': len(SUPPORTED_REGIMES),
        'signal_types': len(SUPPORTED_SIGNAL_TYPES),
        'strategy_categories': {
            'momentum': len(MOMENTUM_STRATEGIES),
            'mean_reversion': len(MEAN_REVERSION_STRATEGIES),
            'defensive': len(DEFENSIVE_STRATEGIES),
            'crypto': len(CRYPTO_STRATEGIES)
        },
        'default_min_confidence': f"{DEFAULT_MIN_CONFIDENCE}%",
        'default_position_size': f"{DEFAULT_MAX_POSITION_SIZE:.1%}",
        'default_risk': f"{DEFAULT_RISK_PER_TRADE:.1%}"
    }