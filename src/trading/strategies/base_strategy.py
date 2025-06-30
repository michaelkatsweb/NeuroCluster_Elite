#!/usr/bin/env python3
"""
File: base_strategy.py
Path: NeuroCluster-Elite/src/trading/strategies/base_strategy.py
Description: Base strategy class for all trading strategies

This module defines the base strategy class that all trading strategies inherit from,
providing common functionality and enforcing the strategy interface.

Features:
- Abstract base class for strategy implementation
- Common strategy utilities and helpers
- Performance tracking and metrics
- Risk management integration
- Strategy state management

Author: Your Name
Created: 2025-06-28
Version: 1.0.0
License: MIT
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import numpy as np

# Import from our modules
from src.core.neurocluster_elite import RegimeType, AssetType, MarketData

# Configure logging
logger = logging.getLogger(__name__)

# ==================== ENUMS AND DATA STRUCTURES ====================

class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"
    HEDGE = "HEDGE"
    CLOSE = "CLOSE"
    REDUCE = "REDUCE"

class StrategyState(Enum):
    """Strategy execution states"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    ERROR = "error"

@dataclass
class TradingSignal:
    """Enhanced trading signal structure"""
    symbol: str
    asset_type: AssetType
    signal_type: SignalType
    regime: RegimeType
    confidence: float
    entry_price: float
    current_price: float
    timestamp: datetime
    
    # Position sizing
    position_size: Optional[float] = None
    position_value: Optional[float] = None
    
    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    max_risk_pct: float = 0.02
    
    # Strategy context
    strategy_name: str = ""
    reasoning: str = ""
    technical_factors: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyMetrics:
    """Strategy performance metrics"""
    strategy_name: str
    total_signals: int = 0
    successful_signals: int = 0
    failed_signals: int = 0
    success_rate: float = 0.0
    avg_confidence: float = 0.0
    total_pnl: float = 0.0
    avg_pnl_per_trade: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    last_signal_time: Optional[datetime] = None
    
    def update_metrics(self, signal_success: bool, pnl: float, confidence: float):
        """Update strategy metrics with new trade result"""
        self.total_signals += 1
        
        if signal_success:
            self.successful_signals += 1
        else:
            self.failed_signals += 1
        
        self.success_rate = self.successful_signals / self.total_signals if self.total_signals > 0 else 0
        
        # Update average confidence
        self.avg_confidence = (self.avg_confidence * (self.total_signals - 1) + confidence) / self.total_signals
        
        # Update P&L metrics
        self.total_pnl += pnl
        self.avg_pnl_per_trade = self.total_pnl / self.total_signals if self.total_signals > 0 else 0
        
        self.last_signal_time = datetime.now()

# ==================== BASE STRATEGY CLASS ====================

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    
    This class defines the interface that all trading strategies must implement
    and provides common functionality for strategy execution and management.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize base strategy
        
        Args:
            config: Strategy configuration dictionary
        """
        self.config = config or {}
        self.strategy_name = self.__class__.__name__
        self.state = StrategyState.ACTIVE
        self.metrics = StrategyMetrics(strategy_name=self.strategy_name)
        
        # Strategy parameters
        self.min_confidence = self.config.get('min_confidence', 70.0)
        self.max_position_size = self.config.get('max_position_size', 0.1)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.02)
        
        # Technical analysis settings
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.volatility_threshold = self.config.get('volatility_threshold', 20)
        
        # Strategy state tracking
        self.last_signal_time = None
        self.signal_history: List[TradingSignal] = []
        self.active_positions: Dict[str, Any] = {}
        
        logger.info(f"🎯 {self.strategy_name} initialized")
    
    @abstractmethod
    def generate_signal(self, market_data: MarketData, regime: RegimeType, 
                       confidence: float) -> Optional[TradingSignal]:
        """
        Generate trading signal based on market data and regime
        
        Args:
            market_data: Current market data
            regime: Detected market regime
            confidence: Regime detection confidence
            
        Returns:
            Trading signal or None
        """
        pass
    
    @abstractmethod
    def get_strategy_description(self) -> str:
        """
        Get strategy description
        
        Returns:
            Strategy description string
        """
        pass
    
    def validate_signal(self, signal: TradingSignal) -> Tuple[bool, str]:
        """
        Validate trading signal before execution
        
        Args:
            signal: Trading signal to validate
            
        Returns:
            Tuple of (is_valid, reason)
        """
        
        # Check confidence threshold
        if signal.confidence < self.min_confidence:
            return False, f"Confidence {signal.confidence:.1f}% below threshold {self.min_confidence}%"
        
        # Check if signal is too recent
        if self.last_signal_time:
            time_since_last = (datetime.now() - self.last_signal_time).total_seconds()
            min_interval = self.config.get('min_signal_interval', 30)  # 30 seconds
            
            if time_since_last < min_interval:
                return False, f"Signal too recent (last: {time_since_last:.0f}s ago)"
        
        # Check position size
        if signal.position_size and signal.position_size > self.max_position_size:
            return False, f"Position size {signal.position_size:.3f} exceeds maximum {self.max_position_size:.3f}"
        
        # Asset-specific validation
        if not self._validate_asset_specific(signal):
            return False, "Asset-specific validation failed"
        
        return True, "Signal validated"
    
    def _validate_asset_specific(self, signal: TradingSignal) -> bool:
        """
        Asset-specific signal validation
        
        Args:
            signal: Trading signal
            
        Returns:
            True if valid for asset type
        """
        
        # Cryptocurrency specific validation
        if signal.asset_type == AssetType.CRYPTO:
            # Require higher confidence for crypto due to volatility
            crypto_min_confidence = self.config.get('crypto_min_confidence', 80.0)
            if signal.confidence < crypto_min_confidence:
                return False
        
        # Forex specific validation
        elif signal.asset_type == AssetType.FOREX:
            # Check trading hours for forex
            current_hour = datetime.now().hour
            if not (0 <= current_hour <= 22):  # Simplified forex hours check
                return False
        
        return True
    
    def calculate_position_size(self, signal: TradingSignal, portfolio_value: float) -> float:
        """
        Calculate optimal position size for signal
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            
        Returns:
            Position size in dollars
        """
        
        # Base position size from risk per trade
        risk_amount = portfolio_value * self.risk_per_trade
        
        # Calculate position size based on stop loss
        if signal.stop_loss:
            risk_per_share = abs(signal.entry_price - signal.stop_loss)
            position_size_by_risk = risk_amount / risk_per_share if risk_per_share > 0 else 0
        else:
            # Default to percentage of portfolio
            position_size_by_risk = portfolio_value * 0.05  # 5% default
        
        # Apply confidence scaling
        confidence_multiplier = min(signal.confidence / 100, 1.0)
        adjusted_size = position_size_by_risk * confidence_multiplier
        
        # Apply maximum position size limit
        max_position_value = portfolio_value * self.max_position_size
        final_size = min(adjusted_size, max_position_value)
        
        return final_size
    
    def calculate_stop_loss(self, signal: TradingSignal) -> float:
        """
        Calculate stop loss level for signal
        
        Args:
            signal: Trading signal
            
        Returns:
            Stop loss price
        """
        
        # Default stop loss percentage by asset type
        stop_loss_pcts = {
            AssetType.STOCK: 0.05,     # 5%
            AssetType.CRYPTO: 0.10,    # 10% (higher volatility)
            AssetType.FOREX: 0.02,     # 2% (lower volatility)
            AssetType.COMMODITY: 0.07, # 7%
            AssetType.ETF: 0.04        # 4%
        }
        
        stop_loss_pct = stop_loss_pcts.get(signal.asset_type, 0.05)
        
        # Adjust for volatility if available
        if 'volatility' in signal.technical_factors:
            volatility = signal.technical_factors['volatility']
            if volatility > self.volatility_threshold:
                stop_loss_pct *= 1.5  # Wider stops for high volatility
        
        # Calculate stop loss price
        if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return signal.entry_price * (1 - stop_loss_pct)
        else:  # SELL signals
            return signal.entry_price * (1 + stop_loss_pct)
    
    def calculate_take_profit(self, signal: TradingSignal) -> float:
        """
        Calculate take profit level for signal
        
        Args:
            signal: Trading signal
            
        Returns:
            Take profit price
        """
        
        # Calculate risk amount
        if signal.stop_loss:
            risk_amount = abs(signal.entry_price - signal.stop_loss)
        else:
            risk_amount = signal.entry_price * 0.05  # Default 5%
        
        # Target risk-reward ratio
        target_rr_ratio = self.config.get('target_risk_reward', 2.0)
        reward_amount = risk_amount * target_rr_ratio
        
        # Calculate take profit price
        if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return signal.entry_price + reward_amount
        else:  # SELL signals
            return signal.entry_price - reward_amount
    
    def add_technical_factors(self, signal: TradingSignal, market_data: MarketData):
        """
        Add technical analysis factors to signal
        
        Args:
            signal: Trading signal to enhance
            market_data: Market data with technical indicators
        """
        
        technical_factors = {}
        
        # Add available technical indicators
        if market_data.rsi is not None:
            technical_factors['rsi'] = market_data.rsi
            technical_factors['rsi_condition'] = self._classify_rsi(market_data.rsi)
        
        if market_data.macd is not None:
            technical_factors['macd'] = market_data.macd
            technical_factors['macd_signal'] = 'bullish' if market_data.macd > 0 else 'bearish'
        
        if market_data.volatility is not None:
            technical_factors['volatility'] = market_data.volatility
            technical_factors['volatility_level'] = self._classify_volatility(market_data.volatility)
        
        if market_data.momentum is not None:
            technical_factors['momentum'] = market_data.momentum
            technical_factors['momentum_strength'] = abs(market_data.momentum)
        
        if market_data.sentiment_score is not None:
            technical_factors['sentiment'] = market_data.sentiment_score
            technical_factors['sentiment_bias'] = self._classify_sentiment(market_data.sentiment_score)
        
        signal.technical_factors = technical_factors
    
    def _classify_rsi(self, rsi: float) -> str:
        """Classify RSI value"""
        if rsi <= self.rsi_oversold:
            return 'oversold'
        elif rsi >= self.rsi_overbought:
            return 'overbought'
        else:
            return 'neutral'
    
    def _classify_volatility(self, volatility: float) -> str:
        """Classify volatility level"""
        if volatility < self.volatility_threshold * 0.5:
            return 'low'
        elif volatility > self.volatility_threshold * 1.5:
            return 'high'
        else:
            return 'moderate'
    
    def _classify_sentiment(self, sentiment: float) -> str:
        """Classify sentiment score"""
        if sentiment > 0.3:
            return 'positive'
        elif sentiment < -0.3:
            return 'negative'
        else:
            return 'neutral'
    
    def update_strategy_state(self, signal: TradingSignal, success: bool, pnl: float = 0):
        """
        Update strategy state after signal execution
        
        Args:
            signal: Executed trading signal
            success: Whether signal was successful
            pnl: Profit/loss from signal
        """
        
        # Update metrics
        self.metrics.update_metrics(success, pnl, signal.confidence)
        
        # Store signal in history
        self.signal_history.append(signal)
        
        # Keep limited history
        if len(self.signal_history) > 100:
            self.signal_history = self.signal_history[-100:]
        
        # Update last signal time
        self.last_signal_time = signal.timestamp
        
        logger.debug(f"{self.strategy_name} signal executed: {signal.symbol} {signal.signal_type.value}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get strategy performance summary
        
        Returns:
            Performance summary dictionary
        """
        
        return {
            'strategy_name': self.strategy_name,
            'state': self.state.value,
            'total_signals': self.metrics.total_signals,
            'success_rate': self.metrics.success_rate,
            'avg_confidence': self.metrics.avg_confidence,
            'total_pnl': self.metrics.total_pnl,
            'avg_pnl_per_trade': self.metrics.avg_pnl_per_trade,
            'sharpe_ratio': self.metrics.sharpe_ratio,
            'last_signal_time': self.metrics.last_signal_time.isoformat() if self.metrics.last_signal_time else None,
            'config': self.config
        }
    
    def reset_strategy(self):
        """Reset strategy state and metrics"""
        
        self.metrics = StrategyMetrics(strategy_name=self.strategy_name)
        self.signal_history.clear()
        self.active_positions.clear()
        self.last_signal_time = None
        self.state = StrategyState.ACTIVE
        
        logger.info(f"🔄 {self.strategy_name} reset")
    
    def set_state(self, state: StrategyState):
        """Set strategy state"""
        self.state = state
        logger.info(f"🎯 {self.strategy_name} state changed to {state.value}")
    
    def is_active(self) -> bool:
        """Check if strategy is active"""
        return self.state == StrategyState.ACTIVE
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get comprehensive strategy information"""
        
        return {
            'name': self.strategy_name,
            'description': self.get_strategy_description(),
            'state': self.state.value,
            'performance': self.get_performance_summary(),
            'config': self.config,
            'last_signals': [
                {
                    'symbol': s.symbol,
                    'signal_type': s.signal_type.value,
                    'confidence': s.confidence,
                    'timestamp': s.timestamp.isoformat()
                }
                for s in self.signal_history[-5:]  # Last 5 signals
            ]
        }

# ==================== UTILITY FUNCTIONS ====================

def create_signal(symbol: str, asset_type: AssetType, signal_type: SignalType,
                 regime: RegimeType, confidence: float, entry_price: float,
                 current_price: float, strategy_name: str,
                 reasoning: str = "") -> TradingSignal:
    """
    Utility function to create a trading signal
    
    Args:
        symbol: Trading symbol
        asset_type: Asset type
        signal_type: Signal type
        regime: Market regime
        confidence: Signal confidence
        entry_price: Entry price
        current_price: Current market price
        strategy_name: Name of generating strategy
        reasoning: Signal reasoning
        
    Returns:
        Trading signal
    """
    
    return TradingSignal(
        symbol=symbol,
        asset_type=asset_type,
        signal_type=signal_type,
        regime=regime,
        confidence=confidence,
        entry_price=entry_price,
        current_price=current_price,
        timestamp=datetime.now(),
        strategy_name=strategy_name,
        reasoning=reasoning
    )

# ==================== TESTING ====================

def test_base_strategy():
    """Test base strategy functionality"""
    
    print("🧪 Testing Base Strategy")
    print("=" * 40)
    
    # Create a test strategy implementation
    class TestStrategy(BaseStrategy):
        def generate_signal(self, market_data: MarketData, regime: RegimeType, confidence: float) -> Optional[TradingSignal]:
            if confidence > 80:
                return create_signal(
                    symbol=market_data.symbol,
                    asset_type=market_data.asset_type,
                    signal_type=SignalType.BUY,
                    regime=regime,
                    confidence=confidence,
                    entry_price=market_data.price,
                    current_price=market_data.price,
                    strategy_name=self.strategy_name,
                    reasoning="Test signal"
                )
            return None
        
        def get_strategy_description(self) -> str:
            return "Test strategy for demonstration"
    
    # Test strategy
    strategy = TestStrategy({'min_confidence': 75})
    
    # Create test market data
    market_data = MarketData(
        symbol='TEST',
        asset_type=AssetType.STOCK,
        price=100.0,
        change=1.0,
        change_percent=1.0,
        volume=10000,
        timestamp=datetime.now(),
        rsi=65.0,
        volatility=15.0
    )
    
    # Generate signal
    signal = strategy.generate_signal(market_data, RegimeType.BULL, 85.0)
    
    if signal:
        # Validate signal
        is_valid, reason = strategy.validate_signal(signal)
        print(f"✅ Signal generated: {signal.signal_type.value}")
        print(f"✅ Signal valid: {is_valid} - {reason}")
        
        # Test position sizing
        position_size = strategy.calculate_position_size(signal, 100000)
        print(f"✅ Position size: ${position_size:.2f}")
        
        # Test stop loss calculation
        stop_loss = strategy.calculate_stop_loss(signal)
        print(f"✅ Stop loss: ${stop_loss:.2f}")
        
        # Update strategy state
        strategy.update_strategy_state(signal, True, 100.0)
        
        # Get performance summary
        performance = strategy.get_performance_summary()
        print(f"✅ Performance: {performance['success_rate']:.1%} success rate")
    
    print("\n🎉 Base strategy tests completed!")

if __name__ == "__main__":
    test_base_strategy()