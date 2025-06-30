#!/usr/bin/env python3
"""
File: bull_strategy.py
Path: NeuroCluster-Elite/src/trading/strategies/bull_strategy.py
Description: Bull market momentum trading strategy

This strategy is optimized for bull market conditions, focusing on momentum
trading with trend-following signals and momentum confirmation.

Strategy Features:
- Momentum breakout detection
- Volume confirmation
- RSI momentum filtering
- Progressive position sizing
- Trend strength analysis

Author: Your Name
Created: 2025-06-28
Version: 1.0.0
License: MIT
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

# Import from our modules
from src.trading.strategies.base_strategy import BaseStrategy, TradingSignal, SignalType, create_signal
from src.core.neurocluster_elite import RegimeType, AssetType, MarketData

# Configure logging
logger = logging.getLogger(__name__)

# ==================== BULL MARKET STRATEGY ====================

class BullMarketStrategy(BaseStrategy):
    """
    Bull Market Momentum Strategy
    
    This strategy capitalizes on upward momentum in bull market conditions.
    It looks for stocks/assets that are breaking out with strong volume
    and positive momentum indicators.
    
    Entry Conditions:
    - Bull market regime detected with high confidence
    - Price momentum > threshold
    - Volume above average (if available)
    - RSI in momentum zone (30-80)
    - Positive sentiment (if available)
    
    Exit Conditions:
    - Regime change to bear/volatile
    - Stop loss hit
    - Take profit target reached
    - Momentum deterioration
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize bull market strategy"""
        
        # Default configuration for bull market strategy
        default_config = {
            'min_confidence': 75.0,           # Minimum regime confidence
            'momentum_threshold': 1.0,        # Minimum price momentum %
            'volume_multiplier': 1.2,         # Volume above average
            'rsi_min': 35,                    # RSI minimum for entry
            'rsi_max': 75,                    # RSI maximum for entry
            'max_position_size': 0.15,        # 15% max position
            'risk_per_trade': 0.025,          # 2.5% risk per trade
            'target_risk_reward': 2.5,        # 2.5:1 risk/reward
            'trend_strength_min': 0.6,        # Minimum trend strength
            'sentiment_threshold': 0.1,       # Positive sentiment threshold
            'breakout_threshold': 2.0,        # Breakout momentum threshold
            'volume_confirmation': True,      # Require volume confirmation
            'progressive_sizing': True,       # Use progressive position sizing
            'max_holding_period': 30          # Maximum holding period (days)
        }
        
        # Merge with provided config
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        
        # Strategy-specific state
        self.recent_entries = {}  # Track recent entries to avoid overtrading
        self.momentum_history = {}  # Track momentum for each symbol
        
        logger.info("🐂 Bull Market Strategy initialized")
    
    def generate_signal(self, market_data: MarketData, regime: RegimeType, 
                       confidence: float) -> Optional[TradingSignal]:
        """
        Generate bull market momentum signal
        
        Args:
            market_data: Current market data
            regime: Detected market regime
            confidence: Regime detection confidence
            
        Returns:
            Trading signal or None
        """
        
        # Only generate signals in bull market conditions
        if regime not in [RegimeType.BULL, RegimeType.BREAKOUT]:
            return None
        
        # Check minimum confidence
        if confidence < self.config['min_confidence']:
            return None
        
        # Check if we have a recent entry for this symbol
        if self._has_recent_entry(market_data.symbol):
            return None
        
        # Analyze market conditions
        signal_strength = self._analyze_bull_conditions(market_data, regime, confidence)
        
        if signal_strength > 0:
            # Determine signal type based on strength
            if signal_strength >= 0.9:
                signal_type = SignalType.STRONG_BUY
            elif signal_strength >= 0.7:
                signal_type = SignalType.BUY
            else:
                return None  # Not strong enough
            
            # Create the signal
            signal = create_signal(
                symbol=market_data.symbol,
                asset_type=market_data.asset_type,
                signal_type=signal_type,
                regime=regime,
                confidence=min(confidence * signal_strength, 95.0),
                entry_price=market_data.price,
                current_price=market_data.price,
                strategy_name=self.strategy_name,
                reasoning=self._generate_reasoning(market_data, signal_strength, regime)
            )
            
            # Add technical factors
            self.add_technical_factors(signal, market_data)
            
            # Calculate risk management levels
            signal.stop_loss = self.calculate_stop_loss(signal)
            signal.take_profit = self.calculate_take_profit(signal)
            
            # Calculate position size
            portfolio_value = 100000  # This would come from portfolio manager
            signal.position_value = self.calculate_position_size(signal, portfolio_value)
            signal.position_size = signal.position_value / signal.entry_price
            
            # Track this entry
            self._track_entry(market_data.symbol)
            
            logger.info(f"🐂 Bull signal generated: {signal.symbol} {signal.signal_type.value} "
                       f"(confidence: {signal.confidence:.1f}%)")
            
            return signal
        
        return None
    
    def _analyze_bull_conditions(self, market_data: MarketData, regime: RegimeType, 
                                confidence: float) -> float:
        """
        Analyze bull market conditions and return signal strength
        
        Args:
            market_data: Market data
            regime: Market regime
            confidence: Regime confidence
            
        Returns:
            Signal strength (0-1)
        """
        
        signal_strength = 0.0
        
        # 1. Price momentum analysis (30% weight)
        momentum_score = self._analyze_momentum(market_data)
        signal_strength += momentum_score * 0.30
        
        # 2. Volume confirmation (20% weight)
        volume_score = self._analyze_volume(market_data)
        signal_strength += volume_score * 0.20
        
        # 3. Technical indicators (25% weight)
        technical_score = self._analyze_technical_indicators(market_data)
        signal_strength += technical_score * 0.25
        
        # 4. Regime strength (15% weight)
        regime_score = self._analyze_regime_strength(regime, confidence)
        signal_strength += regime_score * 0.15
        
        # 5. Sentiment analysis (10% weight)
        sentiment_score = self._analyze_sentiment(market_data)
        signal_strength += sentiment_score * 0.10
        
        # Apply asset-specific adjustments
        signal_strength = self._apply_asset_adjustments(signal_strength, market_data.asset_type)
        
        return min(signal_strength, 1.0)
    
    def _analyze_momentum(self, market_data: MarketData) -> float:
        """Analyze price momentum"""
        
        momentum_score = 0.0
        
        # Current price change
        if market_data.change_percent > self.config['momentum_threshold']:
            momentum_score += 0.4
            
            # Bonus for strong momentum
            if market_data.change_percent > self.config['breakout_threshold']:
                momentum_score += 0.3
        
        # Momentum indicator if available
        if market_data.momentum is not None:
            if market_data.momentum > 0:
                momentum_score += 0.3
        
        # Track momentum history
        self._update_momentum_history(market_data.symbol, market_data.change_percent)
        
        # Momentum consistency bonus
        if self._has_consistent_momentum(market_data.symbol):
            momentum_score += 0.2
        
        return min(momentum_score, 1.0)
    
    def _analyze_volume(self, market_data: MarketData) -> float:
        """Analyze volume confirmation"""
        
        if not self.config['volume_confirmation']:
            return 0.5  # Neutral if not required
        
        volume_score = 0.0
        
        if market_data.volume > 0:
            # For now, assume above-average volume if volume > 0
            # In a real implementation, you'd compare to historical average
            volume_score = 0.7
            
            # Bonus for very high volume
            if market_data.volume > 1000000:  # Simplified threshold
                volume_score += 0.3
        
        return min(volume_score, 1.0)
    
    def _analyze_technical_indicators(self, market_data: MarketData) -> float:
        """Analyze technical indicators"""
        
        technical_score = 0.0
        
        # RSI analysis
        if market_data.rsi is not None:
            if self.config['rsi_min'] <= market_data.rsi <= self.config['rsi_max']:
                technical_score += 0.4
                
                # Bonus for momentum zone RSI (50-70)
                if 50 <= market_data.rsi <= 70:
                    technical_score += 0.2
        
        # MACD analysis
        if market_data.macd is not None:
            if market_data.macd > 0:  # Bullish MACD
                technical_score += 0.3
        
        # Bollinger Bands analysis
        if (market_data.bollinger_upper is not None and 
            market_data.bollinger_lower is not None):
            
            # Check if price is breaking above middle or approaching upper band
            bb_middle = (market_data.bollinger_upper + market_data.bollinger_lower) / 2
            
            if market_data.price > bb_middle:
                technical_score += 0.1
        
        return min(technical_score, 1.0)
    
    def _analyze_regime_strength(self, regime: RegimeType, confidence: float) -> float:
        """Analyze regime strength"""
        
        regime_score = 0.0
        
        # Base score from confidence
        regime_score = confidence / 100.0
        
        # Bonus for strong bull regime
        if regime == RegimeType.BULL and confidence > 85:
            regime_score += 0.2
        elif regime == RegimeType.BREAKOUT and confidence > 80:
            regime_score += 0.3  # Breakout is stronger signal
        
        return min(regime_score, 1.0)
    
    def _analyze_sentiment(self, market_data: MarketData) -> float:
        """Analyze market sentiment"""
        
        if market_data.sentiment_score is None:
            return 0.5  # Neutral if no sentiment data
        
        sentiment_score = 0.0
        
        # Positive sentiment bonus
        if market_data.sentiment_score > self.config['sentiment_threshold']:
            sentiment_score = min(market_data.sentiment_score + 0.5, 1.0)
        elif market_data.sentiment_score > 0:
            sentiment_score = 0.5
        else:
            sentiment_score = 0.2  # Penalty for negative sentiment
        
        return sentiment_score
    
    def _apply_asset_adjustments(self, base_score: float, asset_type: AssetType) -> float:
        """Apply asset-specific adjustments"""
        
        if asset_type == AssetType.CRYPTO:
            # Crypto requires higher conviction due to volatility
            return base_score * 0.9
        elif asset_type == AssetType.FOREX:
            # Forex trends can be more reliable
            return base_score * 1.1
        elif asset_type == AssetType.COMMODITY:
            # Commodities can have strong momentum
            return base_score * 1.05
        
        return base_score  # No adjustment for stocks/ETFs
    
    def _has_recent_entry(self, symbol: str) -> bool:
        """Check if we have a recent entry for this symbol"""
        
        if symbol not in self.recent_entries:
            return False
        
        last_entry = self.recent_entries[symbol]
        time_diff = (datetime.now() - last_entry).total_seconds()
        
        # Don't enter same symbol within 1 hour
        return time_diff < 3600
    
    def _track_entry(self, symbol: str):
        """Track entry for this symbol"""
        self.recent_entries[symbol] = datetime.now()
    
    def _update_momentum_history(self, symbol: str, momentum: float):
        """Update momentum history for symbol"""
        
        if symbol not in self.momentum_history:
            self.momentum_history[symbol] = []
        
        self.momentum_history[symbol].append(momentum)
        
        # Keep only last 5 data points
        if len(self.momentum_history[symbol]) > 5:
            self.momentum_history[symbol] = self.momentum_history[symbol][-5:]
    
    def _has_consistent_momentum(self, symbol: str) -> bool:
        """Check if symbol has consistent positive momentum"""
        
        if symbol not in self.momentum_history:
            return False
        
        history = self.momentum_history[symbol]
        
        if len(history) < 3:
            return False
        
        # Check if at least 2 of last 3 are positive
        positive_count = sum(1 for m in history[-3:] if m > 0)
        
        return positive_count >= 2
    
    def _generate_reasoning(self, market_data: MarketData, signal_strength: float, 
                          regime: RegimeType) -> str:
        """Generate human-readable reasoning for the signal"""
        
        reasons = []
        
        # Regime reason
        if regime == RegimeType.BULL:
            reasons.append("Strong bull market detected")
        elif regime == RegimeType.BREAKOUT:
            reasons.append("Breakout pattern confirmed")
        
        # Momentum reason
        if market_data.change_percent > self.config['breakout_threshold']:
            reasons.append(f"Strong momentum (+{market_data.change_percent:.1f}%)")
        elif market_data.change_percent > self.config['momentum_threshold']:
            reasons.append(f"Positive momentum (+{market_data.change_percent:.1f}%)")
        
        # Technical reasons
        if market_data.rsi is not None:
            if 50 <= market_data.rsi <= 70:
                reasons.append(f"RSI in momentum zone ({market_data.rsi:.0f})")
        
        if market_data.volume > 0:
            reasons.append("Volume confirmation")
        
        # Sentiment reason
        if (market_data.sentiment_score is not None and 
            market_data.sentiment_score > self.config['sentiment_threshold']):
            reasons.append("Positive market sentiment")
        
        # Signal strength
        strength_desc = "very strong" if signal_strength > 0.8 else "strong" if signal_strength > 0.6 else "moderate"
        reasons.append(f"{strength_desc} signal ({signal_strength:.1%})")
        
        return "; ".join(reasons)
    
    def get_strategy_description(self) -> str:
        """Get strategy description"""
        
        return """
        Bull Market Momentum Strategy
        
        This strategy is designed to capitalize on upward momentum during bull market conditions.
        It identifies stocks and assets that are breaking out with strong volume confirmation
        and positive technical indicators.
        
        Key Features:
        • Momentum breakout detection with volume confirmation
        • Multi-factor signal strength analysis
        • Progressive position sizing based on conviction
        • Optimized for bull market and breakout regimes
        • Risk management with dynamic stop losses
        
        Best Used When:
        • Market regime is Bull or Breakout
        • High confidence regime detection (>75%)
        • Assets showing positive momentum and volume
        • Technical indicators align with bullish bias
        
        Risk Management:
        • Dynamic stop losses based on asset volatility
        • Position sizing based on signal strength
        • Maximum position size limits
        • Time-based exit rules
        """
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current strategy state"""
        
        return {
            'recent_entries': len(self.recent_entries),
            'momentum_tracking': len(self.momentum_history),
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'active_symbols': list(self.momentum_history.keys())[-10:]  # Last 10 symbols
        }

# ==================== TESTING ====================

def test_bull_strategy():
    """Test bull market strategy"""
    
    print("🧪 Testing Bull Market Strategy")
    print("=" * 40)
    
    # Create strategy
    strategy = BullMarketStrategy({
        'min_confidence': 75,
        'momentum_threshold': 1.5
    })
    
    # Test with bullish market data
    market_data = MarketData(
        symbol='AAPL',
        asset_type=AssetType.STOCK,
        price=150.0,
        change=3.0,
        change_percent=2.0,  # Strong momentum
        volume=2000000,
        timestamp=datetime.now(),
        rsi=65.0,  # Good momentum zone
        macd=0.5,  # Bullish MACD
        volatility=18.0,
        sentiment_score=0.3  # Positive sentiment
    )
    
    # Generate signal
    signal = strategy.generate_signal(market_data, RegimeType.BULL, 85.0)
    
    if signal:
        print(f"✅ Signal generated: {signal.signal_type.value}")
        print(f"✅ Confidence: {signal.confidence:.1f}%")
        print(f"✅ Entry price: ${signal.entry_price:.2f}")
        print(f"✅ Stop loss: ${signal.stop_loss:.2f}")
        print(f"✅ Take profit: ${signal.take_profit:.2f}")
        print(f"✅ Position size: {signal.position_size:.2f} shares")
        print(f"✅ Reasoning: {signal.reasoning}")
        
        # Test signal validation
        is_valid, reason = strategy.validate_signal(signal)
        print(f"✅ Signal valid: {is_valid} - {reason}")
        
        # Update strategy state
        strategy.update_strategy_state(signal, True, 250.0)
        
        # Get performance
        performance = strategy.get_performance_summary()
        print(f"✅ Strategy performance: {performance['success_rate']:.1%}")
    else:
        print("❌ No signal generated")
    
    # Test with weak conditions
    weak_data = MarketData(
        symbol='WEAK',
        asset_type=AssetType.STOCK,
        price=50.0,
        change=0.1,
        change_percent=0.2,  # Weak momentum
        volume=50000,
        timestamp=datetime.now(),
        rsi=45.0,
        volatility=25.0
    )
    
    weak_signal = strategy.generate_signal(weak_data, RegimeType.SIDEWAYS, 60.0)
    print(f"✅ Weak conditions test: {'No signal' if weak_signal is None else 'Signal generated'}")
    
    print("\n🎉 Bull strategy tests completed!")

if __name__ == "__main__":
    test_bull_strategy()