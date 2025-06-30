#!/usr/bin/env python3
"""
File: bear_strategy.py
Path: NeuroCluster-Elite/src/trading/strategies/bear_strategy.py
Description: Bear market defensive and short-selling strategy

This strategy is optimized for bear market conditions, focusing on defensive
positioning, short-selling opportunities, and capital preservation.

Strategy Features:
- Breakdown pattern detection
- High volatility exploitation
- Defensive position management
- Short-selling with confirmation
- Risk-off positioning

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

# ==================== BEAR MARKET STRATEGY ====================

class BearMarketStrategy(BaseStrategy):
    """
    Bear Market Defensive Strategy
    
    This strategy is designed for bear market conditions and focuses on:
    1. Defensive positioning (reduce exposure)
    2. Short-selling opportunities with high probability
    3. Capital preservation during market stress
    4. Volatility exploitation
    
    Entry Conditions (Short):
    - Bear market regime with high confidence
    - Breakdown patterns confirmed
    - High volatility with negative momentum
    - RSI in overbought territory (for shorts)
    - Negative sentiment confirmation
    
    Defensive Conditions:
    - Reduce existing long positions
    - Hedge portfolio exposure
    - Increase cash allocation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize bear market strategy"""
        
        # Default configuration for bear market strategy
        default_config = {
            'min_confidence': 80.0,           # Higher confidence for bear signals
            'breakdown_threshold': -2.0,      # Minimum breakdown momentum %
            'volatility_threshold': 25.0,     # High volatility threshold
            'rsi_overbought': 70,            # RSI overbought for shorts
            'rsi_oversold': 30,              # RSI oversold (avoid shorts)
            'max_position_size': 0.08,        # 8% max position (more conservative)
            'risk_per_trade': 0.015,          # 1.5% risk per trade (conservative)
            'target_risk_reward': 2.0,        # 2:1 risk/reward (tighter)
            'sentiment_threshold': -0.2,      # Negative sentiment threshold
            'volume_spike_multiplier': 1.5,   # Volume spike for confirmation
            'enable_short_selling': True,     # Enable short selling
            'defensive_mode': True,           # Enable defensive positioning
            'max_holding_period': 14,         # Shorter holding period (days)
            'correlation_threshold': 0.8,     # High correlation threshold
            'vix_threshold': 25.0             # VIX threshold for fear
        }
        
        # Merge with provided config
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        
        # Strategy-specific state
        self.breakdown_signals = {}  # Track breakdown signals
        self.defensive_positions = {}  # Track defensive positioning
        self.market_stress_level = 0.0  # Current market stress assessment
        
        logger.info("🐻 Bear Market Strategy initialized")
    
    def generate_signal(self, market_data: MarketData, regime: RegimeType, 
                       confidence: float) -> Optional[TradingSignal]:
        """
        Generate bear market signal (short or defensive)
        
        Args:
            market_data: Current market data
            regime: Detected market regime
            confidence: Regime detection confidence
            
        Returns:
            Trading signal or None
        """
        
        # Update market stress assessment
        self._update_market_stress(market_data, regime, confidence)
        
        # Determine signal type based on regime and conditions
        if regime in [RegimeType.BEAR, RegimeType.BREAKDOWN, RegimeType.VOLATILE]:
            
            # Check for short-selling opportunity
            if (self.config['enable_short_selling'] and 
                confidence >= self.config['min_confidence']):
                
                short_signal = self._analyze_short_opportunity(market_data, regime, confidence)
                if short_signal:
                    return short_signal
            
            # Check for defensive positioning
            if self.config['defensive_mode']:
                defensive_signal = self._analyze_defensive_opportunity(market_data, regime, confidence)
                if defensive_signal:
                    return defensive_signal
        
        # Check for position reduction signals in any bearish condition
        if (regime in [RegimeType.BEAR, RegimeType.BREAKDOWN, RegimeType.VOLATILE, RegimeType.DISTRIBUTION] and
            confidence > 70):
            
            reduction_signal = self._analyze_position_reduction(market_data, regime, confidence)
            if reduction_signal:
                return reduction_signal
        
        return None
    
    def _analyze_short_opportunity(self, market_data: MarketData, regime: RegimeType, 
                                  confidence: float) -> Optional[TradingSignal]:
        """Analyze potential short-selling opportunity"""
        
        signal_strength = self._calculate_short_signal_strength(market_data, regime, confidence)
        
        if signal_strength > 0.7:  # High threshold for short selling
            
            # Determine signal type
            if signal_strength >= 0.9:
                signal_type = SignalType.STRONG_SELL
            else:
                signal_type = SignalType.SELL
            
            # Create short signal
            signal = create_signal(
                symbol=market_data.symbol,
                asset_type=market_data.asset_type,
                signal_type=signal_type,
                regime=regime,
                confidence=min(confidence * signal_strength, 95.0),
                entry_price=market_data.price,
                current_price=market_data.price,
                strategy_name=self.strategy_name,
                reasoning=self._generate_short_reasoning(market_data, signal_strength, regime)
            )
            
            # Add technical factors
            self.add_technical_factors(signal, market_data)
            
            # Calculate risk management for short
            signal.stop_loss = self._calculate_short_stop_loss(signal)
            signal.take_profit = self._calculate_short_take_profit(signal)
            
            # Conservative position sizing for shorts
            portfolio_value = 100000  # This would come from portfolio manager
            signal.position_value = self.calculate_position_size(signal, portfolio_value) * 0.7  # More conservative
            signal.position_size = signal.position_value / signal.entry_price
            
            logger.info(f"🐻 Short signal generated: {signal.symbol} {signal.signal_type.value} "
                       f"(confidence: {signal.confidence:.1f}%)")
            
            return signal
        
        return None
    
    def _analyze_defensive_opportunity(self, market_data: MarketData, regime: RegimeType, 
                                     confidence: float) -> Optional[TradingSignal]:
        """Analyze defensive positioning opportunity"""
        
        # Look for hedging opportunities or risk reduction
        if (regime in [RegimeType.BEAR, RegimeType.VOLATILE] and 
            confidence > 75 and 
            self.market_stress_level > 0.6):
            
            # Generate hedge signal
            signal = create_signal(
                symbol=market_data.symbol,
                asset_type=market_data.asset_type,
                signal_type=SignalType.HEDGE,
                regime=regime,
                confidence=confidence,
                entry_price=market_data.price,
                current_price=market_data.price,
                strategy_name=self.strategy_name,
                reasoning=f"Defensive hedge in {regime.value} with {self.market_stress_level:.1%} market stress"
            )
            
            # Smaller hedge positions
            portfolio_value = 100000
            signal.position_value = portfolio_value * 0.05  # 5% hedge
            signal.position_size = signal.position_value / signal.entry_price
            
            return signal
        
        return None
    
    def _analyze_position_reduction(self, market_data: MarketData, regime: RegimeType, 
                                   confidence: float) -> Optional[TradingSignal]:
        """Analyze need for position reduction"""
        
        # Generate position reduction signal when market turns bearish
        if (regime in [RegimeType.BEAR, RegimeType.BREAKDOWN] and 
            confidence > 70):
            
            signal = create_signal(
                symbol=market_data.symbol,
                asset_type=market_data.asset_type,
                signal_type=SignalType.REDUCE,
                regime=regime,
                confidence=confidence,
                entry_price=market_data.price,
                current_price=market_data.price,
                strategy_name=self.strategy_name,
                reasoning=f"Risk reduction due to {regime.value} conditions"
            )
            
            return signal
        
        return None
    
    def _calculate_short_signal_strength(self, market_data: MarketData, regime: RegimeType, 
                                        confidence: float) -> float:
        """Calculate signal strength for short opportunities"""
        
        signal_strength = 0.0
        
        # 1. Breakdown momentum (35% weight)
        breakdown_score = self._analyze_breakdown_momentum(market_data)
        signal_strength += breakdown_score * 0.35
        
        # 2. Volatility and stress (25% weight)
        volatility_score = self._analyze_volatility_stress(market_data)
        signal_strength += volatility_score * 0.25
        
        # 3. Technical indicators (20% weight)
        technical_score = self._analyze_bear_technicals(market_data)
        signal_strength += technical_score * 0.20
        
        # 4. Regime strength (15% weight)
        regime_score = confidence / 100.0
        if regime == RegimeType.BREAKDOWN:
            regime_score *= 1.2  # Breakdown is stronger signal
        signal_strength += regime_score * 0.15
        
        # 5. Sentiment analysis (5% weight)
        sentiment_score = self._analyze_bear_sentiment(market_data)
        signal_strength += sentiment_score * 0.05
        
        return min(signal_strength, 1.0)
    
    def _analyze_breakdown_momentum(self, market_data: MarketData) -> float:
        """Analyze breakdown momentum"""
        
        momentum_score = 0.0
        
        # Strong negative momentum
        if market_data.change_percent < self.config['breakdown_threshold']:
            momentum_score += 0.5
            
            # Bonus for very strong breakdown
            if market_data.change_percent < self.config['breakdown_threshold'] * 2:
                momentum_score += 0.3
        
        # Negative momentum indicator
        if market_data.momentum is not None and market_data.momentum < 0:
            momentum_score += 0.2
        
        return min(momentum_score, 1.0)
    
    def _analyze_volatility_stress(self, market_data: MarketData) -> float:
        """Analyze volatility and market stress"""
        
        volatility_score = 0.0
        
        # High volatility
        if (market_data.volatility is not None and 
            market_data.volatility > self.config['volatility_threshold']):
            volatility_score += 0.6
            
            # Very high volatility bonus
            if market_data.volatility > self.config['volatility_threshold'] * 1.5:
                volatility_score += 0.4
        
        # Volume spike (panic selling)
        if market_data.volume > 0:
            # Simplified volume spike detection
            volatility_score += 0.3
        
        return min(volatility_score, 1.0)
    
    def _analyze_bear_technicals(self, market_data: MarketData) -> float:
        """Analyze technical indicators for bearish signals"""
        
        technical_score = 0.0
        
        # RSI analysis for shorts
        if market_data.rsi is not None:
            if market_data.rsi > self.config['rsi_overbought']:
                technical_score += 0.4  # Overbought - good for shorts
            elif market_data.rsi < self.config['rsi_oversold']:
                technical_score -= 0.3  # Oversold - avoid shorts
        
        # MACD bearish
        if market_data.macd is not None and market_data.macd < 0:
            technical_score += 0.3
        
        # Bollinger Band analysis
        if (market_data.bollinger_upper is not None and 
            market_data.bollinger_lower is not None):
            
            # Price near upper band in bear market = good short setup
            if market_data.price > market_data.bollinger_upper * 0.95:
                technical_score += 0.3
        
        return max(technical_score, 0.0)  # Don't go negative
    
    def _analyze_bear_sentiment(self, market_data: MarketData) -> float:
        """Analyze sentiment for bearish conditions"""
        
        if market_data.sentiment_score is None:
            return 0.5  # Neutral
        
        # Negative sentiment is good for bear strategy
        if market_data.sentiment_score < self.config['sentiment_threshold']:
            return 1.0
        elif market_data.sentiment_score < 0:
            return 0.7
        else:
            return 0.2  # Positive sentiment is bad for bear strategy
    
    def _calculate_short_stop_loss(self, signal: TradingSignal) -> float:
        """Calculate stop loss for short position"""
        
        # For shorts, stop loss is above entry price
        stop_loss_pct = self.config.get('stop_loss_pct', 0.05)
        
        # Adjust for asset type
        if signal.asset_type == AssetType.CRYPTO:
            stop_loss_pct *= 2.0  # Wider stops for crypto
        elif signal.asset_type == AssetType.FOREX:
            stop_loss_pct *= 0.5  # Tighter stops for forex
        
        return signal.entry_price * (1 + stop_loss_pct)
    
    def _calculate_short_take_profit(self, signal: TradingSignal) -> float:
        """Calculate take profit for short position"""
        
        # For shorts, take profit is below entry price
        stop_loss = signal.stop_loss or self._calculate_short_stop_loss(signal)
        risk_amount = stop_loss - signal.entry_price
        
        # Use risk/reward ratio
        reward_amount = risk_amount * self.config['target_risk_reward']
        
        return signal.entry_price - reward_amount
    
    def _update_market_stress(self, market_data: MarketData, regime: RegimeType, confidence: float):
        """Update market stress level assessment"""
        
        stress_factors = []
        
        # Regime stress
        if regime == RegimeType.BEAR:
            stress_factors.append(0.7)
        elif regime == RegimeType.BREAKDOWN:
            stress_factors.append(0.9)
        elif regime == RegimeType.VOLATILE:
            stress_factors.append(0.6)
        
        # Volatility stress
        if market_data.volatility is not None:
            vol_stress = min(market_data.volatility / 50.0, 1.0)  # Normalize
            stress_factors.append(vol_stress)
        
        # Sentiment stress
        if market_data.sentiment_score is not None:
            sentiment_stress = max(-market_data.sentiment_score, 0) / 0.5  # Normalize negative sentiment
            stress_factors.append(min(sentiment_stress, 1.0))
        
        # Calculate average stress
        if stress_factors:
            self.market_stress_level = sum(stress_factors) / len(stress_factors)
        else:
            self.market_stress_level = 0.5  # Default moderate stress
    
    def _generate_short_reasoning(self, market_data: MarketData, signal_strength: float, 
                                 regime: RegimeType) -> str:
        """Generate reasoning for short signal"""
        
        reasons = []
        
        # Regime reason
        if regime == RegimeType.BEAR:
            reasons.append("Bear market confirmed")
        elif regime == RegimeType.BREAKDOWN:
            reasons.append("Breakdown pattern detected")
        elif regime == RegimeType.VOLATILE:
            reasons.append("High volatility environment")
        
        # Momentum reason
        if market_data.change_percent < self.config['breakdown_threshold']:
            reasons.append(f"Strong breakdown ({market_data.change_percent:.1f}%)")
        
        # Technical reasons
        if market_data.rsi is not None and market_data.rsi > self.config['rsi_overbought']:
            reasons.append(f"Overbought RSI ({market_data.rsi:.0f})")
        
        if market_data.volatility is not None and market_data.volatility > self.config['volatility_threshold']:
            reasons.append(f"High volatility ({market_data.volatility:.1f}%)")
        
        # Market stress
        if self.market_stress_level > 0.7:
            reasons.append(f"High market stress ({self.market_stress_level:.1%})")
        
        # Signal strength
        strength_desc = "very strong" if signal_strength > 0.8 else "strong"
        reasons.append(f"{strength_desc} short setup ({signal_strength:.1%})")
        
        return "; ".join(reasons)
    
    def get_strategy_description(self) -> str:
        """Get strategy description"""
        
        return """
        Bear Market Defensive Strategy
        
        This strategy is designed for bear market conditions and market stress periods.
        It focuses on capital preservation, defensive positioning, and selective
        short-selling opportunities.
        
        Key Features:
        • Breakdown pattern detection with volume confirmation
        • High volatility exploitation for short opportunities
        • Defensive hedging and position reduction signals
        • Conservative position sizing and risk management
        • Market stress assessment and adaptation
        
        Signal Types:
        • SHORT/STRONG_SELL: Direct short-selling opportunities
        • HEDGE: Defensive hedging positions
        • REDUCE: Position reduction recommendations
        
        Best Used When:
        • Market regime is Bear, Breakdown, or High Volatility
        • High confidence regime detection (>80%)
        • Technical indicators show weakness
        • Negative sentiment and high market stress
        
        Risk Management:
        • Conservative position sizing (max 8% per position)
        • Wider stop losses for shorts due to volatility
        • Shorter holding periods (max 14 days)
        • Continuous market stress monitoring
        """
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current strategy state"""
        
        return {
            'market_stress_level': self.market_stress_level,
            'breakdown_signals': len(self.breakdown_signals),
            'defensive_positions': len(self.defensive_positions),
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'enable_short_selling': self.config['enable_short_selling'],
            'defensive_mode': self.config['defensive_mode']
        }

# ==================== TESTING ====================

def test_bear_strategy():
    """Test bear market strategy"""
    
    print("🧪 Testing Bear Market Strategy")
    print("=" * 40)
    
    # Create strategy
    strategy = BearMarketStrategy({
        'min_confidence': 80,
        'breakdown_threshold': -2.5,
        'enable_short_selling': True
    })
    
    # Test with bearish market data
    market_data = MarketData(
        symbol='WEAK',
        asset_type=AssetType.STOCK,
        price=45.0,
        change=-3.5,
        change_percent=-7.2,  # Strong breakdown
        volume=3000000,  # High volume
        timestamp=datetime.now(),
        rsi=75.0,  # Overbought (good for short)
        macd=-0.8,  # Bearish MACD
        volatility=35.0,  # High volatility
        sentiment_score=-0.4  # Negative sentiment
    )
    
    # Generate signal
    signal = strategy.generate_signal(market_data, RegimeType.BREAKDOWN, 88.0)
    
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
        strategy.update_strategy_state(signal, True, 180.0)
        
        # Check market stress
        print(f"✅ Market stress level: {strategy.market_stress_level:.1%}")
    else:
        print("❌ No signal generated")
    
    # Test defensive signal
    defensive_data = MarketData(
        symbol='SPY',
        asset_type=AssetType.ETF,
        price=400.0,
        change=-8.0,
        change_percent=-2.0,
        volume=50000000,
        timestamp=datetime.now(),
        volatility=28.0
    )
    
    defensive_signal = strategy.generate_signal(defensive_data, RegimeType.BEAR, 85.0)
    if defensive_signal:
        print(f"✅ Defensive signal: {defensive_signal.signal_type.value}")
    
    print("\n🎉 Bear strategy tests completed!")

if __name__ == "__main__":
    test_bear_strategy()