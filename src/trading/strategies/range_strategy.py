#!/usr/bin/env python3
"""
File: range_strategy.py
Path: NeuroCluster-Elite/src/trading/strategies/range_strategy.py
Description: Advanced range trading strategy for sideways and consolidating markets

This module implements sophisticated range trading strategies that capitalize on
price oscillations within defined support and resistance levels, optimized for
sideways markets and consolidation patterns.

Features:
- Dynamic support/resistance level identification
- Mean reversion signals within ranges
- Range breakout detection and avoidance
- Volume-based range validation
- Multiple timeframe range analysis
- Adaptive position sizing based on range width
- Risk management with range-specific stops
- Range quality assessment and filtering

Author: Your Name
Created: 2025-06-28
Version: 1.0.0
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import math
from scipy import stats
from scipy.signal import find_peaks, argrelextrema
import warnings

# Import our modules
try:
    from src.core.neurocluster_elite import RegimeType, AssetType, MarketData
    from src.trading.strategies.base_strategy import BaseStrategy, TradingSignal, SignalType, StrategyState, StrategyMetrics
    from src.utils.helpers import calculate_sharpe_ratio, format_percentage
except ImportError:
    # Fallback for testing
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== RANGE STRATEGY ENUMS AND STRUCTURES ====================

class RangeType(Enum):
    """Types of trading ranges"""
    HORIZONTAL = "horizontal"
    ASCENDING = "ascending"  # Ascending triangle
    DESCENDING = "descending"  # Descending triangle
    EXPANDING = "expanding"  # Expanding range
    CONTRACTING = "contracting"  # Contracting range

class RangeQuality(Enum):
    """Quality of identified range"""
    EXCELLENT = "excellent"  # Clear, well-defined range
    GOOD = "good"           # Good range with minor issues
    MODERATE = "moderate"   # Acceptable range
    POOR = "poor"          # Weak range definition
    INVALID = "invalid"    # Not a valid range

class RangePosition(Enum):
    """Position within the range"""
    BOTTOM = "bottom"      # Near support
    LOWER_THIRD = "lower_third"
    MIDDLE = "middle"      # Middle of range
    UPPER_THIRD = "upper_third"
    TOP = "top"           # Near resistance
    OUTSIDE = "outside"   # Outside the range

@dataclass
class TradingRange:
    """Trading range definition"""
    support_level: float
    resistance_level: float
    range_type: RangeType
    quality: RangeQuality
    
    # Range metrics
    range_width: float
    range_width_pct: float
    touches_support: int
    touches_resistance: int
    
    # Time metrics
    start_time: datetime
    end_time: datetime
    duration_bars: int
    
    # Volume analysis
    avg_volume_in_range: float
    volume_at_support: float
    volume_at_resistance: float
    
    # Range validation
    confidence_score: float
    breakout_probability: float
    mean_reversion_strength: float
    
    # Additional metrics
    false_breakout_count: int = 0
    successful_range_trades: int = 0
    range_efficiency: float = 0.0  # How well price respects the range

@dataclass
class RangeSignal:
    """Range trading signal"""
    base_signal: TradingSignal
    trading_range: TradingRange
    range_position: RangePosition
    target_level: float
    range_trade_type: str  # "buy_support", "sell_resistance", "exit"
    expected_range_duration: int  # Expected bars to target
    range_invalidation_level: float
    
    # Range-specific metrics
    distance_to_target_pct: float
    risk_reward_in_range: float
    range_momentum: float

# ==================== RANGE STRATEGY IMPLEMENTATION ====================

class AdvancedRangeStrategy(BaseStrategy):
    """
    Advanced range trading strategy for sideways markets
    
    This strategy identifies high-quality trading ranges and executes mean
    reversion trades by buying near support and selling near resistance,
    while avoiding range breakouts.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize range trading strategy"""
        
        self.config = config or self._get_default_config()
        self.active_ranges = {}
        self.range_history = {}
        self.range_performance = {}
        
        # Strategy state
        self.current_range = None
        self.last_range_update = None
        
        super().__init__()
        logger.info("Advanced Range Trading Strategy initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'range_detection': {
                'min_touches': 3,           # Minimum touches for valid range
                'touch_tolerance': 0.015,   # 1.5% tolerance for level touches
                'min_range_width': 0.02,    # 2% minimum range width
                'max_range_width': 0.15,    # 15% maximum range width
                'min_duration_bars': 15,    # Minimum range duration
                'lookback_period': 100      # Bars to look back for range
            },
            'entry_signals': {
                'support_entry_zone': 0.25,     # Enter in bottom 25% of range
                'resistance_entry_zone': 0.75,   # Enter in top 75% of range
                'momentum_threshold': 0.005,     # 0.5% momentum for confirmation
                'volume_confirmation': 1.2,     # 1.2x average volume
                'rsi_oversold': 30,             # RSI oversold level
                'rsi_overbought': 70            # RSI overbought level
            },
            'range_validation': {
                'min_quality_score': 60.0,     # Minimum range quality
                'max_false_breakouts': 2,      # Max false breakouts allowed
                'volume_consistency': 0.7,     # Volume consistency requirement
                'price_respect_ratio': 0.8     # How often price respects range
            },
            'risk_management': {
                'position_size_base': 0.02,    # 2% base position
                'max_position_size': 0.05,     # 5% max position
                'stop_loss_buffer': 0.005,     # 0.5% buffer beyond range
                'profit_target_pct': 0.8,      # Take 80% of range move
                'max_range_exposure': 0.1      # Max 10% in range trades
            },
            'exit_conditions': {
                'range_break_exit': True,      # Exit on range break
                'time_stop_bars': 50,          # Max bars in trade
                'profit_target_hit': True,     # Exit at profit target
                'momentum_reversal': True      # Exit on momentum reversal
            }
        }
    
    def generate_signal(self, 
                       data: pd.DataFrame, 
                       symbol: str,
                       regime: RegimeType, 
                       confidence: float,
                       additional_data: Dict = None) -> Optional[RangeSignal]:
        """
        Generate range trading signal
        
        Args:
            data: OHLCV data
            symbol: Asset symbol
            regime: Current market regime
            confidence: Regime confidence
            additional_data: Additional market data
            
        Returns:
            RangeSignal or None
        """
        
        try:
            if len(data) < 30:  # Need minimum data for range analysis
                return None
            
            # Only trade in sideways or consolidating regimes
            if regime not in [RegimeType.SIDEWAYS, RegimeType.ACCUMULATION, RegimeType.DISTRIBUTION]:
                return None
            
            # Update or identify current trading range
            current_range = self._identify_trading_range(data, symbol)
            
            if not current_range or current_range.quality == RangeQuality.INVALID:
                return None
            
            # Validate range quality
            if not self._validate_range_quality(current_range, data):
                return None
            
            # Determine current position within range
            current_price = data['close'].iloc[-1]
            range_position = self._calculate_range_position(current_price, current_range)
            
            # Generate range trading signals
            range_signal = self._generate_range_signal(
                data, current_range, range_position, symbol, regime
            )
            
            if range_signal:
                # Validate signal quality
                if self._validate_range_signal(range_signal, data):
                    return range_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating range signal for {symbol}: {e}")
            return None
    
    def _identify_trading_range(self, data: pd.DataFrame, symbol: str) -> Optional[TradingRange]:
        """Identify current trading range"""
        
        try:
            config = self.config['range_detection']
            lookback = config['lookback_period']
            
            # Use recent data for range identification
            recent_data = data.tail(lookback)
            
            if len(recent_data) < config['min_duration_bars']:
                return None
            
            # Find swing highs and lows
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            # Find peaks and troughs
            high_peaks, _ = find_peaks(highs, distance=5)
            low_troughs, _ = find_peaks(-lows, distance=5)
            
            if len(high_peaks) < 2 or len(low_troughs) < 2:
                return None
            
            # Identify potential support and resistance levels
            resistance_candidates = self._find_resistance_levels(highs, high_peaks, recent_data)
            support_candidates = self._find_support_levels(lows, low_troughs, recent_data)
            
            if not resistance_candidates or not support_candidates:
                return None
            
            # Find best range combination
            best_range = self._find_best_range_combination(
                resistance_candidates, support_candidates, recent_data
            )
            
            return best_range
            
        except Exception as e:
            logger.warning(f"Error identifying trading range: {e}")
            return None
    
    def _find_resistance_levels(self, highs: np.ndarray, peaks: np.ndarray, data: pd.DataFrame) -> List[Dict]:
        """Find potential resistance levels"""
        
        try:
            config = self.config['range_detection']
            tolerance = config['touch_tolerance']
            min_touches = config['min_touches']
            
            resistance_levels = []
            
            # Group similar peaks
            peak_prices = highs[peaks]
            peak_indices = peaks
            
            for i, price in enumerate(peak_prices):
                # Find other peaks within tolerance
                similar_peaks = []
                for j, other_price in enumerate(peak_prices):
                    if abs(price - other_price) / price <= tolerance:
                        similar_peaks.append({
                            'price': other_price,
                            'index': peak_indices[j],
                            'timestamp': data.index[peak_indices[j]]
                        })
                
                if len(similar_peaks) >= min_touches:
                    avg_price = np.mean([p['price'] for p in similar_peaks])
                    
                    # Calculate level quality metrics
                    touches = len(similar_peaks)
                    price_consistency = 1.0 - (np.std([p['price'] for p in similar_peaks]) / avg_price)
                    
                    # Calculate volume at level
                    volume_at_level = np.mean([
                        data['volume'].iloc[p['index']] for p in similar_peaks
                        if 'volume' in data.columns
                    ]) if 'volume' in data.columns else 0
                    
                    resistance_levels.append({
                        'price': avg_price,
                        'touches': touches,
                        'consistency': price_consistency,
                        'volume': volume_at_level,
                        'first_touch': min(p['timestamp'] for p in similar_peaks),
                        'last_touch': max(p['timestamp'] for p in similar_peaks),
                        'quality_score': touches * 20 + price_consistency * 30
                    })
            
            # Remove duplicates and sort by quality
            unique_levels = []
            for level in resistance_levels:
                is_duplicate = False
                for existing in unique_levels:
                    if abs(level['price'] - existing['price']) / level['price'] <= tolerance:
                        # Keep the better quality level
                        if level['quality_score'] > existing['quality_score']:
                            unique_levels.remove(existing)
                            unique_levels.append(level)
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_levels.append(level)
            
            return sorted(unique_levels, key=lambda x: x['quality_score'], reverse=True)
            
        except Exception as e:
            logger.warning(f"Error finding resistance levels: {e}")
            return []
    
    def _find_support_levels(self, lows: np.ndarray, troughs: np.ndarray, data: pd.DataFrame) -> List[Dict]:
        """Find potential support levels"""
        
        try:
            config = self.config['range_detection']
            tolerance = config['touch_tolerance']
            min_touches = config['min_touches']
            
            support_levels = []
            
            # Group similar troughs
            trough_prices = lows[troughs]
            trough_indices = troughs
            
            for i, price in enumerate(trough_prices):
                # Find other troughs within tolerance
                similar_troughs = []
                for j, other_price in enumerate(trough_prices):
                    if abs(price - other_price) / price <= tolerance:
                        similar_troughs.append({
                            'price': other_price,
                            'index': trough_indices[j],
                            'timestamp': data.index[trough_indices[j]]
                        })
                
                if len(similar_troughs) >= min_touches:
                    avg_price = np.mean([p['price'] for p in similar_troughs])
                    
                    # Calculate level quality metrics
                    touches = len(similar_troughs)
                    price_consistency = 1.0 - (np.std([p['price'] for p in similar_troughs]) / avg_price)
                    
                    # Calculate volume at level
                    volume_at_level = np.mean([
                        data['volume'].iloc[p['index']] for p in similar_troughs
                        if 'volume' in data.columns
                    ]) if 'volume' in data.columns else 0
                    
                    support_levels.append({
                        'price': avg_price,
                        'touches': touches,
                        'consistency': price_consistency,
                        'volume': volume_at_level,
                        'first_touch': min(p['timestamp'] for p in similar_troughs),
                        'last_touch': max(p['timestamp'] for p in similar_troughs),
                        'quality_score': touches * 20 + price_consistency * 30
                    })
            
            # Remove duplicates and sort by quality
            unique_levels = []
            for level in support_levels:
                is_duplicate = False
                for existing in unique_levels:
                    if abs(level['price'] - existing['price']) / level['price'] <= tolerance:
                        # Keep the better quality level
                        if level['quality_score'] > existing['quality_score']:
                            unique_levels.remove(existing)
                            unique_levels.append(level)
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_levels.append(level)
            
            return sorted(unique_levels, key=lambda x: x['quality_score'], reverse=True)
            
        except Exception as e:
            logger.warning(f"Error finding support levels: {e}")
            return []
    
    def _find_best_range_combination(self, 
                                   resistance_levels: List[Dict],
                                   support_levels: List[Dict],
                                   data: pd.DataFrame) -> Optional[TradingRange]:
        """Find the best support/resistance combination for a trading range"""
        
        try:
            config = self.config['range_detection']
            best_range = None
            best_score = 0.0
            
            # Try combinations of resistance and support levels
            for resistance in resistance_levels[:3]:  # Top 3 resistance levels
                for support in support_levels[:3]:    # Top 3 support levels
                    
                    # Check if it forms a valid range
                    if resistance['price'] <= support['price']:
                        continue
                    
                    range_width = resistance['price'] - support['price']
                    range_width_pct = range_width / support['price']
                    
                    # Check range width constraints
                    if (range_width_pct < config['min_range_width'] or 
                        range_width_pct > config['max_range_width']):
                        continue
                    
                    # Calculate range quality
                    range_quality = self._calculate_range_quality(
                        resistance, support, data, range_width_pct
                    )
                    
                    if range_quality > best_score:
                        best_score = range_quality
                        best_range = self._create_trading_range(
                            resistance, support, data, range_width, range_width_pct, range_quality
                        )
            
            return best_range
            
        except Exception as e:
            logger.warning(f"Error finding best range combination: {e}")
            return None
    
    def _calculate_range_quality(self, 
                               resistance: Dict,
                               support: Dict,
                               data: pd.DataFrame,
                               range_width_pct: float) -> float:
        """Calculate quality score for a potential trading range"""
        
        try:
            score = 0.0
            
            # Level quality (0-40 points)
            level_quality = (resistance['quality_score'] + support['quality_score']) / 2
            score += min(40.0, level_quality)
            
            # Range width appropriateness (0-20 points)
            ideal_width = 0.06  # 6% ideal range width
            width_score = 20.0 * (1 - abs(range_width_pct - ideal_width) / ideal_width)
            score += max(0.0, width_score)
            
            # Touch count (0-20 points)
            total_touches = resistance['touches'] + support['touches']
            touch_score = min(20.0, total_touches * 2)
            score += touch_score
            
            # Price respect within range (0-20 points)
            respect_score = self._calculate_price_respect(resistance, support, data)
            score += respect_score
            
            return min(100.0, score)
            
        except Exception as e:
            logger.warning(f"Error calculating range quality: {e}")
            return 0.0
    
    def _calculate_price_respect(self, resistance: Dict, support: Dict, data: pd.DataFrame) -> float:
        """Calculate how well price respects the range boundaries"""
        
        try:
            resistance_price = resistance['price']
            support_price = support['price']
            
            # Count how often price stays within range
            within_range = 0
            total_bars = 0
            
            for price in data['close']:
                if support_price <= price <= resistance_price:
                    within_range += 1
                total_bars += 1
            
            respect_ratio = within_range / total_bars if total_bars > 0 else 0
            return respect_ratio * 20.0  # 0-20 points
            
        except Exception as e:
            logger.warning(f"Error calculating price respect: {e}")
            return 0.0
    
    def _create_trading_range(self, 
                            resistance: Dict,
                            support: Dict,
                            data: pd.DataFrame,
                            range_width: float,
                            range_width_pct: float,
                            quality_score: float) -> TradingRange:
        """Create TradingRange object"""
        
        try:
            # Determine range type
            range_type = RangeType.HORIZONTAL  # Default
            
            # Calculate additional metrics
            avg_volume = data['volume'].mean() if 'volume' in data.columns else 0
            
            # Determine quality level
            if quality_score >= 80:
                quality = RangeQuality.EXCELLENT
            elif quality_score >= 65:
                quality = RangeQuality.GOOD
            elif quality_score >= 50:
                quality = RangeQuality.MODERATE
            elif quality_score >= 35:
                quality = RangeQuality.POOR
            else:
                quality = RangeQuality.INVALID
            
            trading_range = TradingRange(
                support_level=support['price'],
                resistance_level=resistance['price'],
                range_type=range_type,
                quality=quality,
                range_width=range_width,
                range_width_pct=range_width_pct,
                touches_support=support['touches'],
                touches_resistance=resistance['touches'],
                start_time=min(resistance['first_touch'], support['first_touch']),
                end_time=max(resistance['last_touch'], support['last_touch']),
                duration_bars=len(data),
                avg_volume_in_range=avg_volume,
                volume_at_support=support['volume'],
                volume_at_resistance=resistance['volume'],
                confidence_score=quality_score,
                breakout_probability=self._estimate_breakout_probability(data, resistance, support),
                mean_reversion_strength=self._calculate_mean_reversion_strength(data, resistance, support)
            )
            
            return trading_range
            
        except Exception as e:
            logger.warning(f"Error creating trading range: {e}")
            return None
    
    def _estimate_breakout_probability(self, data: pd.DataFrame, resistance: Dict, support: Dict) -> float:
        """Estimate probability of range breakout"""
        
        try:
            # Simple heuristic based on recent price action and volume
            recent_data = data.tail(10)
            
            if len(recent_data) == 0:
                return 0.5  # Neutral
            
            # Check recent momentum toward boundaries
            recent_closes = recent_data['close']
            resistance_price = resistance['price']
            support_price = support['price']
            range_width = resistance_price - support_price
            
            # Calculate proximity to boundaries
            current_price = recent_closes.iloc[-1]
            distance_to_resistance = (resistance_price - current_price) / range_width
            distance_to_support = (current_price - support_price) / range_width
            
            # Higher probability if approaching boundaries with momentum
            momentum = (recent_closes.iloc[-1] / recent_closes.iloc[0]) - 1
            
            if distance_to_resistance < 0.1 and momentum > 0.01:  # Near resistance with upward momentum
                return 0.7
            elif distance_to_support < 0.1 and momentum < -0.01:  # Near support with downward momentum
                return 0.7
            else:
                return 0.3  # Low breakout probability
                
        except Exception as e:
            logger.warning(f"Error estimating breakout probability: {e}")
            return 0.5
    
    def _calculate_mean_reversion_strength(self, data: pd.DataFrame, resistance: Dict, support: Dict) -> float:
        """Calculate mean reversion strength within the range"""
        
        try:
            resistance_price = resistance['price']
            support_price = support['price']
            range_midpoint = (resistance_price + support_price) / 2
            
            # Calculate how often price reverts to midpoint
            reversions = 0
            total_opportunities = 0
            
            prices = data['close'].values
            for i in range(1, len(prices)):
                prev_price = prices[i-1]
                curr_price = prices[i]
                
                # Check if price was away from midpoint and moved toward it
                prev_distance = abs(prev_price - range_midpoint)
                curr_distance = abs(curr_price - range_midpoint)
                
                if prev_distance > curr_distance:  # Moving toward midpoint
                    reversions += 1
                
                total_opportunities += 1
            
            reversion_ratio = reversions / total_opportunities if total_opportunities > 0 else 0.5
            return reversion_ratio
            
        except Exception as e:
            logger.warning(f"Error calculating mean reversion strength: {e}")
            return 0.5
    
    def _calculate_range_position(self, current_price: float, trading_range: TradingRange) -> RangePosition:
        """Calculate current position within the trading range"""
        
        try:
            support = trading_range.support_level
            resistance = trading_range.resistance_level
            range_width = resistance - support
            
            if current_price < support:
                return RangePosition.OUTSIDE
            elif current_price > resistance:
                return RangePosition.OUTSIDE
            
            # Calculate position within range (0 = support, 1 = resistance)
            position_ratio = (current_price - support) / range_width
            
            if position_ratio <= 0.2:
                return RangePosition.BOTTOM
            elif position_ratio <= 0.4:
                return RangePosition.LOWER_THIRD
            elif position_ratio <= 0.6:
                return RangePosition.MIDDLE
            elif position_ratio <= 0.8:
                return RangePosition.UPPER_THIRD
            else:
                return RangePosition.TOP
                
        except Exception as e:
            logger.warning(f"Error calculating range position: {e}")
            return RangePosition.MIDDLE
    
    def _generate_range_signal(self, 
                             data: pd.DataFrame,
                             trading_range: TradingRange,
                             range_position: RangePosition,
                             symbol: str,
                             regime: RegimeType) -> Optional[RangeSignal]:
        """Generate range trading signal based on position and conditions"""
        
        try:
            current_price = data['close'].iloc[-1]
            config = self.config['entry_signals']
            
            # Check for buy signal near support
            if range_position in [RangePosition.BOTTOM, RangePosition.LOWER_THIRD]:
                
                # Additional confirmation checks
                if self._check_buy_conditions(data, trading_range):
                    
                    # Create buy signal
                    target_price = trading_range.resistance_level * config.get('profit_target_pct', 0.8)
                    stop_loss = trading_range.support_level * (1 - self.config['risk_management']['stop_loss_buffer'])
                    
                    base_signal = TradingSignal(
                        symbol=symbol,
                        asset_type=AssetType.STOCK,  # Default
                        signal_type=SignalType.BUY,
                        regime=regime,
                        confidence=trading_range.confidence_score,
                        entry_price=current_price,
                        current_price=current_price,
                        timestamp=data.index[-1],
                        stop_loss=stop_loss,
                        take_profit=target_price,
                        strategy_name="AdvancedRangeStrategy",
                        reasoning=f"Range buy signal near support at {trading_range.support_level:.2f}"
                    )
                    
                    range_signal = RangeSignal(
                        base_signal=base_signal,
                        trading_range=trading_range,
                        range_position=range_position,
                        target_level=target_price,
                        range_trade_type="buy_support",
                        expected_range_duration=self._estimate_time_to_target(trading_range),
                        range_invalidation_level=stop_loss,
                        distance_to_target_pct=(target_price - current_price) / current_price,
                        risk_reward_in_range=(target_price - current_price) / (current_price - stop_loss),
                        range_momentum=self._calculate_range_momentum(data, trading_range)
                    )
                    
                    return range_signal
            
            # Check for sell signal near resistance
            elif range_position in [RangePosition.TOP, RangePosition.UPPER_THIRD]:
                
                # Additional confirmation checks
                if self._check_sell_conditions(data, trading_range):
                    
                    # Create sell signal
                    target_price = trading_range.support_level * (1 + config.get('profit_target_pct', 0.8) * 
                                                                (1 - trading_range.support_level / trading_range.resistance_level))
                    stop_loss = trading_range.resistance_level * (1 + self.config['risk_management']['stop_loss_buffer'])
                    
                    base_signal = TradingSignal(
                        symbol=symbol,
                        asset_type=AssetType.STOCK,
                        signal_type=SignalType.SELL,
                        regime=regime,
                        confidence=trading_range.confidence_score,
                        entry_price=current_price,
                        current_price=current_price,
                        timestamp=data.index[-1],
                        stop_loss=stop_loss,
                        take_profit=target_price,
                        strategy_name="AdvancedRangeStrategy",
                        reasoning=f"Range sell signal near resistance at {trading_range.resistance_level:.2f}"
                    )
                    
                    range_signal = RangeSignal(
                        base_signal=base_signal,
                        trading_range=trading_range,
                        range_position=range_position,
                        target_level=target_price,
                        range_trade_type="sell_resistance",
                        expected_range_duration=self._estimate_time_to_target(trading_range),
                        range_invalidation_level=stop_loss,
                        distance_to_target_pct=(current_price - target_price) / current_price,
                        risk_reward_in_range=(current_price - target_price) / (stop_loss - current_price),
                        range_momentum=self._calculate_range_momentum(data, trading_range)
                    )
                    
                    return range_signal
            
            return None
            
        except Exception as e:
            logger.warning(f"Error generating range signal: {e}")
            return None
    
    def _check_buy_conditions(self, data: pd.DataFrame, trading_range: TradingRange) -> bool:
        """Check additional conditions for buy signal"""
        
        try:
            config = self.config['entry_signals']
            
            # Check momentum
            if len(data) >= 5:
                recent_momentum = (data['close'].iloc[-1] / data['close'].iloc[-5]) - 1
                if recent_momentum < -config['momentum_threshold']:
                    return False  # Don't buy with strong downward momentum
            
            # Check volume if available
            if 'volume' in data.columns and len(data) >= 10:
                current_volume = data['volume'].iloc[-1]
                avg_volume = data['volume'].tail(10).mean()
                if current_volume < avg_volume * config['volume_confirmation']:
                    return False  # Need volume confirmation
            
            # Check RSI if we can calculate it
            if len(data) >= 14:
                try:
                    # Simple RSI calculation
                    delta = data['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    current_rsi = rsi.iloc[-1]
                    if current_rsi > config['rsi_oversold']:
                        return False  # Wait for oversold condition
                except:
                    pass  # RSI calculation failed, continue without it
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking buy conditions: {e}")
            return False
    
    def _check_sell_conditions(self, data: pd.DataFrame, trading_range: TradingRange) -> bool:
        """Check additional conditions for sell signal"""
        
        try:
            config = self.config['entry_signals']
            
            # Check momentum
            if len(data) >= 5:
                recent_momentum = (data['close'].iloc[-1] / data['close'].iloc[-5]) - 1
                if recent_momentum > config['momentum_threshold']:
                    return False  # Don't sell with strong upward momentum
            
            # Check volume if available
            if 'volume' in data.columns and len(data) >= 10:
                current_volume = data['volume'].iloc[-1]
                avg_volume = data['volume'].tail(10).mean()
                if current_volume < avg_volume * config['volume_confirmation']:
                    return False  # Need volume confirmation
            
            # Check RSI if we can calculate it
            if len(data) >= 14:
                try:
                    # Simple RSI calculation
                    delta = data['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    current_rsi = rsi.iloc[-1]
                    if current_rsi < config['rsi_overbought']:
                        return False  # Wait for overbought condition
                except:
                    pass  # RSI calculation failed, continue without it
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking sell conditions: {e}")
            return False
    
    def _validate_range_quality(self, trading_range: TradingRange, data: pd.DataFrame) -> bool:
        """Validate range meets quality requirements"""
        
        try:
            config = self.config['range_validation']
            
            # Check minimum quality score
            if trading_range.confidence_score < config['min_quality_score']:
                return False
            
            # Check range is not too prone to breakouts
            if trading_range.breakout_probability > 0.8:
                return False
            
            # Check range has good mean reversion characteristics
            if trading_range.mean_reversion_strength < 0.4:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating range quality: {e}")
            return False
    
    def _validate_range_signal(self, signal: RangeSignal, data: pd.DataFrame) -> bool:
        """Validate range signal meets requirements"""
        
        try:
            # Check risk/reward ratio
            if signal.risk_reward_in_range < 1.5:  # Minimum 1.5:1 R/R
                return False
            
            # Check signal confidence
            if signal.base_signal.confidence < 60.0:
                return False
            
            # Check we're not outside the range
            if signal.range_position == RangePosition.OUTSIDE:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating range signal: {e}")
            return False
    
    def _estimate_time_to_target(self, trading_range: TradingRange) -> int:
        """Estimate time to reach target in bars"""
        
        try:
            # Base estimate on historical range efficiency
            base_time = max(5, trading_range.duration_bars // 4)
            
            # Adjust based on range quality
            if trading_range.quality == RangeQuality.EXCELLENT:
                return max(3, base_time // 2)
            elif trading_range.quality == RangeQuality.GOOD:
                return max(5, int(base_time * 0.7))
            else:
                return base_time
                
        except Exception as e:
            logger.warning(f"Error estimating time to target: {e}")
            return 10
    
    def _calculate_range_momentum(self, data: pd.DataFrame, trading_range: TradingRange) -> float:
        """Calculate current momentum within the range"""
        
        try:
            if len(data) < 5:
                return 0.0
            
            # Calculate recent momentum
            momentum = (data['close'].iloc[-1] / data['close'].iloc[-5]) - 1
            
            # Normalize to range scale
            range_width = trading_range.resistance_level - trading_range.support_level
            range_momentum = momentum / (range_width / trading_range.support_level)
            
            return range_momentum
            
        except Exception as e:
            logger.warning(f"Error calculating range momentum: {e}")
            return 0.0
    
    def get_strategy_metrics(self) -> StrategyMetrics:
        """Get strategy performance metrics"""
        
        try:
            # This would be implemented with actual performance tracking
            return StrategyMetrics(
                strategy_name="AdvancedRangeStrategy",
                total_signals=0,
                successful_signals=0,
                failed_signals=0,
                success_rate=0.0,
                avg_confidence=0.0,
                total_pnl=0.0,
                avg_pnl_per_trade=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0
            )
            
        except Exception as e:
            logger.error(f"Error getting strategy metrics: {e}")
            return StrategyMetrics(strategy_name="AdvancedRangeStrategy")

# ==================== TESTING ====================

def test_range_strategy():
    """Test range strategy functionality"""
    
    print("📊 Testing Advanced Range Strategy")
    print("=" * 50)
    
    # Create sample data with range-bound price action
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    
    # Create range-bound data
    support_level = 98
    resistance_level = 102
    range_midpoint = (support_level + resistance_level) / 2
    
    prices = []
    for i in range(100):
        # Oscillate around midpoint with occasional touches of support/resistance
        base_price = range_midpoint + 2 * np.sin(i * 0.2) + np.random.randn() * 0.5
        
        # Ensure touches of support and resistance
        if i % 25 == 0:  # Touch resistance
            base_price = resistance_level + np.random.randn() * 0.2
        elif i % 30 == 0:  # Touch support
            base_price = support_level + np.random.randn() * 0.2
        
        prices.append(max(95, min(105, base_price)))  # Constrain to reasonable range
    
    sample_data = pd.DataFrame({
        'open': prices,
        'close': [p * (1 + np.random.randn() * 0.002) for p in prices],
        'volume': np.random.randint(1000, 5000, 100)
    }, index=dates)
    
    # Add high/low
    sample_data['high'] = sample_data[['open', 'close']].max(axis=1) * (1 + np.random.rand(100) * 0.005)
    sample_data['low'] = sample_data[['open', 'close']].min(axis=1) * (1 - np.random.rand(100) * 0.005)
    
    # Create range strategy
    range_strategy = AdvancedRangeStrategy()
    
    print(f"✅ Strategy initialized:")
    print(f"   Min range width: {range_strategy.config['range_detection']['min_range_width']:.1%}")
    print(f"   Max range width: {range_strategy.config['range_detection']['max_range_width']:.1%}")
    print(f"   Min touches: {range_strategy.config['range_detection']['min_touches']}")
    print(f"   Touch tolerance: {range_strategy.config['range_detection']['touch_tolerance']:.1%}")
    
    # Test range identification
    trading_range = range_strategy._identify_trading_range(sample_data, 'TEST')
    
    if trading_range:
        print(f"\n📈 Identified trading range:")
        print(f"   Support level: ${trading_range.support_level:.2f}")
        print(f"   Resistance level: ${trading_range.resistance_level:.2f}")
        print(f"   Range width: {trading_range.range_width_pct:.2%}")
        print(f"   Range type: {trading_range.range_type.value}")
        print(f"   Quality: {trading_range.quality.value}")
        print(f"   Confidence score: {trading_range.confidence_score:.1f}")
        print(f"   Support touches: {trading_range.touches_support}")
        print(f"   Resistance touches: {trading_range.touches_resistance}")
        print(f"   Mean reversion strength: {trading_range.mean_reversion_strength:.2f}")
        print(f"   Breakout probability: {trading_range.breakout_probability:.2f}")
    else:
        print(f"\n📈 No trading range identified")
    
    # Generate signal
    signal = range_strategy.generate_signal(
        sample_data, 
        'TEST',
        RegimeType.SIDEWAYS, 
        85.0
    )
    
    if signal:
        print(f"\n🚦 Generated range signal:")
        print(f"   Signal type: {signal.base_signal.signal_type.value}")
        print(f"   Trade type: {signal.range_trade_type}")
        print(f"   Range position: {signal.range_position.value}")
        print(f"   Confidence: {signal.base_signal.confidence:.1f}%")
        print(f"   Entry price: ${signal.base_signal.entry_price:.2f}")
        print(f"   Target level: ${signal.target_level:.2f}")
        print(f"   Stop loss: ${signal.base_signal.stop_loss:.2f}")
        print(f"   Risk/Reward: {signal.risk_reward_in_range:.2f}")
        print(f"   Distance to target: {signal.distance_to_target_pct:.2%}")
        print(f"   Expected duration: {signal.expected_range_duration} bars")
        print(f"   Range momentum: {signal.range_momentum:.3f}")
        print(f"   Reasoning: {signal.base_signal.reasoning}")
    else:
        print(f"\n🚦 No range signal generated")
    
    # Test different positions in range
    print(f"\n📍 Testing range position detection:")
    test_prices = [97.5, 99.0, 100.0, 101.0, 102.5]
    
    if trading_range:
        for price in test_prices:
            position = range_strategy._calculate_range_position(price, trading_range)
            print(f"   Price ${price:.2f}: {position.value}")
    
    # Test strategy metrics
    metrics = range_strategy.get_strategy_metrics()
    print(f"\n📊 Strategy metrics:")
    print(f"   Strategy name: {metrics.strategy_name}")
    print(f"   Total signals: {metrics.total_signals}")
    print(f"   Success rate: {metrics.success_rate:.1%}")
    
    print("\n🎉 Range strategy tests completed!")

if __name__ == "__main__":
    test_range_strategy()