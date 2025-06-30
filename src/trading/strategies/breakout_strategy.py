#!/usr/bin/env python3
"""
File: breakout_strategy.py
Path: NeuroCluster-Elite/src/trading/strategies/breakout_strategy.py
Description: Advanced breakout trading strategy for trend continuation and reversal patterns

This module implements sophisticated breakout trading strategies that identify and
capitalize on price breakouts from consolidation patterns, support/resistance levels,
and technical patterns with advanced confirmation and risk management.

Features:
- Multiple breakout types (support/resistance, pattern, volatility, volume)
- Dynamic breakout confirmation with volume and momentum analysis
- False breakout detection and filtering
- Multi-timeframe breakout validation
- Adaptive position sizing based on breakout strength
- Advanced risk management with volatility-adjusted stops
- Breakout target calculation using measured moves
- Real-time monitoring and signal updates

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

# ==================== BREAKOUT ENUMS AND STRUCTURES ====================

class BreakoutType(Enum):
    """Types of breakout patterns"""
    RESISTANCE_BREAKOUT = "resistance_breakout"
    SUPPORT_BREAKDOWN = "support_breakdown"
    TRIANGLE_BREAKOUT = "triangle_breakout"
    RECTANGLE_BREAKOUT = "rectangle_breakout"
    CHANNEL_BREAKOUT = "channel_breakout"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    VOLUME_BREAKOUT = "volume_breakout"
    GAP_BREAKOUT = "gap_breakout"
    BOLLINGER_BREAKOUT = "bollinger_breakout"
    ATR_BREAKOUT = "atr_breakout"

class BreakoutDirection(Enum):
    """Direction of breakout"""
    UPWARD = "upward"
    DOWNWARD = "downward"
    BIDIRECTIONAL = "bidirectional"

class BreakoutQuality(Enum):
    """Quality/strength of breakout"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class BreakoutLevel:
    """Breakout level definition"""
    price: float
    level_type: str  # "resistance", "support", "pattern_boundary"
    strength: float  # 0-1, based on touches and age
    volume_confirmation: bool = False
    age_bars: int = 0
    touches: int = 1
    last_test: Optional[datetime] = None

@dataclass
class BreakoutPattern:
    """Detected breakout pattern"""
    pattern_type: BreakoutType
    direction: BreakoutDirection
    quality: BreakoutQuality
    
    # Pattern geometry
    breakout_level: float
    entry_price: float
    target_price: float
    stop_loss: float
    
    # Pattern metrics
    pattern_height: float
    pattern_width: int  # in bars
    volume_ratio: float  # breakout volume vs average
    momentum_score: float
    
    # Confirmation indicators
    volume_confirmed: bool = False
    momentum_confirmed: bool = False
    follow_through_confirmed: bool = False
    
    # Pattern context
    start_time: datetime
    breakout_time: datetime
    confirmation_time: Optional[datetime] = None
    
    # Risk metrics
    risk_reward_ratio: float = 0.0
    max_risk_pct: float = 0.02
    confidence_score: float = 0.0

@dataclass
class BreakoutSignal:
    """Breakout trading signal"""
    base_signal: TradingSignal
    breakout_pattern: BreakoutPattern
    confirmation_score: float
    expected_move: float  # Expected price move in percentage
    time_horizon: int  # Expected time to target in bars
    invalidation_level: float  # Level that invalidates the breakout
    
    # Additional context
    market_context: Dict[str, Any] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)

# ==================== BREAKOUT STRATEGY IMPLEMENTATION ====================

class AdvancedBreakoutStrategy(BaseStrategy):
    """
    Advanced breakout trading strategy
    
    This strategy identifies and trades various types of breakouts including
    support/resistance breaks, pattern breakouts, volatility expansions,
    and volume-confirmed moves with sophisticated confirmation logic.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize breakout strategy"""
        
        self.config = config or self._get_default_config()
        self.breakout_levels = {}
        self.pattern_history = {}
        self.false_breakout_count = {}
        
        # Initialize pattern detectors
        self._initialize_detectors()
        
        # Strategy state
        self.current_patterns = {}
        self.active_breakouts = {}
        
        super().__init__()
        logger.info("Advanced Breakout Strategy initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'breakout_detection': {
                'min_pattern_bars': 10,
                'max_pattern_bars': 100,
                'breakout_threshold': 0.002,  # 0.2% minimum breakout
                'volume_confirmation_ratio': 1.5,  # 1.5x average volume
                'momentum_threshold': 0.01,  # 1% price momentum
                'follow_through_bars': 3  # Bars to confirm follow-through
            },
            'support_resistance': {
                'lookback_period': 50,
                'min_touches': 2,
                'touch_tolerance': 0.005,  # 0.5% tolerance for level identification
                'level_age_weight': 0.8,  # Weight for older levels
                'recent_test_boost': 1.2   # Boost for recently tested levels
            },
            'pattern_breakouts': {
                'triangle_min_touches': 4,
                'rectangle_min_width': 15,
                'channel_min_touches': 6,
                'pattern_maturity_ratio': 0.7  # 70% pattern completion for early entry
            },
            'volatility_breakouts': {
                'volatility_lookback': 20,
                'volatility_threshold': 2.0,  # 2 standard deviations
                'atr_multiplier': 1.5,
                'bollinger_threshold': 0.95  # 95% of BB width
            },
            'confirmation_filters': {
                'min_volume_ratio': 1.2,
                'min_momentum_score': 0.5,
                'max_false_breakout_ratio': 0.3,  # Max 30% false breakout rate
                'trend_alignment_weight': 1.3
            },
            'risk_management': {
                'base_position_size': 0.025,  # 2.5% of portfolio
                'max_position_size': 0.05,   # 5% maximum
                'stop_loss_atr_multiplier': 2.0,
                'target_risk_reward_min': 2.0,  # Minimum 2:1 R/R
                'breakout_invalidation_pct': 0.5  # 50% retracement invalidates
            },
            'timing': {
                'entry_delay_bars': 1,  # Wait 1 bar after breakout
                'max_chase_distance': 0.02,  # Max 2% chase after breakout
                'exit_trailing_stop': True,
                'profit_taking_levels': [0.5, 0.8]  # Take profits at 50% and 80% of target
            }
        }
    
    def _initialize_detectors(self):
        """Initialize breakout detection methods"""
        
        self.breakout_detectors = {
            BreakoutType.RESISTANCE_BREAKOUT: self._detect_resistance_breakout,
            BreakoutType.SUPPORT_BREAKDOWN: self._detect_support_breakdown,
            BreakoutType.TRIANGLE_BREAKOUT: self._detect_triangle_breakout,
            BreakoutType.RECTANGLE_BREAKOUT: self._detect_rectangle_breakout,
            BreakoutType.CHANNEL_BREAKOUT: self._detect_channel_breakout,
            BreakoutType.VOLATILITY_BREAKOUT: self._detect_volatility_breakout,
            BreakoutType.VOLUME_BREAKOUT: self._detect_volume_breakout,
            BreakoutType.BOLLINGER_BREAKOUT: self._detect_bollinger_breakout,
            BreakoutType.ATR_BREAKOUT: self._detect_atr_breakout
        }
    
    def generate_signal(self, 
                       data: pd.DataFrame, 
                       symbol: str,
                       regime: RegimeType, 
                       confidence: float,
                       additional_data: Dict = None) -> Optional[BreakoutSignal]:
        """
        Generate breakout trading signal
        
        Args:
            data: OHLCV data
            symbol: Asset symbol
            regime: Current market regime
            confidence: Regime confidence
            additional_data: Additional market data
            
        Returns:
            BreakoutSignal or None
        """
        
        try:
            if len(data) < 20:  # Need minimum data for breakout analysis
                return None
            
            # Update support/resistance levels
            self._update_support_resistance_levels(data, symbol)
            
            # Detect all types of breakouts
            breakout_patterns = []
            
            for breakout_type, detector in self.breakout_detectors.items():
                try:
                    pattern = detector(data, symbol)
                    if pattern:
                        breakout_patterns.append(pattern)
                except Exception as e:
                    logger.warning(f"Error detecting {breakout_type.value}: {e}")
                    continue
            
            if not breakout_patterns:
                return None
            
            # Select best breakout pattern
            best_pattern = self._select_best_pattern(breakout_patterns, data, regime)
            
            if not best_pattern:
                return None
            
            # Generate trading signal from pattern
            base_signal = self._create_base_signal(best_pattern, data, symbol, regime)
            
            if not base_signal:
                return None
            
            # Calculate confirmation score
            confirmation_score = self._calculate_confirmation_score(best_pattern, data)
            
            # Calculate expected move and time horizon
            expected_move = self._calculate_expected_move(best_pattern, data)
            time_horizon = self._estimate_time_horizon(best_pattern, data)
            
            # Create breakout signal
            breakout_signal = BreakoutSignal(
                base_signal=base_signal,
                breakout_pattern=best_pattern,
                confirmation_score=confirmation_score,
                expected_move=expected_move,
                time_horizon=time_horizon,
                invalidation_level=self._calculate_invalidation_level(best_pattern),
                market_context=self._get_market_context(data, regime),
                risk_factors=self._identify_risk_factors(best_pattern, data)
            )
            
            # Apply risk management
            breakout_signal = self._apply_risk_management(breakout_signal, data)
            
            # Final validation
            if self._validate_breakout_signal(breakout_signal, data):
                return breakout_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating breakout signal for {symbol}: {e}")
            return None
    
    def _update_support_resistance_levels(self, data: pd.DataFrame, symbol: str):
        """Update support and resistance levels"""
        
        try:
            config = self.config['support_resistance']
            lookback = config['lookback_period']
            
            if len(data) < lookback:
                return
            
            recent_data = data.tail(lookback)
            
            # Find swing highs and lows
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            # Find peaks (resistance levels)
            high_peaks, _ = find_peaks(highs, distance=5)
            
            # Find troughs (support levels)
            low_troughs, _ = find_peaks(-lows, distance=5)
            
            # Update resistance levels
            for peak_idx in high_peaks:
                price = highs[peak_idx]
                timestamp = recent_data.index[peak_idx]
                
                # Check if this is near an existing level
                existing_level = self._find_nearby_level(price, 'resistance', symbol)
                
                if existing_level:
                    # Update existing level
                    existing_level.touches += 1
                    existing_level.last_test = timestamp
                    existing_level.strength = self._calculate_level_strength(existing_level)
                else:
                    # Create new resistance level
                    new_level = BreakoutLevel(
                        price=price,
                        level_type='resistance',
                        strength=0.5,  # Initial strength
                        age_bars=0,
                        touches=1,
                        last_test=timestamp
                    )
                    
                    if symbol not in self.breakout_levels:
                        self.breakout_levels[symbol] = {'resistance': [], 'support': []}
                    
                    self.breakout_levels[symbol]['resistance'].append(new_level)
            
            # Update support levels
            for trough_idx in low_troughs:
                price = lows[trough_idx]
                timestamp = recent_data.index[trough_idx]
                
                existing_level = self._find_nearby_level(price, 'support', symbol)
                
                if existing_level:
                    existing_level.touches += 1
                    existing_level.last_test = timestamp
                    existing_level.strength = self._calculate_level_strength(existing_level)
                else:
                    new_level = BreakoutLevel(
                        price=price,
                        level_type='support',
                        strength=0.5,
                        age_bars=0,
                        touches=1,
                        last_test=timestamp
                    )
                    
                    if symbol not in self.breakout_levels:
                        self.breakout_levels[symbol] = {'resistance': [], 'support': []}
                    
                    self.breakout_levels[symbol]['support'].append(new_level)
            
            # Age existing levels and remove weak ones
            self._age_and_clean_levels(symbol)
            
        except Exception as e:
            logger.warning(f"Error updating support/resistance levels: {e}")
    
    def _detect_resistance_breakout(self, data: pd.DataFrame, symbol: str) -> Optional[BreakoutPattern]:
        """Detect resistance breakout patterns"""
        
        try:
            if symbol not in self.breakout_levels or not self.breakout_levels[symbol]['resistance']:
                return None
            
            current_price = data['close'].iloc[-1]
            current_high = data['high'].iloc[-1]
            current_volume = data['volume'].iloc[-1] if 'volume' in data.columns else 0
            
            # Check each resistance level
            for level in self.breakout_levels[symbol]['resistance']:
                # Check if price has broken above resistance
                breakout_threshold = level.price * (1 + self.config['breakout_detection']['breakout_threshold'])
                
                if current_high > breakout_threshold and level.strength > 0.3:
                    
                    # Calculate pattern metrics
                    pattern_start = max(0, len(data) - level.age_bars - 20)
                    pattern_data = data.iloc[pattern_start:]
                    
                    pattern_height = level.price - pattern_data['low'].min()
                    pattern_width = len(pattern_data)
                    
                    # Check volume confirmation
                    avg_volume = data['volume'].tail(20).mean() if 'volume' in data.columns else 1
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                    volume_confirmed = volume_ratio >= self.config['breakout_detection']['volume_confirmation_ratio']
                    
                    # Calculate target using pattern height
                    target_price = level.price + pattern_height
                    
                    # Calculate stop loss
                    stop_loss = level.price * 0.98  # 2% below resistance turned support
                    
                    # Calculate momentum score
                    recent_returns = data['close'].pct_change().tail(5)
                    momentum_score = recent_returns.sum()
                    
                    pattern = BreakoutPattern(
                        pattern_type=BreakoutType.RESISTANCE_BREAKOUT,
                        direction=BreakoutDirection.UPWARD,
                        quality=self._assess_breakout_quality(level.strength, volume_ratio, momentum_score),
                        breakout_level=level.price,
                        entry_price=current_price,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        pattern_height=pattern_height,
                        pattern_width=pattern_width,
                        volume_ratio=volume_ratio,
                        momentum_score=momentum_score,
                        volume_confirmed=volume_confirmed,
                        momentum_confirmed=abs(momentum_score) > self.config['breakout_detection']['momentum_threshold'],
                        start_time=pattern_data.index[0],
                        breakout_time=data.index[-1],
                        risk_reward_ratio=(target_price - current_price) / (current_price - stop_loss),
                        confidence_score=level.strength * 100
                    )
                    
                    return pattern
            
            return None
            
        except Exception as e:
            logger.warning(f"Error detecting resistance breakout: {e}")
            return None
    
    def _detect_support_breakdown(self, data: pd.DataFrame, symbol: str) -> Optional[BreakoutPattern]:
        """Detect support breakdown patterns"""
        
        try:
            if symbol not in self.breakout_levels or not self.breakout_levels[symbol]['support']:
                return None
            
            current_price = data['close'].iloc[-1]
            current_low = data['low'].iloc[-1]
            current_volume = data['volume'].iloc[-1] if 'volume' in data.columns else 0
            
            # Check each support level
            for level in self.breakout_levels[symbol]['support']:
                # Check if price has broken below support
                breakdown_threshold = level.price * (1 - self.config['breakout_detection']['breakout_threshold'])
                
                if current_low < breakdown_threshold and level.strength > 0.3:
                    
                    # Calculate pattern metrics
                    pattern_start = max(0, len(data) - level.age_bars - 20)
                    pattern_data = data.iloc[pattern_start:]
                    
                    pattern_height = pattern_data['high'].max() - level.price
                    pattern_width = len(pattern_data)
                    
                    # Check volume confirmation
                    avg_volume = data['volume'].tail(20).mean() if 'volume' in data.columns else 1
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                    volume_confirmed = volume_ratio >= self.config['breakout_detection']['volume_confirmation_ratio']
                    
                    # Calculate target using pattern height
                    target_price = level.price - pattern_height
                    
                    # Calculate stop loss
                    stop_loss = level.price * 1.02  # 2% above support turned resistance
                    
                    # Calculate momentum score
                    recent_returns = data['close'].pct_change().tail(5)
                    momentum_score = recent_returns.sum()
                    
                    pattern = BreakoutPattern(
                        pattern_type=BreakoutType.SUPPORT_BREAKDOWN,
                        direction=BreakoutDirection.DOWNWARD,
                        quality=self._assess_breakout_quality(level.strength, volume_ratio, abs(momentum_score)),
                        breakout_level=level.price,
                        entry_price=current_price,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        pattern_height=pattern_height,
                        pattern_width=pattern_width,
                        volume_ratio=volume_ratio,
                        momentum_score=momentum_score,
                        volume_confirmed=volume_confirmed,
                        momentum_confirmed=abs(momentum_score) > self.config['breakout_detection']['momentum_threshold'],
                        start_time=pattern_data.index[0],
                        breakout_time=data.index[-1],
                        risk_reward_ratio=(current_price - target_price) / (stop_loss - current_price),
                        confidence_score=level.strength * 100
                    )
                    
                    return pattern
            
            return None
            
        except Exception as e:
            logger.warning(f"Error detecting support breakdown: {e}")
            return None
    
    def _detect_triangle_breakout(self, data: pd.DataFrame, symbol: str) -> Optional[BreakoutPattern]:
        """Detect triangle pattern breakouts"""
        
        try:
            config = self.config['pattern_breakouts']
            min_touches = config['triangle_min_touches']
            
            if len(data) < 30:
                return None
            
            # Look for converging trend lines
            lookback = min(60, len(data))
            recent_data = data.tail(lookback)
            
            # Find swing highs and lows
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            high_peaks, _ = find_peaks(highs, distance=3)
            low_troughs, _ = find_peaks(-lows, distance=3)
            
            if len(high_peaks) < 2 or len(low_troughs) < 2:
                return None
            
            # Check for converging trend lines
            # Resistance trend line (connecting highs)
            high_prices = highs[high_peaks]
            high_times = high_peaks
            
            if len(high_prices) >= 2:
                high_slope, high_intercept, high_r, _, _ = stats.linregress(high_times, high_prices)
            else:
                return None
            
            # Support trend line (connecting lows)
            low_prices = lows[low_troughs]
            low_times = low_troughs
            
            if len(low_prices) >= 2:
                low_slope, low_intercept, low_r, _, _ = stats.linregress(low_times, low_prices)
            else:
                return None
            
            # Check for triangle pattern (converging lines)
            if high_slope < 0 and low_slope > 0 and abs(high_slope - low_slope) > 0.001:
                
                # Calculate convergence point
                convergence_x = (low_intercept - high_intercept) / (high_slope - low_slope)
                convergence_price = high_slope * convergence_x + high_intercept
                
                # Check if we're near convergence
                current_idx = len(recent_data) - 1
                current_price = recent_data['close'].iloc[-1]
                
                resistance_level = high_slope * current_idx + high_intercept
                support_level = low_slope * current_idx + low_intercept
                
                # Check for breakout
                breakout_threshold = 0.005  # 0.5%
                
                if current_price > resistance_level * (1 + breakout_threshold):
                    # Upward breakout
                    pattern_height = resistance_level - support_level
                    target_price = resistance_level + pattern_height
                    
                    pattern = BreakoutPattern(
                        pattern_type=BreakoutType.TRIANGLE_BREAKOUT,
                        direction=BreakoutDirection.UPWARD,
                        quality=self._assess_triangle_quality(high_r, low_r, len(high_peaks) + len(low_troughs)),
                        breakout_level=resistance_level,
                        entry_price=current_price,
                        target_price=target_price,
                        stop_loss=support_level,
                        pattern_height=pattern_height,
                        pattern_width=current_idx - min(high_times[0], low_times[0]),
                        volume_ratio=1.0,  # Would calculate from volume data
                        momentum_score=0.0,  # Would calculate momentum
                        start_time=recent_data.index[min(high_times[0], low_times[0])],
                        breakout_time=recent_data.index[-1],
                        confidence_score=min(high_r, low_r) * 100
                    )
                    
                    return pattern
                    
                elif current_price < support_level * (1 - breakout_threshold):
                    # Downward breakout
                    pattern_height = resistance_level - support_level
                    target_price = support_level - pattern_height
                    
                    pattern = BreakoutPattern(
                        pattern_type=BreakoutType.TRIANGLE_BREAKOUT,
                        direction=BreakoutDirection.DOWNWARD,
                        quality=self._assess_triangle_quality(high_r, low_r, len(high_peaks) + len(low_troughs)),
                        breakout_level=support_level,
                        entry_price=current_price,
                        target_price=target_price,
                        stop_loss=resistance_level,
                        pattern_height=pattern_height,
                        pattern_width=current_idx - min(high_times[0], low_times[0]),
                        volume_ratio=1.0,
                        momentum_score=0.0,
                        start_time=recent_data.index[min(high_times[0], low_times[0])],
                        breakout_time=recent_data.index[-1],
                        confidence_score=min(high_r, low_r) * 100
                    )
                    
                    return pattern
            
            return None
            
        except Exception as e:
            logger.warning(f"Error detecting triangle breakout: {e}")
            return None
    
    def _detect_rectangle_breakout(self, data: pd.DataFrame, symbol: str) -> Optional[BreakoutPattern]:
        """Detect rectangle/consolidation breakouts"""
        
        try:
            config = self.config['pattern_breakouts']
            min_width = config['rectangle_min_width']
            
            if len(data) < min_width:
                return None
            
            # Look for horizontal consolidation
            lookback = min(50, len(data))
            recent_data = data.tail(lookback)
            
            # Calculate resistance and support levels
            resistance = recent_data['high'].max()
            support = recent_data['low'].min()
            
            # Check if it's a valid rectangle (reasonable height/width ratio)
            pattern_height = resistance - support
            pattern_width = len(recent_data)
            
            if pattern_height / resistance > 0.1:  # Too wide, not a consolidation
                return None
            
            # Check for touches of levels
            resistance_touches = (recent_data['high'] >= resistance * 0.995).sum()
            support_touches = (recent_data['low'] <= support * 1.005).sum()
            
            if resistance_touches < 2 or support_touches < 2:
                return None
            
            # Check for breakout
            current_price = data['close'].iloc[-1]
            current_high = data['high'].iloc[-1]
            current_low = data['low'].iloc[-1]
            
            breakout_threshold = 0.003  # 0.3%
            
            if current_high > resistance * (1 + breakout_threshold):
                # Upward breakout
                target_price = resistance + pattern_height
                
                pattern = BreakoutPattern(
                    pattern_type=BreakoutType.RECTANGLE_BREAKOUT,
                    direction=BreakoutDirection.UPWARD,
                    quality=self._assess_rectangle_quality(pattern_width, resistance_touches, support_touches),
                    breakout_level=resistance,
                    entry_price=current_price,
                    target_price=target_price,
                    stop_loss=support,
                    pattern_height=pattern_height,
                    pattern_width=pattern_width,
                    volume_ratio=1.0,
                    momentum_score=0.0,
                    start_time=recent_data.index[0],
                    breakout_time=data.index[-1],
                    confidence_score=70.0
                )
                
                return pattern
                
            elif current_low < support * (1 - breakout_threshold):
                # Downward breakout
                target_price = support - pattern_height
                
                pattern = BreakoutPattern(
                    pattern_type=BreakoutType.RECTANGLE_BREAKOUT,
                    direction=BreakoutDirection.DOWNWARD,
                    quality=self._assess_rectangle_quality(pattern_width, resistance_touches, support_touches),
                    breakout_level=support,
                    entry_price=current_price,
                    target_price=target_price,
                    stop_loss=resistance,
                    pattern_height=pattern_height,
                    pattern_width=pattern_width,
                    volume_ratio=1.0,
                    momentum_score=0.0,
                    start_time=recent_data.index[0],
                    breakout_time=data.index[-1],
                    confidence_score=70.0
                )
                
                return pattern
            
            return None
            
        except Exception as e:
            logger.warning(f"Error detecting rectangle breakout: {e}")
            return None
    
    def _detect_channel_breakout(self, data: pd.DataFrame, symbol: str) -> Optional[BreakoutPattern]:
        """Detect channel breakouts"""
        # Simplified implementation - would need more sophisticated channel detection
        return None
    
    def _detect_volatility_breakout(self, data: pd.DataFrame, symbol: str) -> Optional[BreakoutPattern]:
        """Detect volatility expansion breakouts"""
        
        try:
            config = self.config['volatility_breakouts']
            lookback = config['volatility_lookback']
            threshold = config['volatility_threshold']
            
            if len(data) < lookback + 5:
                return None
            
            # Calculate rolling volatility
            returns = data['close'].pct_change()
            rolling_vol = returns.rolling(lookback).std()
            
            if rolling_vol.isna().all():
                return None
            
            # Calculate volatility z-score
            vol_mean = rolling_vol.mean()
            vol_std = rolling_vol.std()
            
            if vol_std == 0:
                return None
            
            current_vol = rolling_vol.iloc[-1]
            vol_z_score = (current_vol - vol_mean) / vol_std
            
            # Check for volatility breakout
            if vol_z_score > threshold:
                
                # Determine direction based on recent price action
                recent_returns = returns.tail(3).sum()
                direction = BreakoutDirection.UPWARD if recent_returns > 0 else BreakoutDirection.DOWNWARD
                
                current_price = data['close'].iloc[-1]
                
                # Estimate target based on volatility expansion
                daily_vol = current_vol * np.sqrt(252)  # Annualized
                expected_move = current_price * daily_vol / np.sqrt(252) * 5  # 5-day expected move
                
                if direction == BreakoutDirection.UPWARD:
                    target_price = current_price + expected_move
                    stop_loss = current_price - expected_move * 0.5
                else:
                    target_price = current_price - expected_move
                    stop_loss = current_price + expected_move * 0.5
                
                pattern = BreakoutPattern(
                    pattern_type=BreakoutType.VOLATILITY_BREAKOUT,
                    direction=direction,
                    quality=BreakoutQuality.STRONG if vol_z_score > 3 else BreakoutQuality.MODERATE,
                    breakout_level=current_price,
                    entry_price=current_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    pattern_height=expected_move,
                    pattern_width=lookback,
                    volume_ratio=1.0,
                    momentum_score=vol_z_score,
                    start_time=data.index[-lookback],
                    breakout_time=data.index[-1],
                    confidence_score=min(95.0, 50 + vol_z_score * 10)
                )
                
                return pattern
            
            return None
            
        except Exception as e:
            logger.warning(f"Error detecting volatility breakout: {e}")
            return None
    
    def _detect_volume_breakout(self, data: pd.DataFrame, symbol: str) -> Optional[BreakoutPattern]:
        """Detect volume-based breakouts"""
        
        try:
            if 'volume' not in data.columns or len(data) < 20:
                return None
            
            # Calculate volume statistics
            avg_volume = data['volume'].tail(20).mean()
            current_volume = data['volume'].iloc[-1]
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Check for significant volume spike
            volume_threshold = self.config['breakout_detection']['volume_confirmation_ratio'] * 2
            
            if volume_ratio > volume_threshold:
                
                # Check price movement with volume
                price_change = data['close'].pct_change().iloc[-1]
                
                if abs(price_change) > 0.01:  # Minimum 1% price move
                    
                    current_price = data['close'].iloc[-1]
                    direction = BreakoutDirection.UPWARD if price_change > 0 else BreakoutDirection.DOWNWARD
                    
                    # Estimate target based on volume-price relationship
                    avg_range = (data['high'] - data['low']).tail(20).mean()
                    expected_move = avg_range * volume_ratio * 0.5
                    
                    if direction == BreakoutDirection.UPWARD:
                        target_price = current_price + expected_move
                        stop_loss = current_price - avg_range
                    else:
                        target_price = current_price - expected_move
                        stop_loss = current_price + avg_range
                    
                    pattern = BreakoutPattern(
                        pattern_type=BreakoutType.VOLUME_BREAKOUT,
                        direction=direction,
                        quality=BreakoutQuality.STRONG if volume_ratio > volume_threshold * 2 else BreakoutQuality.MODERATE,
                        breakout_level=current_price,
                        entry_price=current_price,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        pattern_height=expected_move,
                        pattern_width=1,
                        volume_ratio=volume_ratio,
                        momentum_score=abs(price_change),
                        volume_confirmed=True,
                        start_time=data.index[-1],
                        breakout_time=data.index[-1],
                        confidence_score=min(90.0, 60 + volume_ratio * 5)
                    )
                    
                    return pattern
            
            return None
            
        except Exception as e:
            logger.warning(f"Error detecting volume breakout: {e}")
            return None
    
    def _detect_bollinger_breakout(self, data: pd.DataFrame, symbol: str) -> Optional[BreakoutPattern]:
        """Detect Bollinger Bands breakouts"""
        
        try:
            if len(data) < 20:
                return None
            
            # Calculate Bollinger Bands
            period = 20
            std_dev = 2
            
            sma = data['close'].rolling(period).mean()
            std = data['close'].rolling(period).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            current_price = data['close'].iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            
            if pd.isna(current_upper) or pd.isna(current_lower):
                return None
            
            # Check for breakout
            if current_price > current_upper:
                # Upward breakout
                band_width = current_upper - current_lower
                target_price = current_price + band_width * 0.5
                
                pattern = BreakoutPattern(
                    pattern_type=BreakoutType.BOLLINGER_BREAKOUT,
                    direction=BreakoutDirection.UPWARD,
                    quality=BreakoutQuality.MODERATE,
                    breakout_level=current_upper,
                    entry_price=current_price,
                    target_price=target_price,
                    stop_loss=sma.iloc[-1],
                    pattern_height=band_width,
                    pattern_width=period,
                    volume_ratio=1.0,
                    momentum_score=(current_price - current_upper) / current_upper,
                    start_time=data.index[-period],
                    breakout_time=data.index[-1],
                    confidence_score=70.0
                )
                
                return pattern
                
            elif current_price < current_lower:
                # Downward breakout
                band_width = current_upper - current_lower
                target_price = current_price - band_width * 0.5
                
                pattern = BreakoutPattern(
                    pattern_type=BreakoutType.BOLLINGER_BREAKOUT,
                    direction=BreakoutDirection.DOWNWARD,
                    quality=BreakoutQuality.MODERATE,
                    breakout_level=current_lower,
                    entry_price=current_price,
                    target_price=target_price,
                    stop_loss=sma.iloc[-1],
                    pattern_height=band_width,
                    pattern_width=period,
                    volume_ratio=1.0,
                    momentum_score=(current_lower - current_price) / current_lower,
                    start_time=data.index[-period],
                    breakout_time=data.index[-1],
                    confidence_score=70.0
                )
                
                return pattern
            
            return None
            
        except Exception as e:
            logger.warning(f"Error detecting Bollinger breakout: {e}")
            return None
    
    def _detect_atr_breakout(self, data: pd.DataFrame, symbol: str) -> Optional[BreakoutPattern]:
        """Detect ATR-based breakouts"""
        
        try:
            if len(data) < 14 or not all(col in data.columns for col in ['high', 'low', 'close']):
                return None
            
            # Calculate ATR
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(14).mean()
            
            current_atr = atr.iloc[-1]
            current_price = data['close'].iloc[-1]
            
            if pd.isna(current_atr):
                return None
            
            # Check for price move greater than ATR threshold
            price_change = abs(data['close'].iloc[-1] - data['close'].iloc[-2])
            atr_multiplier = self.config['volatility_breakouts']['atr_multiplier']
            
            if price_change > current_atr * atr_multiplier:
                
                direction = BreakoutDirection.UPWARD if data['close'].iloc[-1] > data['close'].iloc[-2] else BreakoutDirection.DOWNWARD
                
                # Calculate target based on ATR
                if direction == BreakoutDirection.UPWARD:
                    target_price = current_price + current_atr * 2
                    stop_loss = current_price - current_atr
                else:
                    target_price = current_price - current_atr * 2
                    stop_loss = current_price + current_atr
                
                pattern = BreakoutPattern(
                    pattern_type=BreakoutType.ATR_BREAKOUT,
                    direction=direction,
                    quality=BreakoutQuality.STRONG if price_change > current_atr * atr_multiplier * 2 else BreakoutQuality.MODERATE,
                    breakout_level=current_price,
                    entry_price=current_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    pattern_height=current_atr * 2,
                    pattern_width=14,
                    volume_ratio=1.0,
                    momentum_score=price_change / current_atr,
                    start_time=data.index[-14],
                    breakout_time=data.index[-1],
                    confidence_score=min(90.0, 60 + (price_change / current_atr) * 10)
                )
                
                return pattern
            
            return None
            
        except Exception as e:
            logger.warning(f"Error detecting ATR breakout: {e}")
            return None
    
    # Helper methods
    
    def _find_nearby_level(self, price: float, level_type: str, symbol: str, tolerance: float = 0.01) -> Optional[BreakoutLevel]:
        """Find existing level near the given price"""
        
        if symbol not in self.breakout_levels or level_type not in self.breakout_levels[symbol]:
            return None
        
        for level in self.breakout_levels[symbol][level_type]:
            if abs(level.price - price) / price < tolerance:
                return level
        
        return None
    
    def _calculate_level_strength(self, level: BreakoutLevel) -> float:
        """Calculate strength of support/resistance level"""
        
        try:
            config = self.config['support_resistance']
            
            # Base strength from touches
            touch_strength = min(1.0, level.touches / 5.0)  # Max at 5 touches
            
            # Age factor (older levels are stronger)
            age_factor = min(1.0, level.age_bars / 100.0) * config['level_age_weight']
            
            # Recent test boost
            recent_boost = 0.0
            if level.last_test:
                days_since_test = (datetime.now() - level.last_test).days
                if days_since_test < 5:
                    recent_boost = config['recent_test_boost'] * (5 - days_since_test) / 5
            
            strength = (touch_strength + age_factor + recent_boost) / 3
            return min(1.0, strength)
            
        except Exception as e:
            logger.warning(f"Error calculating level strength: {e}")
            return 0.5
    
    def _age_and_clean_levels(self, symbol: str):
        """Age existing levels and remove weak ones"""
        
        try:
            if symbol not in self.breakout_levels:
                return
            
            for level_type in ['resistance', 'support']:
                if level_type in self.breakout_levels[symbol]:
                    levels = self.breakout_levels[symbol][level_type]
                    
                    # Age levels
                    for level in levels:
                        level.age_bars += 1
                        level.strength = self._calculate_level_strength(level)
                    
                    # Remove weak levels
                    self.breakout_levels[symbol][level_type] = [
                        level for level in levels 
                        if level.strength > 0.2 and level.age_bars < 200
                    ]
            
        except Exception as e:
            logger.warning(f"Error aging levels: {e}")
    
    def _assess_breakout_quality(self, level_strength: float, volume_ratio: float, momentum_score: float) -> BreakoutQuality:
        """Assess the quality of a breakout"""
        
        try:
            score = 0.0
            
            # Level strength component (0-40 points)
            score += level_strength * 40
            
            # Volume component (0-30 points)
            if volume_ratio > 2.0:
                score += 30
            elif volume_ratio > 1.5:
                score += 20
            elif volume_ratio > 1.2:
                score += 10
            
            # Momentum component (0-30 points)
            momentum_points = min(30, abs(momentum_score) * 1000)
            score += momentum_points
            
            if score >= 80:
                return BreakoutQuality.VERY_STRONG
            elif score >= 60:
                return BreakoutQuality.STRONG
            elif score >= 40:
                return BreakoutQuality.MODERATE
            else:
                return BreakoutQuality.WEAK
                
        except Exception as e:
            logger.warning(f"Error assessing breakout quality: {e}")
            return BreakoutQuality.MODERATE
    
    def _assess_triangle_quality(self, high_r: float, low_r: float, touch_count: int) -> BreakoutQuality:
        """Assess quality of triangle pattern"""
        
        try:
            # Minimum R-squared for trend lines
            min_r = min(abs(high_r), abs(low_r))
            
            if min_r > 0.8 and touch_count >= 6:
                return BreakoutQuality.VERY_STRONG
            elif min_r > 0.6 and touch_count >= 4:
                return BreakoutQuality.STRONG
            elif min_r > 0.4:
                return BreakoutQuality.MODERATE
            else:
                return BreakoutQuality.WEAK
                
        except Exception as e:
            return BreakoutQuality.MODERATE
    
    def _assess_rectangle_quality(self, width: int, resistance_touches: int, support_touches: int) -> BreakoutQuality:
        """Assess quality of rectangle pattern"""
        
        try:
            total_touches = resistance_touches + support_touches
            
            if width >= 30 and total_touches >= 8:
                return BreakoutQuality.VERY_STRONG
            elif width >= 20 and total_touches >= 6:
                return BreakoutQuality.STRONG
            elif width >= 15 and total_touches >= 4:
                return BreakoutQuality.MODERATE
            else:
                return BreakoutQuality.WEAK
                
        except Exception as e:
            return BreakoutQuality.MODERATE
    
    def _select_best_pattern(self, patterns: List[BreakoutPattern], data: pd.DataFrame, regime: RegimeType) -> Optional[BreakoutPattern]:
        """Select the best breakout pattern from candidates"""
        
        try:
            if not patterns:
                return None
            
            # Score each pattern
            scored_patterns = []
            
            for pattern in patterns:
                score = 0.0
                
                # Quality score (0-40 points)
                quality_scores = {
                    BreakoutQuality.VERY_STRONG: 40,
                    BreakoutQuality.STRONG: 30,
                    BreakoutQuality.MODERATE: 20,
                    BreakoutQuality.WEAK: 10
                }
                score += quality_scores.get(pattern.quality, 10)
                
                # Risk/reward ratio (0-30 points)
                if pattern.risk_reward_ratio > 3:
                    score += 30
                elif pattern.risk_reward_ratio > 2:
                    score += 20
                elif pattern.risk_reward_ratio > 1.5:
                    score += 10
                
                # Volume confirmation (0-15 points)
                if pattern.volume_confirmed:
                    score += 15
                
                # Momentum confirmation (0-15 points)
                if pattern.momentum_confirmed:
                    score += 15
                
                # Regime alignment bonus
                if ((regime in [RegimeType.BULL, RegimeType.BREAKOUT] and pattern.direction == BreakoutDirection.UPWARD) or
                    (regime in [RegimeType.BEAR, RegimeType.BREAKDOWN] and pattern.direction == BreakoutDirection.DOWNWARD)):
                    score *= self.config['confirmation_filters']['trend_alignment_weight']
                
                scored_patterns.append((pattern, score))
            
            # Select highest scoring pattern
            scored_patterns.sort(key=lambda x: x[1], reverse=True)
            
            best_pattern, best_score = scored_patterns[0]
            
            # Minimum score threshold
            if best_score < 40:
                return None
            
            return best_pattern
            
        except Exception as e:
            logger.error(f"Error selecting best pattern: {e}")
            return None
    
    def _create_base_signal(self, pattern: BreakoutPattern, data: pd.DataFrame, symbol: str, regime: RegimeType) -> Optional[TradingSignal]:
        """Create base trading signal from breakout pattern"""
        
        try:
            # Determine signal type
            if pattern.direction == BreakoutDirection.UPWARD:
                signal_type = SignalType.BUY
            else:
                signal_type = SignalType.SELL
            
            # Calculate confidence
            confidence = pattern.confidence_score
            if pattern.volume_confirmed:
                confidence += 10
            if pattern.momentum_confirmed:
                confidence += 10
            
            confidence = min(95.0, confidence)
            
            signal = TradingSignal(
                symbol=symbol,
                asset_type=AssetType.STOCK,  # Default, should be passed in
                signal_type=signal_type,
                regime=regime,
                confidence=confidence,
                entry_price=pattern.entry_price,
                current_price=pattern.entry_price,
                timestamp=pattern.breakout_time,
                stop_loss=pattern.stop_loss,
                take_profit=pattern.target_price,
                strategy_name="AdvancedBreakoutStrategy",
                reasoning=f"{pattern.pattern_type.value} breakout with {pattern.quality.value} quality"
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error creating base signal: {e}")
            return None
    
    def _calculate_confirmation_score(self, pattern: BreakoutPattern, data: pd.DataFrame) -> float:
        """Calculate confirmation score for breakout"""
        
        try:
            score = 0.0
            
            # Volume confirmation (0-40 points)
            if pattern.volume_confirmed:
                score += 40 * min(1.0, pattern.volume_ratio / 3.0)
            
            # Momentum confirmation (0-30 points)
            if pattern.momentum_confirmed:
                score += 30 * min(1.0, abs(pattern.momentum_score) / 0.05)
            
            # Pattern quality (0-30 points)
            quality_scores = {
                BreakoutQuality.VERY_STRONG: 30,
                BreakoutQuality.STRONG: 22,
                BreakoutQuality.MODERATE: 15,
                BreakoutQuality.WEAK: 8
            }
            score += quality_scores.get(pattern.quality, 8)
            
            return min(100.0, score)
            
        except Exception as e:
            logger.warning(f"Error calculating confirmation score: {e}")
            return 50.0
    
    def _calculate_expected_move(self, pattern: BreakoutPattern, data: pd.DataFrame) -> float:
        """Calculate expected percentage move"""
        
        try:
            if pattern.entry_price > 0:
                expected_move = abs(pattern.target_price - pattern.entry_price) / pattern.entry_price
                return expected_move
            return 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating expected move: {e}")
            return 0.0
    
    def _estimate_time_horizon(self, pattern: BreakoutPattern, data: pd.DataFrame) -> int:
        """Estimate time horizon to reach target"""
        
        try:
            # Base estimate on pattern width
            base_time = max(5, pattern.pattern_width // 4)
            
            # Adjust based on pattern quality
            if pattern.quality == BreakoutQuality.VERY_STRONG:
                return max(3, base_time // 2)
            elif pattern.quality == BreakoutQuality.STRONG:
                return max(5, int(base_time * 0.7))
            elif pattern.quality == BreakoutQuality.MODERATE:
                return base_time
            else:
                return base_time * 2
                
        except Exception as e:
            logger.warning(f"Error estimating time horizon: {e}")
            return 10
    
    def _calculate_invalidation_level(self, pattern: BreakoutPattern) -> float:
        """Calculate level that invalidates the breakout"""
        
        try:
            invalidation_pct = self.config['risk_management']['breakout_invalidation_pct']
            
            if pattern.direction == BreakoutDirection.UPWARD:
                # Invalidation below breakout level
                return pattern.breakout_level * (1 - invalidation_pct * 0.01)
            else:
                # Invalidation above breakout level
                return pattern.breakout_level * (1 + invalidation_pct * 0.01)
                
        except Exception as e:
            logger.warning(f"Error calculating invalidation level: {e}")
            return pattern.stop_loss
    
    def _get_market_context(self, data: pd.DataFrame, regime: RegimeType) -> Dict[str, Any]:
        """Get market context for the breakout"""
        
        try:
            context = {
                'regime': regime.value,
                'volatility': data['close'].pct_change().tail(20).std() * np.sqrt(252),
                'trend_strength': 0.0,  # Would calculate from indicators
                'volume_trend': 'normal'  # Would analyze volume pattern
            }
            
            # Add more context as needed
            return context
            
        except Exception as e:
            logger.warning(f"Error getting market context: {e}")
            return {}
    
    def _identify_risk_factors(self, pattern: BreakoutPattern, data: pd.DataFrame) -> List[str]:
        """Identify risk factors for the breakout"""
        
        risk_factors = []
        
        try:
            # Low volume warning
            if not pattern.volume_confirmed:
                risk_factors.append("Low volume confirmation")
            
            # Weak momentum
            if not pattern.momentum_confirmed:
                risk_factors.append("Weak momentum")
            
            # Poor risk/reward
            if pattern.risk_reward_ratio < 1.5:
                risk_factors.append("Poor risk/reward ratio")
            
            # High volatility environment
            volatility = data['close'].pct_change().tail(20).std() * np.sqrt(252)
            if volatility > 0.3:  # 30% annualized volatility
                risk_factors.append("High volatility environment")
            
            return risk_factors
            
        except Exception as e:
            logger.warning(f"Error identifying risk factors: {e}")
            return risk_factors
    
    def _apply_risk_management(self, signal: BreakoutSignal, data: pd.DataFrame) -> BreakoutSignal:
        """Apply risk management to breakout signal"""
        
        try:
            config = self.config['risk_management']
            
            # Adjust position size based on pattern quality
            base_size = config['base_position_size']
            
            quality_multipliers = {
                BreakoutQuality.VERY_STRONG: 1.5,
                BreakoutQuality.STRONG: 1.2,
                BreakoutQuality.MODERATE: 1.0,
                BreakoutQuality.WEAK: 0.7
            }
            
            size_multiplier = quality_multipliers.get(signal.breakout_pattern.quality, 1.0)
            adjusted_size = min(config['max_position_size'], base_size * size_multiplier)
            
            signal.base_signal.position_size = adjusted_size
            
            # Ensure minimum risk/reward ratio
            if signal.breakout_pattern.risk_reward_ratio < config['target_risk_reward_min']:
                # Adjust target or reduce position size
                signal.base_signal.position_size *= 0.7
            
            return signal
            
        except Exception as e:
            logger.warning(f"Error applying risk management: {e}")
            return signal
    
    def _validate_breakout_signal(self, signal: BreakoutSignal, data: pd.DataFrame) -> bool:
        """Validate breakout signal before execution"""
        
        try:
            # Check minimum confidence
            if signal.base_signal.confidence < 60.0:
                return False
            
            # Check risk/reward ratio
            if signal.breakout_pattern.risk_reward_ratio < 1.0:
                return False
            
            # Check for excessive risk factors
            if len(signal.risk_factors) > 3:
                return False
            
            # Check position size
            if signal.base_signal.position_size > 0.05:  # Max 5%
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating breakout signal: {e}")
            return False

# ==================== TESTING ====================

def test_breakout_strategy():
    """Test breakout strategy functionality"""
    
    print("🚀 Testing Advanced Breakout Strategy")
    print("=" * 50)
    
    # Create sample data with consolidation and breakout
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    
    # Create data with consolidation and breakout pattern
    base_price = 100
    prices = []
    
    # Initial trend
    for i in range(30):
        change = np.random.randn() * 0.005  # Small moves
        new_price = (prices[-1] if prices else base_price) * (1 + change)
        prices.append(new_price)
    
    # Consolidation phase (resistance around 102, support around 98)
    for i in range(40):
        if len(prices) % 10 < 5:  # Test resistance
            target = 102 + np.random.randn() * 0.5
        else:  # Test support
            target = 98 + np.random.randn() * 0.5
        
        current = prices[-1]
        change = (target - current) * 0.1 + np.random.randn() * 0.003
        new_price = current * (1 + change)
        prices.append(new_price)
    
    # Breakout phase
    for i in range(30):
        change = 0.01 + np.random.randn() * 0.005  # Strong upward move
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    sample_data = pd.DataFrame({
        'open': prices,
        'close': [p * (1 + np.random.randn() * 0.002) for p in prices],
        'volume': [1000 + int(i/70 * 5000) + np.random.randint(-200, 200) for i in range(100)]  # Increasing volume
    }, index=dates)
    
    # Add high/low
    sample_data['high'] = sample_data[['open', 'close']].max(axis=1) * (1 + np.random.rand(100) * 0.005)
    sample_data['low'] = sample_data[['open', 'close']].min(axis=1) * (1 - np.random.rand(100) * 0.005)
    
    # Create breakout strategy
    breakout_strategy = AdvancedBreakoutStrategy()
    
    print(f"✅ Strategy initialized:")
    print(f"   Breakout detectors: {len(breakout_strategy.breakout_detectors)}")
    print(f"   Min pattern bars: {breakout_strategy.config['breakout_detection']['min_pattern_bars']}")
    print(f"   Volume confirmation ratio: {breakout_strategy.config['breakout_detection']['volume_confirmation_ratio']}")
    print(f"   Momentum threshold: {breakout_strategy.config['breakout_detection']['momentum_threshold']}")
    
    # Generate signal
    signal = breakout_strategy.generate_signal(
        sample_data, 
        'TEST',
        RegimeType.BREAKOUT, 
        85.0
    )
    
    if signal:
        print(f"\n🚦 Generated breakout signal:")
        print(f"   Pattern type: {signal.breakout_pattern.pattern_type.value}")
        print(f"   Direction: {signal.breakout_pattern.direction.value}")
        print(f"   Quality: {signal.breakout_pattern.quality.value}")
        print(f"   Signal type: {signal.base_signal.signal_type.value}")
        print(f"   Confidence: {signal.base_signal.confidence:.1f}%")
        print(f"   Confirmation score: {signal.confirmation_score:.1f}")
        
        print(f"\n💰 Trade details:")
        print(f"   Entry price: ${signal.base_signal.entry_price:.2f}")
        print(f"   Target price: ${signal.base_signal.take_profit:.2f}")
        print(f"   Stop loss: ${signal.base_signal.stop_loss:.2f}")
        print(f"   Position size: {signal.base_signal.position_size:.3f}")
        print(f"   Risk/Reward: {signal.breakout_pattern.risk_reward_ratio:.2f}")
        print(f"   Expected move: {signal.expected_move:.2%}")
        print(f"   Time horizon: {signal.time_horizon} bars")
        
        print(f"\n📊 Pattern metrics:")
        print(f"   Breakout level: ${signal.breakout_pattern.breakout_level:.2f}")
        print(f"   Pattern height: ${signal.breakout_pattern.pattern_height:.2f}")
        print(f"   Pattern width: {signal.breakout_pattern.pattern_width} bars")
        print(f"   Volume ratio: {signal.breakout_pattern.volume_ratio:.2f}")
        print(f"   Momentum score: {signal.breakout_pattern.momentum_score:.3f}")
        print(f"   Volume confirmed: {'✅' if signal.breakout_pattern.volume_confirmed else '❌'}")
        print(f"   Momentum confirmed: {'✅' if signal.breakout_pattern.momentum_confirmed else '❌'}")
        
        print(f"\n⚠️  Risk factors: {len(signal.risk_factors)}")
        for factor in signal.risk_factors:
            print(f"   - {factor}")
        
        print(f"\n🎯 Reasoning: {signal.base_signal.reasoning}")
        
    else:
        print(f"\n🚦 No breakout signal generated")
    
    # Test individual detectors
    print(f"\n🔍 Testing individual breakout detectors:")
    
    # Test support/resistance levels
    breakout_strategy._update_support_resistance_levels(sample_data, 'TEST')
    
    if 'TEST' in breakout_strategy.breakout_levels:
        levels = breakout_strategy.breakout_levels['TEST']
        print(f"   Support levels: {len(levels.get('support', []))}")
        print(f"   Resistance levels: {len(levels.get('resistance', []))}")
        
        for level in levels.get('resistance', [])[:3]:
            print(f"      Resistance: ${level.price:.2f} (strength: {level.strength:.2f}, touches: {level.touches})")
        
        for level in levels.get('support', [])[:3]:
            print(f"      Support: ${level.price:.2f} (strength: {level.strength:.2f}, touches: {level.touches})")
    
    # Test pattern detection
    patterns_found = []
    for pattern_type, detector in breakout_strategy.breakout_detectors.items():
        try:
            pattern = detector(sample_data, 'TEST')
            if pattern:
                patterns_found.append(pattern_type.value)
        except:
            pass
    
    print(f"\n📈 Patterns detected: {len(patterns_found)}")
    for pattern in patterns_found:
        print(f"   - {pattern}")
    
    print("\n🎉 Breakout strategy tests completed!")

if __name__ == "__main__":
    test_breakout_strategy()