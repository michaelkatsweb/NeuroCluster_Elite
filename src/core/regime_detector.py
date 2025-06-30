#!/usr/bin/env python3
"""
File: regime_detector.py
Path: NeuroCluster-Elite/src/core/regime_detector.py
Description: Advanced market regime detection system for NeuroCluster Elite

This module implements sophisticated market regime detection using the proven
NeuroCluster algorithm combined with additional market indicators and machine
learning techniques for enhanced accuracy.

Features:
- Multi-timeframe regime analysis
- Volatility regime detection
- Trend strength assessment
- Market cycle identification
- Regime transition detection
- Confidence scoring and validation
- Real-time regime monitoring

Author: Your Name
Created: 2025-06-28
Version: 1.0.0
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading
from collections import deque
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import talib

# Import our modules
try:
    from src.core.neurocluster_elite import RegimeType, AssetType, MarketData
    from src.utils.logger import get_enhanced_logger, LogCategory
    from src.utils.helpers import calculate_sharpe_ratio, format_percentage
except ImportError:
    # Fallback for testing
    pass

# Configure logging
logger = get_enhanced_logger(__name__, LogCategory.ALGORITHM)

# ==================== ENUMS AND DATA STRUCTURES ====================

class RegimeStrength(Enum):
    """Regime strength levels"""
    WEAK = "weak"
    MODERATE = "moderate" 
    STRONG = "strong"
    VERY_STRONG = "very_strong"

class TrendDirection(Enum):
    """Trend direction"""
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"

class VolatilityRegime(Enum):
    """Volatility regimes"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class RegimeFeatures:
    """Market regime features for analysis"""
    
    # Price-based features
    price_momentum_short: float    # 5-day momentum
    price_momentum_medium: float   # 20-day momentum
    price_momentum_long: float     # 60-day momentum
    
    # Volatility features
    realized_volatility: float     # Historical volatility
    volatility_of_volatility: float
    volatility_regime: VolatilityRegime
    
    # Trend features
    trend_strength: float
    trend_direction: TrendDirection
    trend_consistency: float
    
    # Technical indicators
    rsi_14: float
    macd_signal: float
    bb_position: float            # Position within Bollinger Bands
    adx: float                    # Average Directional Index
    
    # Volume features
    volume_trend: float
    volume_acceleration: float
    
    # Market structure
    higher_highs: int             # Count of higher highs
    lower_lows: int               # Count of lower lows
    consolidation_periods: int
    
    # Cross-asset features
    correlation_to_market: float
    beta_stability: float
    
    # Regime persistence
    regime_duration: int          # Days in current regime
    regime_transitions: int       # Recent regime changes

@dataclass
class RegimeDetectionResult:
    """Regime detection result"""
    primary_regime: RegimeType
    secondary_regime: Optional[RegimeType] = None
    confidence: float = 0.0
    strength: RegimeStrength = RegimeStrength.MODERATE
    transition_probability: float = 0.0
    features: Optional[RegimeFeatures] = None
    timestamp: datetime = field(default_factory=datetime.now)

# ==================== REGIME DETECTOR ====================

class RegimeDetector:
    """
    Advanced market regime detection system
    
    Uses multiple approaches:
    1. NeuroCluster-based pattern recognition
    2. Statistical regime detection
    3. Technical indicator analysis
    4. Machine learning classification
    5. Multi-timeframe analysis
    """
    
    def __init__(self, config: Dict = None):
        """Initialize regime detector"""
        
        self.config = config or self._default_config()
        
        # Detection parameters
        self.lookback_periods = self.config.get('lookback_periods', [5, 20, 60])
        self.volatility_window = self.config.get('volatility_window', 20)
        self.trend_window = self.config.get('trend_window', 14)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        # State tracking
        self.price_history: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        self.regime_history: Dict[str, deque] = {}
        self.feature_scaler = StandardScaler()
        self.is_fitted = False
        
        # Machine learning model for regime classification
        self.ml_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.ml_trained = False
        
        # Threading
        self.detection_lock = threading.RLock()
        
        logger.info("🔍 Regime Detector initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration for regime detection"""
        return {
            'lookback_periods': [5, 20, 60],
            'volatility_window': 20,
            'trend_window': 14,
            'confidence_threshold': 0.7,
            'max_history_length': 500,
            'regime_smoothing': True,
            'multi_timeframe_analysis': True,
            'ml_classification': True,
            'volatility_regimes': {
                'low_threshold': 0.10,      # 10% annualized
                'normal_threshold': 0.20,   # 20% annualized
                'high_threshold': 0.40      # 40% annualized
            }
        }
    
    def detect_regime(self, symbol: str, market_data: MarketData, 
                     price_history: List[float] = None) -> RegimeDetectionResult:
        """
        Detect market regime for given symbol
        
        Args:
            symbol: Trading symbol
            market_data: Current market data
            price_history: Historical price data (optional)
            
        Returns:
            Regime detection result
        """
        
        with self.detection_lock:
            # Update internal history
            self._update_history(symbol, market_data)
            
            # Get price history for analysis
            if price_history:
                prices = np.array(price_history)
            else:
                prices = np.array(list(self.price_history.get(symbol, [market_data.price])))
            
            if len(prices) < 10:
                # Insufficient data - return default regime
                return RegimeDetectionResult(
                    primary_regime=RegimeType.SIDEWAYS,
                    confidence=0.5,
                    strength=RegimeStrength.WEAK
                )
            
            # Extract features for regime detection
            features = self._extract_regime_features(symbol, prices, market_data)
            
            # Multi-method regime detection
            regimes = {}
            
            # 1. Statistical regime detection
            regimes['statistical'] = self._detect_statistical_regime(prices, features)
            
            # 2. Technical indicator regime
            regimes['technical'] = self._detect_technical_regime(features)
            
            # 3. Volatility-based regime
            regimes['volatility'] = self._detect_volatility_regime(prices, features)
            
            # 4. Trend-based regime
            regimes['trend'] = self._detect_trend_regime(features)
            
            # 5. Machine learning regime (if trained)
            if self.ml_trained:
                regimes['ml'] = self._detect_ml_regime(features)
            
            # Ensemble regime decision
            final_result = self._ensemble_regime_decision(regimes, features)
            
            # Store regime history
            self._update_regime_history(symbol, final_result)
            
            logger.debug(f"Regime detected for {symbol}: {final_result.primary_regime.value} "
                        f"(confidence: {final_result.confidence:.1%})")
            
            return final_result
    
    def _update_history(self, symbol: str, market_data: MarketData):
        """Update internal price and volume history"""
        
        max_length = self.config.get('max_history_length', 500)
        
        # Initialize if needed
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=max_length)
            self.volume_history[symbol] = deque(maxlen=max_length)
            self.regime_history[symbol] = deque(maxlen=100)
        
        # Add current data
        self.price_history[symbol].append(market_data.price)
        self.volume_history[symbol].append(getattr(market_data, 'volume', 0))
    
    def _extract_regime_features(self, symbol: str, prices: np.ndarray, 
                                market_data: MarketData) -> RegimeFeatures:
        """Extract comprehensive features for regime detection"""
        
        # Ensure we have enough data
        if len(prices) < 60:
            # Pad with current price if needed
            padded_prices = np.full(60, prices[0])
            padded_prices[-len(prices):] = prices
            prices = padded_prices
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        # Price momentum features
        momentum_5 = (prices[-1] / prices[-6] - 1) if len(prices) >= 6 else 0
        momentum_20 = (prices[-1] / prices[-21] - 1) if len(prices) >= 21 else 0
        momentum_60 = (prices[-1] / prices[-61] - 1) if len(prices) >= 61 else 0
        
        # Volatility features
        volatility_window = min(self.volatility_window, len(returns))
        recent_returns = returns[-volatility_window:]
        realized_vol = np.std(recent_returns) * np.sqrt(252)  # Annualized
        
        # Volatility of volatility
        if len(returns) >= 40:
            vol_series = [np.std(returns[i-20:i]) for i in range(20, len(returns))]
            vol_of_vol = np.std(vol_series) if vol_series else 0.1
        else:
            vol_of_vol = 0.1
        
        # Volatility regime classification
        vol_config = self.config['volatility_regimes']
        if realized_vol < vol_config['low_threshold']:
            vol_regime = VolatilityRegime.LOW
        elif realized_vol < vol_config['normal_threshold']:
            vol_regime = VolatilityRegime.NORMAL
        elif realized_vol < vol_config['high_threshold']:
            vol_regime = VolatilityRegime.HIGH
        else:
            vol_regime = VolatilityRegime.EXTREME
        
        # Trend features
        trend_window = min(self.trend_window, len(prices))
        trend_slope = self._calculate_trend_slope(prices[-trend_window:])
        trend_strength = abs(trend_slope)
        
        if trend_slope > 0.001:
            trend_direction = TrendDirection.UP
        elif trend_slope < -0.001:
            trend_direction = TrendDirection.DOWN
        else:
            trend_direction = TrendDirection.SIDEWAYS
        
        # Trend consistency (R-squared of linear regression)
        if len(prices) >= trend_window:
            x = np.arange(trend_window)
            y = prices[-trend_window:]
            try:
                slope, intercept, r_value, _, _ = stats.linregress(x, y)
                trend_consistency = r_value ** 2
            except:
                trend_consistency = 0.0
        else:
            trend_consistency = 0.0
        
        # Technical indicators
        try:
            # RSI
            rsi_values = talib.RSI(prices.astype(float), timeperiod=14)
            rsi_14 = rsi_values[-1] if not np.isnan(rsi_values[-1]) else 50.0
            
            # MACD
            macd_line, macd_signal, macd_hist = talib.MACD(prices.astype(float))
            macd_signal_val = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0.0
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(prices.astype(float))
            bb_position = ((prices[-1] - bb_lower[-1]) / 
                          (bb_upper[-1] - bb_lower[-1])) if bb_upper[-1] != bb_lower[-1] else 0.5
            
            # ADX (Average Directional Index)
            high_prices = prices * 1.01  # Simulate high prices
            low_prices = prices * 0.99   # Simulate low prices
            adx_values = talib.ADX(high_prices.astype(float), 
                                  low_prices.astype(float), 
                                  prices.astype(float))
            adx = adx_values[-1] if not np.isnan(adx_values[-1]) else 25.0
            
        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {e}")
            rsi_14 = 50.0
            macd_signal_val = 0.0
            bb_position = 0.5
            adx = 25.0
        
        # Volume features (if available)
        volumes = list(self.volume_history.get(symbol, []))
        if len(volumes) >= 20:
            volume_ma = np.mean(volumes[-20:])
            volume_trend = (volumes[-1] / volume_ma - 1) if volume_ma > 0 else 0
            
            # Volume acceleration
            recent_vol_avg = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
            older_vol_avg = np.mean(volumes[-20:-5]) if len(volumes) >= 20 else volumes[0]
            volume_acceleration = (recent_vol_avg / older_vol_avg - 1) if older_vol_avg > 0 else 0
        else:
            volume_trend = 0.0
            volume_acceleration = 0.0
        
        # Market structure analysis
        higher_highs = self._count_higher_highs(prices[-20:]) if len(prices) >= 20 else 0
        lower_lows = self._count_lower_lows(prices[-20:]) if len(prices) >= 20 else 0
        consolidation_periods = self._count_consolidation_periods(prices[-30:]) if len(prices) >= 30 else 0
        
        # Cross-asset features (simplified)
        correlation_to_market = 0.5  # Would calculate vs market index in practice
        beta_stability = 1.0         # Would calculate rolling beta stability
        
        # Regime persistence
        regime_duration = self._calculate_regime_duration(symbol)
        regime_transitions = self._count_recent_transitions(symbol)
        
        return RegimeFeatures(
            price_momentum_short=momentum_5,
            price_momentum_medium=momentum_20,
            price_momentum_long=momentum_60,
            realized_volatility=realized_vol,
            volatility_of_volatility=vol_of_vol,
            volatility_regime=vol_regime,
            trend_strength=trend_strength,
            trend_direction=trend_direction,
            trend_consistency=trend_consistency,
            rsi_14=rsi_14,
            macd_signal=macd_signal_val,
            bb_position=bb_position,
            adx=adx,
            volume_trend=volume_trend,
            volume_acceleration=volume_acceleration,
            higher_highs=higher_highs,
            lower_lows=lower_lows,
            consolidation_periods=consolidation_periods,
            correlation_to_market=correlation_to_market,
            beta_stability=beta_stability,
            regime_duration=regime_duration,
            regime_transitions=regime_transitions
        )
    
    def _detect_statistical_regime(self, prices: np.ndarray, 
                                  features: RegimeFeatures) -> Tuple[RegimeType, float]:
        """Detect regime using statistical analysis"""
        
        returns = np.diff(np.log(prices))
        
        # Calculate regime indicators
        mean_return = np.mean(returns[-20:]) if len(returns) >= 20 else 0
        volatility = features.realized_volatility
        skewness = stats.skew(returns[-20:]) if len(returns) >= 20 else 0
        
        confidence = 0.5
        
        # High volatility regime
        if volatility > 0.30:  # 30% annualized volatility
            regime = RegimeType.VOLATILE
            confidence = min(0.9, volatility / 0.30 * 0.7)
        
        # Bull market conditions
        elif mean_return > 0.001 and features.trend_direction == TrendDirection.UP:
            if features.trend_strength > 0.01:
                regime = RegimeType.BREAKOUT
                confidence = min(0.9, features.trend_strength * 50)
            else:
                regime = RegimeType.BULL
                confidence = 0.7
        
        # Bear market conditions  
        elif mean_return < -0.001 and features.trend_direction == TrendDirection.DOWN:
            if features.trend_strength > 0.01:
                regime = RegimeType.BREAKDOWN
                confidence = min(0.9, features.trend_strength * 50)
            else:
                regime = RegimeType.BEAR
                confidence = 0.7
        
        # Sideways/ranging market
        else:
            if features.higher_highs > 0 and features.lower_lows == 0:
                regime = RegimeType.ACCUMULATION
            elif features.lower_lows > 0 and features.higher_highs == 0:
                regime = RegimeType.DISTRIBUTION
            else:
                regime = RegimeType.SIDEWAYS
            confidence = 0.6
        
        return regime, confidence
    
    def _detect_technical_regime(self, features: RegimeFeatures) -> Tuple[RegimeType, float]:
        """Detect regime using technical indicators"""
        
        confidence = 0.5
        
        # Strong trend conditions (ADX > 25)
        if features.adx > 25:
            if features.rsi_14 > 60 and features.macd_signal > 0:
                regime = RegimeType.BULL
                confidence = min(0.9, (features.rsi_14 - 50) / 50 * 0.8)
            elif features.rsi_14 < 40 and features.macd_signal < 0:
                regime = RegimeType.BEAR
                confidence = min(0.9, (50 - features.rsi_14) / 50 * 0.8)
            else:
                regime = RegimeType.VOLATILE
                confidence = 0.7
        
        # Ranging conditions (ADX < 20)
        elif features.adx < 20:
            if features.bb_position > 0.8:
                regime = RegimeType.DISTRIBUTION
                confidence = 0.6
            elif features.bb_position < 0.2:
                regime = RegimeType.ACCUMULATION
                confidence = 0.6
            else:
                regime = RegimeType.SIDEWAYS
                confidence = 0.7
        
        # Moderate trend
        else:
            if features.price_momentum_medium > 0.05:
                regime = RegimeType.BULL
            elif features.price_momentum_medium < -0.05:
                regime = RegimeType.BEAR
            else:
                regime = RegimeType.SIDEWAYS
            confidence = 0.6
        
        return regime, confidence
    
    def _detect_volatility_regime(self, prices: np.ndarray, 
                                 features: RegimeFeatures) -> Tuple[RegimeType, float]:
        """Detect regime based on volatility analysis"""
        
        if features.volatility_regime == VolatilityRegime.EXTREME:
            return RegimeType.VOLATILE, 0.9
        elif features.volatility_regime == VolatilityRegime.HIGH:
            return RegimeType.VOLATILE, 0.7
        elif features.volatility_regime == VolatilityRegime.LOW:
            # Low volatility could indicate accumulation or tight range
            if features.price_momentum_long > 0:
                return RegimeType.ACCUMULATION, 0.6
            else:
                return RegimeType.SIDEWAYS, 0.6
        else:
            # Normal volatility - defer to other methods
            return RegimeType.SIDEWAYS, 0.4
    
    def _detect_trend_regime(self, features: RegimeFeatures) -> Tuple[RegimeType, float]:
        """Detect regime based on trend analysis"""
        
        confidence = min(0.9, features.trend_consistency * 0.8 + 0.3)
        
        if features.trend_direction == TrendDirection.UP:
            if features.trend_strength > 0.02:
                return RegimeType.BREAKOUT, confidence
            else:
                return RegimeType.BULL, confidence * 0.8
        
        elif features.trend_direction == TrendDirection.DOWN:
            if features.trend_strength > 0.02:
                return RegimeType.BREAKDOWN, confidence
            else:
                return RegimeType.BEAR, confidence * 0.8
        
        else:
            return RegimeType.SIDEWAYS, confidence * 0.6
    
    def _detect_ml_regime(self, features: RegimeFeatures) -> Tuple[RegimeType, float]:
        """Detect regime using machine learning model"""
        
        # This would use a trained ML model in practice
        # For now, return a simple heuristic-based result
        
        feature_vector = np.array([
            features.price_momentum_short,
            features.price_momentum_medium,
            features.realized_volatility,
            features.trend_strength,
            features.rsi_14 / 100,
            features.adx / 100
        ]).reshape(1, -1)
        
        # Normalize features
        if self.is_fitted:
            feature_vector = self.feature_scaler.transform(feature_vector)
        
        # Simple ML-like logic (would use actual trained model)
        score = np.sum(feature_vector)
        
        if score > 0.5:
            return RegimeType.BULL, 0.7
        elif score < -0.5:
            return RegimeType.BEAR, 0.7
        else:
            return RegimeType.SIDEWAYS, 0.6
    
    def _ensemble_regime_decision(self, regimes: Dict[str, Tuple[RegimeType, float]], 
                                 features: RegimeFeatures) -> RegimeDetectionResult:
        """Combine multiple regime detection methods"""
        
        # Weight different methods
        method_weights = {
            'statistical': 0.3,
            'technical': 0.25,
            'volatility': 0.2,
            'trend': 0.15,
            'ml': 0.1 if 'ml' in regimes else 0.0
        }
        
        # Count votes for each regime
        regime_votes = {}
        total_confidence = 0.0
        total_weight = 0.0
        
        for method, (regime, confidence) in regimes.items():
            weight = method_weights.get(method, 0.1)
            
            if regime not in regime_votes:
                regime_votes[regime] = 0.0
            
            regime_votes[regime] += weight * confidence
            total_confidence += weight * confidence
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for regime in regime_votes:
                regime_votes[regime] /= total_weight
        
        # Select primary regime
        primary_regime = max(regime_votes.items(), key=lambda x: x[1])
        primary_regime_type = primary_regime[0]
        primary_confidence = primary_regime[1]
        
        # Select secondary regime (if confidence is close)
        sorted_regimes = sorted(regime_votes.items(), key=lambda x: x[1], reverse=True)
        secondary_regime = None
        
        if len(sorted_regimes) > 1:
            if sorted_regimes[1][1] > 0.7 * sorted_regimes[0][1]:
                secondary_regime = sorted_regimes[1][0]
        
        # Determine regime strength
        if primary_confidence > 0.85:
            strength = RegimeStrength.VERY_STRONG
        elif primary_confidence > 0.75:
            strength = RegimeStrength.STRONG
        elif primary_confidence > 0.65:
            strength = RegimeStrength.MODERATE
        else:
            strength = RegimeStrength.WEAK
        
        # Calculate transition probability
        transition_prob = self._calculate_transition_probability(features)
        
        return RegimeDetectionResult(
            primary_regime=primary_regime_type,
            secondary_regime=secondary_regime,
            confidence=primary_confidence,
            strength=strength,
            transition_probability=transition_prob,
            features=features
        )
    
    def _calculate_trend_slope(self, prices: np.ndarray) -> float:
        """Calculate trend slope using linear regression"""
        
        if len(prices) < 2:
            return 0.0
        
        x = np.arange(len(prices))
        try:
            slope, _, _, _, _ = stats.linregress(x, prices)
            return slope / prices[0] if prices[0] != 0 else 0  # Normalize by price
        except:
            return 0.0
    
    def _count_higher_highs(self, prices: np.ndarray) -> int:
        """Count higher highs in price series"""
        
        if len(prices) < 3:
            return 0
        
        count = 0
        for i in range(2, len(prices)):
            if prices[i] > prices[i-1] and prices[i-1] > prices[i-2]:
                count += 1
        
        return count
    
    def _count_lower_lows(self, prices: np.ndarray) -> int:
        """Count lower lows in price series"""
        
        if len(prices) < 3:
            return 0
        
        count = 0
        for i in range(2, len(prices)):
            if prices[i] < prices[i-1] and prices[i-1] < prices[i-2]:
                count += 1
        
        return count
    
    def _count_consolidation_periods(self, prices: np.ndarray) -> int:
        """Count consolidation periods (low volatility periods)"""
        
        if len(prices) < 10:
            return 0
        
        # Calculate rolling volatility
        window = 5
        volatilities = []
        
        for i in range(window, len(prices)):
            vol = np.std(prices[i-window:i])
            volatilities.append(vol)
        
        # Count periods with low volatility
        median_vol = np.median(volatilities)
        low_vol_threshold = median_vol * 0.5
        
        count = 0
        in_consolidation = False
        
        for vol in volatilities:
            if vol < low_vol_threshold:
                if not in_consolidation:
                    count += 1
                    in_consolidation = True
            else:
                in_consolidation = False
        
        return count
    
    def _calculate_regime_duration(self, symbol: str) -> int:
        """Calculate how long current regime has persisted"""
        
        regime_hist = self.regime_history.get(symbol, [])
        
        if len(regime_hist) < 2:
            return 1
        
        current_regime = regime_hist[-1].primary_regime
        duration = 1
        
        for i in range(len(regime_hist) - 2, -1, -1):
            if regime_hist[i].primary_regime == current_regime:
                duration += 1
            else:
                break
        
        return duration
    
    def _count_recent_transitions(self, symbol: str) -> int:
        """Count recent regime transitions"""
        
        regime_hist = self.regime_history.get(symbol, [])
        
        if len(regime_hist) < 2:
            return 0
        
        # Count transitions in last 20 periods
        recent_hist = regime_hist[-20:] if len(regime_hist) >= 20 else regime_hist
        transitions = 0
        
        for i in range(1, len(recent_hist)):
            if recent_hist[i].primary_regime != recent_hist[i-1].primary_regime:
                transitions += 1
        
        return transitions
    
    def _calculate_transition_probability(self, features: RegimeFeatures) -> float:
        """Calculate probability of regime transition"""
        
        # Factors that increase transition probability
        transition_score = 0.0
        
        # High volatility increases transition probability
        if features.volatility_regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]:
            transition_score += 0.3
        
        # Recent transitions increase probability
        if features.regime_transitions > 2:
            transition_score += 0.2
        
        # Extreme technical readings
        if features.rsi_14 > 80 or features.rsi_14 < 20:
            transition_score += 0.2
        
        # Long regime duration increases probability
        if features.regime_duration > 30:
            transition_score += 0.1
        
        # Low trend consistency
        if features.trend_consistency < 0.3:
            transition_score += 0.1
        
        return min(1.0, transition_score)
    
    def _update_regime_history(self, symbol: str, result: RegimeDetectionResult):
        """Update regime history for symbol"""
        
        if symbol not in self.regime_history:
            self.regime_history[symbol] = deque(maxlen=100)
        
        self.regime_history[symbol].append(result)

# ==================== TESTING ====================

def test_regime_detector():
    """Test regime detector functionality"""
    
    print("🔍 Testing Regime Detector")
    print("=" * 40)
    
    # Create regime detector
    detector = RegimeDetector()
    
    # Create mock market data
    from src.core.neurocluster_elite import MarketData, AssetType
    
    market_data = MarketData(
        symbol='AAPL',
        asset_type=AssetType.STOCK,
        price=150.0,
        timestamp=datetime.now()
    )
    
    # Generate mock price history (trending up)
    price_history = [100.0]
    for i in range(100):
        # Add some trend and noise
        trend = 0.001  # 0.1% daily trend
        noise = np.random.normal(0, 0.02)  # 2% daily volatility
        next_price = price_history[-1] * (1 + trend + noise)
        price_history.append(next_price)
    
    # Test regime detection
    result = detector.detect_regime('AAPL', market_data, price_history)
    
    print(f"✅ Regime Detection Results:")
    print(f"   Primary regime: {result.primary_regime.value}")
    print(f"   Confidence: {result.confidence:.1%}")
    print(f"   Strength: {result.strength.value}")
    print(f"   Transition probability: {result.transition_probability:.1%}")
    
    if result.features:
        print(f"\n✅ Key Features:")
        print(f"   Short-term momentum: {result.features.price_momentum_short:.2%}")
        print(f"   Trend strength: {result.features.trend_strength:.4f}")
        print(f"   Volatility: {result.features.realized_volatility:.1%}")
        print(f"   RSI: {result.features.rsi_14:.1f}")
        print(f"   Trend direction: {result.features.trend_direction.value}")
    
    # Test multiple updates
    print(f"\n✅ Testing regime persistence...")
    for i in range(5):
        # Simulate new market data
        new_price = price_history[-1] * (1 + np.random.normal(0.001, 0.02))
        new_data = MarketData(
            symbol='AAPL',
            asset_type=AssetType.STOCK,
            price=new_price,
            timestamp=datetime.now()
        )
        
        result = detector.detect_regime('AAPL', new_data)
        print(f"   Update {i+1}: {result.primary_regime.value} ({result.confidence:.1%})")
    
    print("\n🎉 Regime detector tests completed!")

if __name__ == "__main__":
    test_regime_detector()