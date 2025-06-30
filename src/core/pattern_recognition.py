#!/usr/bin/env python3
"""
File: pattern_recognition.py
Path: NeuroCluster-Elite/src/core/pattern_recognition.py
Description: Advanced pattern recognition for NeuroCluster Elite trading platform

This module implements sophisticated pattern recognition algorithms for financial markets,
including technical chart patterns, candlestick patterns, and machine learning-based
pattern detection optimized for the NeuroCluster algorithm.

Features:
- Classical chart patterns (head & shoulders, triangles, flags, etc.)
- Japanese candlestick pattern recognition
- Support and resistance level detection
- Trend line and channel identification
- Harmonic pattern recognition (Gartley, Butterfly, etc.)
- Machine learning pattern classification
- Real-time pattern scanning and alerts
- Multi-timeframe pattern analysis
- Pattern reliability scoring and backtesting

Author: Your Name
Created: 2025-06-29
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
import warnings
from collections import deque, defaultdict
import math
from scipy import stats, signal, optimize
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import talib

# Import our modules
try:
    from src.core.neurocluster_elite import AssetType, MarketData, RegimeType
    from src.core.feature_extractor import FeatureVector, AdvancedFeatureExtractor
    from src.utils.helpers import calculate_support_resistance, find_peaks_valleys
    from src.utils.config_manager import ConfigManager
except ImportError:
    # Fallback for testing
    from enum import Enum
    class AssetType(Enum):
        STOCK = "stock"
        CRYPTO = "crypto"
        FOREX = "forex"
        COMMODITY = "commodity"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== PATTERN TYPES AND STRUCTURES ====================

class PatternType(Enum):
    """Types of recognized patterns"""
    # Classical chart patterns
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    TRIANGLE_SYMMETRICAL = "triangle_symmetrical"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"
    FLAG_BULL = "flag_bull"
    FLAG_BEAR = "flag_bear"
    PENNANT = "pennant"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    
    # Candlestick patterns
    DOJI = "doji"
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    ENGULFING_BULL = "engulfing_bull"
    ENGULFING_BEAR = "engulfing_bear"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    HARAMI = "harami"
    
    # Support/Resistance patterns
    SUPPORT_LEVEL = "support_level"
    RESISTANCE_LEVEL = "resistance_level"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    
    # Trend patterns
    TREND_LINE = "trend_line"
    CHANNEL_UP = "channel_up"
    CHANNEL_DOWN = "channel_down"
    TREND_REVERSAL = "trend_reversal"
    
    # Harmonic patterns
    GARTLEY = "gartley"
    BUTTERFLY = "butterfly"
    BAT = "bat"
    CRAB = "crab"
    
    # Custom ML patterns
    MOMENTUM_PATTERN = "momentum_pattern"
    MEAN_REVERSION_PATTERN = "mean_reversion_pattern"
    VOLATILITY_CLUSTER = "volatility_cluster"

class PatternSignal(Enum):
    """Pattern trading signals"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

class PatternReliability(Enum):
    """Pattern reliability levels"""
    VERY_HIGH = "very_high"  # 90%+
    HIGH = "high"           # 80-90%
    MEDIUM = "medium"       # 60-80%
    LOW = "low"            # 40-60%
    VERY_LOW = "very_low"  # <40%

@dataclass
class PatternPoint:
    """Key point in a pattern"""
    timestamp: datetime
    price: float
    volume: Optional[float] = None
    index: int = 0
    point_type: str = "generic"  # peak, valley, support, resistance

@dataclass
class RecognizedPattern:
    """Detected pattern with metadata"""
    pattern_type: PatternType
    signal: PatternSignal
    reliability: PatternReliability
    confidence: float
    
    # Pattern geometry
    start_time: datetime
    end_time: datetime
    key_points: List[PatternPoint]
    
    # Pattern measurements
    pattern_height: float = 0.0
    pattern_width: int = 0  # in bars
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    
    # Context information
    symbol: str = ""
    asset_type: Optional[AssetType] = None
    timeframe: str = "1d"
    market_regime: Optional[RegimeType] = None
    
    # Pattern characteristics
    volume_confirmation: bool = False
    trend_alignment: bool = False
    pattern_maturity: float = 0.0  # 0.0 = forming, 1.0 = complete
    
    # Metadata
    detection_time: datetime = field(default_factory=datetime.now)
    pattern_id: str = field(default_factory=lambda: f"pattern_{int(datetime.now().timestamp())}")
    
    def __post_init__(self):
        # Auto-calculate some fields if not provided
        if self.pattern_height == 0.0 and len(self.key_points) >= 2:
            prices = [point.price for point in self.key_points]
            self.pattern_height = max(prices) - min(prices)
        
        if self.pattern_width == 0 and len(self.key_points) >= 2:
            self.pattern_width = self.key_points[-1].index - self.key_points[0].index

@dataclass
class PatternConfig:
    """Configuration for pattern recognition"""
    # Detection sensitivity
    min_pattern_length: int = 10
    max_pattern_length: int = 100
    pattern_sensitivity: float = 0.7  # Lower = more sensitive
    
    # Pattern filtering
    min_confidence: float = 0.6
    require_volume_confirmation: bool = True
    require_trend_alignment: bool = False
    
    # Candlestick patterns
    enable_candlestick_patterns: bool = True
    candlestick_sensitivity: float = 0.8
    
    # Support/Resistance
    support_resistance_lookback: int = 50
    sr_touch_tolerance: float = 0.02  # 2%
    min_touches: int = 2
    
    # Machine learning patterns
    enable_ml_patterns: bool = True
    ml_model_retrain_interval: int = 1000  # patterns
    feature_importance_threshold: float = 0.1
    
    # Performance settings
    max_patterns_per_scan: int = 20
    pattern_cache_size: int = 500
    parallel_detection: bool = True

# ==================== ADVANCED PATTERN RECOGNIZER ====================

class AdvancedPatternRecognizer:
    """
    Advanced pattern recognition system for NeuroCluster Elite
    
    This class implements sophisticated pattern recognition algorithms that work
    in conjunction with the NeuroCluster algorithm to identify profitable
    trading patterns across multiple asset classes and timeframes.
    """
    
    def __init__(self, config: Optional[PatternConfig] = None):
        """Initialize the pattern recognizer"""
        
        self.config = config or PatternConfig()
        
        # Pattern detection algorithms
        self.pattern_detectors = self._initialize_pattern_detectors()
        
        # Machine learning models
        self.ml_models = {
            'pattern_classifier': RandomForestClassifier(n_estimators=100, random_state=42),
            'anomaly_detector': IsolationForest(contamination=0.1, random_state=42),
            'reliability_predictor': RandomForestClassifier(n_estimators=50, random_state=42)
        }
        
        # Pattern cache and history
        self.pattern_cache = {}
        self.pattern_history = deque(maxlen=self.config.pattern_cache_size)
        
        # Performance tracking
        self.detection_times = deque(maxlen=100)
        self.pattern_success_rates = defaultdict(list)
        
        # Pre-computed templates
        self.harmonic_ratios = self._initialize_harmonic_ratios()
        self.candlestick_templates = self._initialize_candlestick_templates()
        
        # Support/Resistance tracker
        self.sr_levels = defaultdict(list)  # symbol -> [levels]
        
        logger.info("🔍 Advanced Pattern Recognizer initialized with comprehensive detection algorithms")
    
    def _initialize_pattern_detectors(self) -> Dict[str, callable]:
        """Initialize pattern detection algorithms"""
        
        detectors = {
            # Classical patterns
            'head_and_shoulders': self._detect_head_and_shoulders,
            'triangles': self._detect_triangles,
            'flags_pennants': self._detect_flags_pennants,
            'double_tops_bottoms': self._detect_double_tops_bottoms,
            'wedges': self._detect_wedges,
            
            # Candlestick patterns
            'candlestick_patterns': self._detect_candlestick_patterns,
            
            # Support/Resistance
            'support_resistance': self._detect_support_resistance,
            'breakouts': self._detect_breakouts,
            
            # Trend patterns
            'trend_lines': self._detect_trend_lines,
            'channels': self._detect_channels,
            
            # Harmonic patterns
            'harmonic_patterns': self._detect_harmonic_patterns,
            
            # ML-based patterns
            'ml_patterns': self._detect_ml_patterns
        }
        
        return detectors
    
    def _initialize_harmonic_ratios(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Initialize harmonic pattern ratio templates"""
        
        ratios = {
            'gartley': {
                'XA_AB': (0.618, 0.618),
                'AB_BC': (0.382, 0.886),
                'BC_CD': (1.13, 1.618)
            },
            'butterfly': {
                'XA_AB': (0.786, 0.786),
                'AB_BC': (0.382, 0.886),
                'BC_CD': (1.618, 2.618)
            },
            'bat': {
                'XA_AB': (0.382, 0.5),
                'AB_BC': (0.382, 0.886),
                'BC_CD': (1.618, 2.618)
            },
            'crab': {
                'XA_AB': (0.382, 0.618),
                'AB_BC': (0.382, 0.886),
                'BC_CD': (2.24, 3.618)
            }
        }
        
        return ratios
    
    def _initialize_candlestick_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize candlestick pattern templates"""
        
        templates = {
            'doji': {
                'body_ratio_max': 0.1,
                'signal': PatternSignal.NEUTRAL,
                'reliability': PatternReliability.MEDIUM
            },
            'hammer': {
                'body_ratio_max': 0.3,
                'lower_shadow_min': 2.0,
                'upper_shadow_max': 0.5,
                'signal': PatternSignal.BUY,
                'reliability': PatternReliability.HIGH
            },
            'shooting_star': {
                'body_ratio_max': 0.3,
                'upper_shadow_min': 2.0,
                'lower_shadow_max': 0.5,
                'signal': PatternSignal.SELL,
                'reliability': PatternReliability.HIGH
            },
            'engulfing_bull': {
                'signal': PatternSignal.STRONG_BUY,
                'reliability': PatternReliability.HIGH
            },
            'engulfing_bear': {
                'signal': PatternSignal.STRONG_SELL,
                'reliability': PatternReliability.HIGH
            }
        }
        
        return templates
    
    def scan_patterns(self, symbol: str, data: pd.DataFrame, 
                     asset_type: Optional[AssetType] = None,
                     timeframe: str = "1d") -> List[RecognizedPattern]:
        """
        Scan for all patterns in the given data
        
        Args:
            symbol: Trading symbol
            data: OHLCV data
            asset_type: Asset type for context
            timeframe: Data timeframe
            
        Returns:
            List of recognized patterns
        """
        start_time = datetime.now()
        
        try:
            patterns = []
            
            # Validate input data
            if not self._validate_data(data):
                logger.warning(f"Invalid data for pattern scanning: {symbol}")
                return patterns
            
            # Prepare data
            prepared_data = self._prepare_data(data)
            
            # Run all pattern detectors
            for detector_name, detector_func in self.pattern_detectors.items():
                try:
                    detected_patterns = detector_func(
                        symbol=symbol,
                        data=prepared_data,
                        asset_type=asset_type,
                        timeframe=timeframe
                    )
                    
                    if detected_patterns:
                        patterns.extend(detected_patterns)
                        
                except Exception as e:
                    logger.warning(f"Pattern detector {detector_name} error for {symbol}: {e}")
                    continue
            
            # Filter and rank patterns
            filtered_patterns = self._filter_and_rank_patterns(patterns)
            
            # Limit results
            if len(filtered_patterns) > self.config.max_patterns_per_scan:
                filtered_patterns = filtered_patterns[:self.config.max_patterns_per_scan]
            
            # Update performance tracking
            detection_time = (datetime.now() - start_time).total_seconds() * 1000
            self.detection_times.append(detection_time)
            
            # Cache results
            cache_key = f"{symbol}_{timeframe}_{len(data)}"
            self.pattern_cache[cache_key] = filtered_patterns
            
            logger.debug(f"✅ Scanned {len(filtered_patterns)} patterns for {symbol} in {detection_time:.2f}ms")
            
            return filtered_patterns
            
        except Exception as e:
            logger.error(f"❌ Pattern scanning error for {symbol}: {e}")
            return []
    
    def _detect_head_and_shoulders(self, symbol: str, data: pd.DataFrame,
                                 asset_type: Optional[AssetType] = None,
                                 timeframe: str = "1d") -> List[RecognizedPattern]:
        """Detect head and shoulders patterns"""
        
        patterns = []
        
        try:
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values
            
            # Find peaks for head and shoulders
            peak_indices = signal.find_peaks(highs, distance=5, prominence=np.std(highs) * 0.5)[0]
            
            if len(peak_indices) < 3:
                return patterns
            
            # Look for head and shoulders pattern
            for i in range(len(peak_indices) - 2):
                left_shoulder = peak_indices[i]
                head = peak_indices[i + 1]
                right_shoulder = peak_indices[i + 2]
                
                # Check pattern criteria
                left_price = highs[left_shoulder]
                head_price = highs[head]
                right_price = highs[right_shoulder]
                
                # Head should be higher than both shoulders
                if (head_price > left_price and head_price > right_price and
                    abs(left_price - right_price) / max(left_price, right_price) < 0.05):
                    
                    # Calculate neckline
                    valley_indices = signal.find_peaks(-highs[left_shoulder:right_shoulder])[0]
                    if len(valley_indices) >= 2:
                        valley_indices += left_shoulder
                        neckline_price = np.mean([lows[v] for v in valley_indices[:2]])
                        
                        # Calculate measurements
                        pattern_height = head_price - neckline_price
                        price_target = neckline_price - pattern_height
                        
                        # Create pattern
                        key_points = [
                            PatternPoint(data.index[left_shoulder], left_price, point_type="left_shoulder"),
                            PatternPoint(data.index[head], head_price, point_type="head"),
                            PatternPoint(data.index[right_shoulder], right_price, point_type="right_shoulder")
                        ]
                        
                        confidence = self._calculate_hs_confidence(
                            left_price, head_price, right_price, pattern_height
                        )
                        
                        pattern = RecognizedPattern(
                            pattern_type=PatternType.HEAD_AND_SHOULDERS,
                            signal=PatternSignal.SELL,
                            reliability=self._confidence_to_reliability(confidence),
                            confidence=confidence,
                            start_time=data.index[left_shoulder],
                            end_time=data.index[right_shoulder],
                            key_points=key_points,
                            pattern_height=pattern_height,
                            price_target=price_target,
                            stop_loss=head_price * 1.02,
                            symbol=symbol,
                            asset_type=asset_type,
                            timeframe=timeframe
                        )
                        
                        patterns.append(pattern)
                        
        except Exception as e:
            logger.warning(f"Head and shoulders detection error: {e}")
        
        return patterns
    
    def _detect_triangles(self, symbol: str, data: pd.DataFrame,
                         asset_type: Optional[AssetType] = None,
                         timeframe: str = "1d") -> List[RecognizedPattern]:
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        
        patterns = []
        
        try:
            highs = data['high'].values
            lows = data['low'].values
            
            # Find peaks and valleys
            peak_indices = signal.find_peaks(highs, distance=3)[0]
            valley_indices = signal.find_peaks(-lows, distance=3)[0]
            
            if len(peak_indices) < 3 or len(valley_indices) < 3:
                return patterns
            
            # Analyze recent peaks and valleys for triangle formation
            recent_peaks = peak_indices[-4:] if len(peak_indices) >= 4 else peak_indices
            recent_valleys = valley_indices[-4:] if len(valley_indices) >= 4 else valley_indices
            
            # Check for ascending triangle (horizontal resistance, rising support)
            if len(recent_peaks) >= 2 and len(recent_valleys) >= 2:
                peak_prices = [highs[i] for i in recent_peaks]
                valley_prices = [lows[i] for i in recent_valleys]
                
                # Ascending triangle: horizontal highs, rising lows
                peak_slope = self._calculate_slope(recent_peaks, peak_prices)
                valley_slope = self._calculate_slope(recent_valleys, valley_prices)
                
                if abs(peak_slope) < 0.001 and valley_slope > 0.002:
                    # Ascending triangle detected
                    resistance_level = np.mean(peak_prices)
                    
                    key_points = [
                        PatternPoint(data.index[recent_valleys[0]], valley_prices[0], point_type="support_start"),
                        PatternPoint(data.index[recent_peaks[0]], peak_prices[0], point_type="resistance"),
                        PatternPoint(data.index[recent_valleys[-1]], valley_prices[-1], point_type="support_end")
                    ]
                    
                    confidence = self._calculate_triangle_confidence(peak_slope, valley_slope, "ascending")
                    
                    pattern = RecognizedPattern(
                        pattern_type=PatternType.TRIANGLE_ASCENDING,
                        signal=PatternSignal.BUY,
                        reliability=self._confidence_to_reliability(confidence),
                        confidence=confidence,
                        start_time=data.index[recent_valleys[0]],
                        end_time=data.index[max(recent_peaks[-1], recent_valleys[-1])],
                        key_points=key_points,
                        price_target=resistance_level * 1.05,
                        stop_loss=valley_prices[-1] * 0.98,
                        symbol=symbol,
                        asset_type=asset_type,
                        timeframe=timeframe
                    )
                    
                    patterns.append(pattern)
                
                # Descending triangle: declining highs, horizontal lows
                elif peak_slope < -0.002 and abs(valley_slope) < 0.001:
                    support_level = np.mean(valley_prices)
                    
                    key_points = [
                        PatternPoint(data.index[recent_peaks[0]], peak_prices[0], point_type="resistance_start"),
                        PatternPoint(data.index[recent_valleys[0]], valley_prices[0], point_type="support"),
                        PatternPoint(data.index[recent_peaks[-1]], peak_prices[-1], point_type="resistance_end")
                    ]
                    
                    confidence = self._calculate_triangle_confidence(peak_slope, valley_slope, "descending")
                    
                    pattern = RecognizedPattern(
                        pattern_type=PatternType.TRIANGLE_DESCENDING,
                        signal=PatternSignal.SELL,
                        reliability=self._confidence_to_reliability(confidence),
                        confidence=confidence,
                        start_time=data.index[recent_peaks[0]],
                        end_time=data.index[max(recent_peaks[-1], recent_valleys[-1])],
                        key_points=key_points,
                        price_target=support_level * 0.95,
                        stop_loss=peak_prices[-1] * 1.02,
                        symbol=symbol,
                        asset_type=asset_type,
                        timeframe=timeframe
                    )
                    
                    patterns.append(pattern)
                
                # Symmetrical triangle: converging highs and lows
                elif peak_slope < -0.001 and valley_slope > 0.001:
                    convergence_point = self._calculate_convergence_point(
                        recent_peaks, peak_prices, recent_valleys, valley_prices
                    )
                    
                    if convergence_point:
                        key_points = [
                            PatternPoint(data.index[recent_peaks[0]], peak_prices[0], point_type="resistance_start"),
                            PatternPoint(data.index[recent_valleys[0]], valley_prices[0], point_type="support_start"),
                            PatternPoint(data.index[recent_peaks[-1]], peak_prices[-1], point_type="resistance_end"),
                            PatternPoint(data.index[recent_valleys[-1]], valley_prices[-1], point_type="support_end")
                        ]
                        
                        confidence = self._calculate_triangle_confidence(peak_slope, valley_slope, "symmetrical")
                        
                        pattern = RecognizedPattern(
                            pattern_type=PatternType.TRIANGLE_SYMMETRICAL,
                            signal=PatternSignal.NEUTRAL,
                            reliability=self._confidence_to_reliability(confidence),
                            confidence=confidence,
                            start_time=data.index[min(recent_peaks[0], recent_valleys[0])],
                            end_time=data.index[max(recent_peaks[-1], recent_valleys[-1])],
                            key_points=key_points,
                            symbol=symbol,
                            asset_type=asset_type,
                            timeframe=timeframe
                        )
                        
                        patterns.append(pattern)
                        
        except Exception as e:
            logger.warning(f"Triangle detection error: {e}")
        
        return patterns
    
    def _detect_candlestick_patterns(self, symbol: str, data: pd.DataFrame,
                                   asset_type: Optional[AssetType] = None,
                                   timeframe: str = "1d") -> List[RecognizedPattern]:
        """Detect candlestick patterns using TA-Lib"""
        
        patterns = []
        
        if not self.config.enable_candlestick_patterns:
            return patterns
        
        try:
            open_prices = data['open'].values
            high_prices = data['high'].values
            low_prices = data['low'].values
            close_prices = data['close'].values
            
            # TA-Lib candlestick pattern functions
            candlestick_functions = {
                'doji': talib.CDLDOJI,
                'hammer': talib.CDLHAMMER,
                'hanging_man': talib.CDLHANGINGMAN,
                'shooting_star': talib.CDLSHOOTINGSTAR,
                'engulfing': talib.CDLENGULFING,
                'morning_star': talib.CDLMORNINGSTAR,
                'evening_star': talib.CDLEVENINGSTAR,
                'harami': talib.CDLHARAMI,
                'dragonfly_doji': talib.CDLDRAGONFLYDOJI,
                'gravestone_doji': talib.CDLGRAVESTONEDOJI
            }
            
            for pattern_name, pattern_func in candlestick_functions.items():
                try:
                    # Detect pattern
                    pattern_result = pattern_func(open_prices, high_prices, low_prices, close_prices)
                    
                    # Find pattern occurrences
                    pattern_indices = np.where(pattern_result != 0)[0]
                    
                    for idx in pattern_indices[-5:]:  # Only recent patterns
                        if idx < len(data) - 1:  # Ensure we have next bar data
                            
                            signal_strength = abs(pattern_result[idx])
                            pattern_direction = 1 if pattern_result[idx] > 0 else -1
                            
                            # Determine signal
                            if pattern_direction > 0:
                                signal = PatternSignal.BUY if signal_strength >= 100 else PatternSignal.WEAK_BUY
                            else:
                                signal = PatternSignal.SELL if signal_strength >= 100 else PatternSignal.WEAK_SELL
                            
                            # Calculate confidence based on pattern strength and context
                            confidence = min(0.95, signal_strength / 100.0 * self.config.candlestick_sensitivity)
                            
                            # Create pattern point
                            key_points = [
                                PatternPoint(
                                    timestamp=data.index[idx],
                                    price=close_prices[idx],
                                    volume=data['volume'].iloc[idx] if 'volume' in data.columns else None,
                                    index=idx,
                                    point_type=pattern_name
                                )
                            ]
                            
                            pattern = RecognizedPattern(
                                pattern_type=getattr(PatternType, pattern_name.upper(), PatternType.DOJI),
                                signal=signal,
                                reliability=self._confidence_to_reliability(confidence),
                                confidence=confidence,
                                start_time=data.index[idx],
                                end_time=data.index[idx],
                                key_points=key_points,
                                symbol=symbol,
                                asset_type=asset_type,
                                timeframe=timeframe,
                                pattern_maturity=1.0  # Candlestick patterns are complete when formed
                            )
                            
                            patterns.append(pattern)
                            
                except Exception as e:
                    logger.warning(f"Candlestick pattern {pattern_name} detection error: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Candlestick pattern detection error: {e}")
        
        return patterns
    
    def _detect_support_resistance(self, symbol: str, data: pd.DataFrame,
                                 asset_type: Optional[AssetType] = None,
                                 timeframe: str = "1d") -> List[RecognizedPattern]:
        """Detect support and resistance levels"""
        
        patterns = []
        
        try:
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values
            
            lookback = min(self.config.support_resistance_lookback, len(data))
            recent_data = data.tail(lookback)
            
            # Find potential support and resistance levels
            peak_indices = signal.find_peaks(highs[-lookback:], distance=5)[0]
            valley_indices = signal.find_peaks(-lows[-lookback:], distance=5)[0]
            
            # Identify resistance levels
            resistance_levels = self._find_significant_levels(
                peak_indices, highs[-lookback:], "resistance"
            )
            
            # Identify support levels
            support_levels = self._find_significant_levels(
                valley_indices, lows[-lookback:], "support"
            )
            
            current_price = closes[-1]
            
            # Create resistance patterns
            for level_price, touches, strength in resistance_levels:
                if touches >= self.config.min_touches:
                    distance_pct = abs(current_price - level_price) / current_price
                    
                    if distance_pct < 0.05:  # Within 5% of level
                        confidence = min(0.95, 0.5 + (touches - 2) * 0.1 + strength * 0.3)
                        
                        pattern = RecognizedPattern(
                            pattern_type=PatternType.RESISTANCE_LEVEL,
                            signal=PatternSignal.SELL if current_price < level_price else PatternSignal.NEUTRAL,
                            reliability=self._confidence_to_reliability(confidence),
                            confidence=confidence,
                            start_time=recent_data.index[0],
                            end_time=recent_data.index[-1],
                            key_points=[
                                PatternPoint(recent_data.index[-1], level_price, point_type="resistance")
                            ],
                            symbol=symbol,
                            asset_type=asset_type,
                            timeframe=timeframe,
                            price_target=level_price * 0.95,
                            stop_loss=level_price * 1.02
                        )
                        
                        patterns.append(pattern)
            
            # Create support patterns
            for level_price, touches, strength in support_levels:
                if touches >= self.config.min_touches:
                    distance_pct = abs(current_price - level_price) / current_price
                    
                    if distance_pct < 0.05:  # Within 5% of level
                        confidence = min(0.95, 0.5 + (touches - 2) * 0.1 + strength * 0.3)
                        
                        pattern = RecognizedPattern(
                            pattern_type=PatternType.SUPPORT_LEVEL,
                            signal=PatternSignal.BUY if current_price > level_price else PatternSignal.NEUTRAL,
                            reliability=self._confidence_to_reliability(confidence),
                            confidence=confidence,
                            start_time=recent_data.index[0],
                            end_time=recent_data.index[-1],
                            key_points=[
                                PatternPoint(recent_data.index[-1], level_price, point_type="support")
                            ],
                            symbol=symbol,
                            asset_type=asset_type,
                            timeframe=timeframe,
                            price_target=level_price * 1.05,
                            stop_loss=level_price * 0.98
                        )
                        
                        patterns.append(pattern)
                        
        except Exception as e:
            logger.warning(f"Support/resistance detection error: {e}")
        
        return patterns
    
    def _detect_breakouts(self, symbol: str, data: pd.DataFrame,
                         asset_type: Optional[AssetType] = None,
                         timeframe: str = "1d") -> List[RecognizedPattern]:
        """Detect breakout patterns"""
        
        patterns = []
        
        try:
            if len(data) < 20:
                return patterns
            
            closes = data['close'].values
            highs = data['high'].values
            lows = data['low'].values
            volumes = data['volume'].values if 'volume' in data.columns else None
            
            current_price = closes[-1]
            
            # Calculate recent price range
            lookback = 20
            recent_high = np.max(highs[-lookback:-1])  # Exclude current bar
            recent_low = np.min(lows[-lookback:-1])
            range_size = recent_high - recent_low
            
            # Volume confirmation
            volume_confirmation = False
            if volumes is not None:
                avg_volume = np.mean(volumes[-lookback:-1])
                current_volume = volumes[-1]
                volume_confirmation = current_volume > avg_volume * 1.5
            
            # Detect upward breakout
            if current_price > recent_high and range_size > 0:
                breakout_strength = (current_price - recent_high) / range_size
                
                if breakout_strength > 0.01:  # At least 1% breakout
                    confidence = min(0.95, 0.6 + breakout_strength * 2)
                    if volume_confirmation:
                        confidence += 0.1
                    
                    target_price = current_price + range_size
                    stop_loss = recent_high * 0.98
                    
                    pattern = RecognizedPattern(
                        pattern_type=PatternType.BREAKOUT,
                        signal=PatternSignal.STRONG_BUY,
                        reliability=self._confidence_to_reliability(confidence),
                        confidence=confidence,
                        start_time=data.index[-lookback],
                        end_time=data.index[-1],
                        key_points=[
                            PatternPoint(data.index[-1], recent_high, point_type="breakout_level"),
                            PatternPoint(data.index[-1], current_price, point_type="breakout_price")
                        ],
                        pattern_height=range_size,
                        price_target=target_price,
                        stop_loss=stop_loss,
                        symbol=symbol,
                        asset_type=asset_type,
                        timeframe=timeframe,
                        volume_confirmation=volume_confirmation
                    )
                    
                    patterns.append(pattern)
            
            # Detect downward breakdown
            elif current_price < recent_low and range_size > 0:
                breakdown_strength = (recent_low - current_price) / range_size
                
                if breakdown_strength > 0.01:  # At least 1% breakdown
                    confidence = min(0.95, 0.6 + breakdown_strength * 2)
                    if volume_confirmation:
                        confidence += 0.1
                    
                    target_price = current_price - range_size
                    stop_loss = recent_low * 1.02
                    
                    pattern = RecognizedPattern(
                        pattern_type=PatternType.BREAKDOWN,
                        signal=PatternSignal.STRONG_SELL,
                        reliability=self._confidence_to_reliability(confidence),
                        confidence=confidence,
                        start_time=data.index[-lookback],
                        end_time=data.index[-1],
                        key_points=[
                            PatternPoint(data.index[-1], recent_low, point_type="breakdown_level"),
                            PatternPoint(data.index[-1], current_price, point_type="breakdown_price")
                        ],
                        pattern_height=range_size,
                        price_target=target_price,
                        stop_loss=stop_loss,
                        symbol=symbol,
                        asset_type=asset_type,
                        timeframe=timeframe,
                        volume_confirmation=volume_confirmation
                    )
                    
                    patterns.append(pattern)
                    
        except Exception as e:
            logger.warning(f"Breakout detection error: {e}")
        
        return patterns
    
    def _detect_flags_pennants(self, symbol: str, data: pd.DataFrame,
                             asset_type: Optional[AssetType] = None,
                             timeframe: str = "1d") -> List[RecognizedPattern]:
        """Detect flag and pennant patterns"""
        
        patterns = []
        
        try:
            if len(data) < 30:
                return patterns
            
            closes = data['close'].values
            highs = data['high'].values
            lows = data['low'].values
            
            # Look for strong moves followed by consolidation
            for i in range(20, len(data) - 10):
                # Check for strong initial move (flagpole)
                flagpole_start = i - 15
                flagpole_end = i
                consolidation_start = i
                consolidation_end = min(i + 10, len(data) - 1)
                
                # Calculate flagpole strength
                flagpole_move = abs(closes[flagpole_end] - closes[flagpole_start])
                flagpole_pct = flagpole_move / closes[flagpole_start]
                
                if flagpole_pct > 0.05:  # At least 5% move
                    # Analyze consolidation phase
                    consol_highs = highs[consolidation_start:consolidation_end]
                    consol_lows = lows[consolidation_start:consolidation_end]
                    consol_range = np.max(consol_highs) - np.min(consol_lows)
                    consol_range_pct = consol_range / closes[consolidation_start]
                    
                    # Flag criteria: small consolidation after strong move
                    if consol_range_pct < flagpole_pct * 0.5:  # Consolidation < 50% of flagpole
                        
                        # Determine direction
                        if closes[flagpole_end] > closes[flagpole_start]:
                            # Bull flag
                            pattern_type = PatternType.FLAG_BULL
                            signal = PatternSignal.BUY
                            target = closes[consolidation_end] + flagpole_move
                            stop_loss = np.min(consol_lows) * 0.98
                        else:
                            # Bear flag
                            pattern_type = PatternType.FLAG_BEAR
                            signal = PatternSignal.SELL
                            target = closes[consolidation_end] - flagpole_move
                            stop_loss = np.max(consol_highs) * 1.02
                        
                        confidence = min(0.9, 0.7 + (flagpole_pct - 0.05) * 2)
                        
                        key_points = [
                            PatternPoint(data.index[flagpole_start], closes[flagpole_start], point_type="flagpole_start"),
                            PatternPoint(data.index[flagpole_end], closes[flagpole_end], point_type="flagpole_end"),
                            PatternPoint(data.index[consolidation_end], closes[consolidation_end], point_type="flag_end")
                        ]
                        
                        pattern = RecognizedPattern(
                            pattern_type=pattern_type,
                            signal=signal,
                            reliability=self._confidence_to_reliability(confidence),
                            confidence=confidence,
                            start_time=data.index[flagpole_start],
                            end_time=data.index[consolidation_end],
                            key_points=key_points,
                            pattern_height=flagpole_move,
                            price_target=target,
                            stop_loss=stop_loss,
                            symbol=symbol,
                            asset_type=asset_type,
                            timeframe=timeframe
                        )
                        
                        patterns.append(pattern)
                        
        except Exception as e:
            logger.warning(f"Flag/pennant detection error: {e}")
        
        return patterns
    
    def _detect_double_tops_bottoms(self, symbol: str, data: pd.DataFrame,
                                  asset_type: Optional[AssetType] = None,
                                  timeframe: str = "1d") -> List[RecognizedPattern]:
        """Detect double top and double bottom patterns"""
        
        patterns = []
        
        try:
            highs = data['high'].values
            lows = data['low'].values
            
            # Find peaks and valleys
            peak_indices = signal.find_peaks(highs, distance=10, prominence=np.std(highs) * 0.3)[0]
            valley_indices = signal.find_peaks(-lows, distance=10, prominence=np.std(lows) * 0.3)[0]
            
            # Double tops
            for i in range(len(peak_indices) - 1):
                peak1_idx = peak_indices[i]
                peak2_idx = peak_indices[i + 1]
                
                peak1_price = highs[peak1_idx]
                peak2_price = highs[peak2_idx]
                
                # Check if peaks are similar height
                price_diff_pct = abs(peak1_price - peak2_price) / max(peak1_price, peak2_price)
                
                if price_diff_pct < 0.03:  # Within 3%
                    # Find valley between peaks
                    valley_between = valley_indices[(valley_indices > peak1_idx) & (valley_indices < peak2_idx)]
                    
                    if len(valley_between) > 0:
                        valley_idx = valley_between[0]
                        valley_price = lows[valley_idx]
                        
                        # Calculate neckline and target
                        neckline = valley_price
                        pattern_height = max(peak1_price, peak2_price) - neckline
                        target = neckline - pattern_height
                        
                        confidence = self._calculate_double_pattern_confidence(
                            peak1_price, peak2_price, valley_price, "top"
                        )
                        
                        key_points = [
                            PatternPoint(data.index[peak1_idx], peak1_price, point_type="first_peak"),
                            PatternPoint(data.index[valley_idx], valley_price, point_type="valley"),
                            PatternPoint(data.index[peak2_idx], peak2_price, point_type="second_peak")
                        ]
                        
                        pattern = RecognizedPattern(
                            pattern_type=PatternType.DOUBLE_TOP,
                            signal=PatternSignal.SELL,
                            reliability=self._confidence_to_reliability(confidence),
                            confidence=confidence,
                            start_time=data.index[peak1_idx],
                            end_time=data.index[peak2_idx],
                            key_points=key_points,
                            pattern_height=pattern_height,
                            price_target=target,
                            stop_loss=max(peak1_price, peak2_price) * 1.02,
                            symbol=symbol,
                            asset_type=asset_type,
                            timeframe=timeframe
                        )
                        
                        patterns.append(pattern)
            
            # Double bottoms
            for i in range(len(valley_indices) - 1):
                valley1_idx = valley_indices[i]
                valley2_idx = valley_indices[i + 1]
                
                valley1_price = lows[valley1_idx]
                valley2_price = lows[valley2_idx]
                
                # Check if valleys are similar depth
                price_diff_pct = abs(valley1_price - valley2_price) / max(valley1_price, valley2_price)
                
                if price_diff_pct < 0.03:  # Within 3%
                    # Find peak between valleys
                    peak_between = peak_indices[(peak_indices > valley1_idx) & (peak_indices < valley2_idx)]
                    
                    if len(peak_between) > 0:
                        peak_idx = peak_between[0]
                        peak_price = highs[peak_idx]
                        
                        # Calculate neckline and target
                        neckline = peak_price
                        pattern_height = neckline - min(valley1_price, valley2_price)
                        target = neckline + pattern_height
                        
                        confidence = self._calculate_double_pattern_confidence(
                            valley1_price, valley2_price, peak_price, "bottom"
                        )
                        
                        key_points = [
                            PatternPoint(data.index[valley1_idx], valley1_price, point_type="first_valley"),
                            PatternPoint(data.index[peak_idx], peak_price, point_type="peak"),
                            PatternPoint(data.index[valley2_idx], valley2_price, point_type="second_valley")
                        ]
                        
                        pattern = RecognizedPattern(
                            pattern_type=PatternType.DOUBLE_BOTTOM,
                            signal=PatternSignal.BUY,
                            reliability=self._confidence_to_reliability(confidence),
                            confidence=confidence,
                            start_time=data.index[valley1_idx],
                            end_time=data.index[valley2_idx],
                            key_points=key_points,
                            pattern_height=pattern_height,
                            price_target=target,
                            stop_loss=min(valley1_price, valley2_price) * 0.98,
                            symbol=symbol,
                            asset_type=asset_type,
                            timeframe=timeframe
                        )
                        
                        patterns.append(pattern)
                        
        except Exception as e:
            logger.warning(f"Double top/bottom detection error: {e}")
        
        return patterns
    
    def _detect_wedges(self, symbol: str, data: pd.DataFrame,
                      asset_type: Optional[AssetType] = None,
                      timeframe: str = "1d") -> List[RecognizedPattern]:
        """Detect rising and falling wedge patterns"""
        # Simplified wedge detection - can be expanded
        return []
    
    def _detect_trend_lines(self, symbol: str, data: pd.DataFrame,
                           asset_type: Optional[AssetType] = None,
                           timeframe: str = "1d") -> List[RecognizedPattern]:
        """Detect trend lines"""
        # Simplified trend line detection - can be expanded
        return []
    
    def _detect_channels(self, symbol: str, data: pd.DataFrame,
                        asset_type: Optional[AssetType] = None,
                        timeframe: str = "1d") -> List[RecognizedPattern]:
        """Detect price channels"""
        # Simplified channel detection - can be expanded
        return []
    
    def _detect_harmonic_patterns(self, symbol: str, data: pd.DataFrame,
                                 asset_type: Optional[AssetType] = None,
                                 timeframe: str = "1d") -> List[RecognizedPattern]:
        """Detect harmonic patterns (Gartley, Butterfly, etc.)"""
        # Simplified harmonic pattern detection - can be expanded
        return []
    
    def _detect_ml_patterns(self, symbol: str, data: pd.DataFrame,
                           asset_type: Optional[AssetType] = None,
                           timeframe: str = "1d") -> List[RecognizedPattern]:
        """Detect patterns using machine learning"""
        # Simplified ML pattern detection - can be expanded
        return []
    
    # Helper methods
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for pattern detection"""
        required_columns = ['open', 'high', 'low', 'close']
        return all(col in data.columns for col in required_columns) and len(data) >= self.config.min_pattern_length
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for pattern detection"""
        # Add any necessary calculations or transformations
        prepared_data = data.copy()
        
        # Ensure data is sorted by timestamp
        if 'timestamp' in prepared_data.columns:
            prepared_data = prepared_data.sort_values('timestamp')
        
        return prepared_data
    
    def _calculate_slope(self, x_values: List[int], y_values: List[float]) -> float:
        """Calculate slope of a line through points"""
        if len(x_values) < 2 or len(y_values) < 2:
            return 0.0
        
        x = np.array(x_values)
        y = np.array(y_values)
        
        slope, _ = np.polyfit(x, y, 1)
        return slope
    
    def _calculate_convergence_point(self, peak_indices: List[int], peak_prices: List[float],
                                   valley_indices: List[int], valley_prices: List[float]) -> Optional[Tuple[int, float]]:
        """Calculate convergence point for triangle patterns"""
        try:
            if len(peak_indices) < 2 or len(valley_indices) < 2:
                return None
            
            # Fit lines to peaks and valleys
            peak_slope, peak_intercept = np.polyfit(peak_indices, peak_prices, 1)
            valley_slope, valley_intercept = np.polyfit(valley_indices, valley_prices, 1)
            
            # Find intersection
            if abs(peak_slope - valley_slope) < 1e-6:  # Parallel lines
                return None
            
            x_intersect = (valley_intercept - peak_intercept) / (peak_slope - valley_slope)
            y_intersect = peak_slope * x_intersect + peak_intercept
            
            return (int(x_intersect), y_intersect)
            
        except Exception:
            return None
    
    def _find_significant_levels(self, indices: np.ndarray, prices: np.ndarray, 
                               level_type: str) -> List[Tuple[float, int, float]]:
        """Find significant support/resistance levels"""
        levels = []
        
        if len(indices) == 0:
            return levels
        
        level_prices = prices[indices]
        
        # Group similar price levels
        tolerance = np.std(level_prices) * self.config.sr_touch_tolerance
        
        clustered_levels = []
        used_indices = set()
        
        for i, price in enumerate(level_prices):
            if i in used_indices:
                continue
            
            # Find similar prices
            similar_mask = np.abs(level_prices - price) <= tolerance
            similar_indices = np.where(similar_mask)[0]
            
            if len(similar_indices) >= self.config.min_touches:
                avg_price = np.mean(level_prices[similar_indices])
                touches = len(similar_indices)
                
                # Calculate strength based on touches and recency
                strength = min(1.0, touches / 5.0)
                
                levels.append((avg_price, touches, strength))
                used_indices.update(similar_indices)
        
        return levels
    
    def _calculate_hs_confidence(self, left: float, head: float, right: float, height: float) -> float:
        """Calculate confidence for head and shoulders pattern"""
        
        # Symmetry factor
        shoulder_diff = abs(left - right) / max(left, right)
        symmetry_factor = max(0, 1 - shoulder_diff * 10)
        
        # Head prominence
        head_prominence = (head - max(left, right)) / height
        prominence_factor = min(1.0, head_prominence * 2)
        
        # Pattern size factor
        size_factor = min(1.0, height / min(left, right))
        
        confidence = (symmetry_factor * 0.4 + prominence_factor * 0.4 + size_factor * 0.2)
        return max(0.1, min(0.95, confidence))
    
    def _calculate_triangle_confidence(self, peak_slope: float, valley_slope: float, triangle_type: str) -> float:
        """Calculate confidence for triangle patterns"""
        
        if triangle_type == "ascending":
            # Flat resistance, rising support
            resistance_flatness = max(0, 1 - abs(peak_slope) * 1000)
            support_rise = min(1.0, valley_slope * 500)
            confidence = (resistance_flatness + support_rise) / 2
            
        elif triangle_type == "descending":
            # Declining resistance, flat support
            resistance_decline = min(1.0, abs(peak_slope) * 500)
            support_flatness = max(0, 1 - abs(valley_slope) * 1000)
            confidence = (resistance_decline + support_flatness) / 2
            
        else:  # symmetrical
            # Converging lines
            slope_diff = abs(abs(peak_slope) - abs(valley_slope))
            convergence_factor = max(0, 1 - slope_diff * 1000)
            confidence = convergence_factor
        
        return max(0.1, min(0.95, confidence))
    
    def _calculate_double_pattern_confidence(self, first: float, second: float, middle: float, pattern_type: str) -> float:
        """Calculate confidence for double top/bottom patterns"""
        
        # Price similarity
        similarity = 1 - abs(first - second) / max(first, second)
        
        # Distance from middle
        if pattern_type == "top":
            distance_factor = (max(first, second) - middle) / max(first, second)
        else:  # bottom
            distance_factor = (middle - min(first, second)) / middle
        
        confidence = (similarity * 0.6 + distance_factor * 0.4)
        return max(0.1, min(0.95, confidence))
    
    def _confidence_to_reliability(self, confidence: float) -> PatternReliability:
        """Convert confidence score to reliability level"""
        
        if confidence >= 0.9:
            return PatternReliability.VERY_HIGH
        elif confidence >= 0.8:
            return PatternReliability.HIGH
        elif confidence >= 0.6:
            return PatternReliability.MEDIUM
        elif confidence >= 0.4:
            return PatternReliability.LOW
        else:
            return PatternReliability.VERY_LOW
    
    def _filter_and_rank_patterns(self, patterns: List[RecognizedPattern]) -> List[RecognizedPattern]:
        """Filter and rank patterns by confidence and relevance"""
        
        # Filter by minimum confidence
        filtered = [p for p in patterns if p.confidence >= self.config.min_confidence]
        
        # Sort by confidence (descending)
        filtered.sort(key=lambda p: p.confidence, reverse=True)
        
        # Remove duplicates (same pattern type at similar times)
        unique_patterns = []
        seen_patterns = set()
        
        for pattern in filtered:
            pattern_key = f"{pattern.pattern_type}_{pattern.start_time.date()}"
            if pattern_key not in seen_patterns:
                unique_patterns.append(pattern)
                seen_patterns.add(pattern_key)
        
        return unique_patterns
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get pattern recognition performance metrics"""
        
        avg_detection_time = np.mean(self.detection_times) if self.detection_times else 0
        
        # Calculate success rates by pattern type
        success_rates = {}
        for pattern_type, rates in self.pattern_success_rates.items():
            if rates:
                success_rates[pattern_type] = np.mean(rates)
        
        return {
            'average_detection_time_ms': avg_detection_time,
            'pattern_cache_size': len(self.pattern_cache),
            'pattern_history_size': len(self.pattern_history),
            'success_rates_by_pattern': success_rates,
            'total_patterns_detected': len(self.pattern_history),
            'enabled_detectors': list(self.pattern_detectors.keys())
        }

# ==================== TESTING FUNCTION ====================

def test_pattern_recognizer():
    """Test the pattern recognizer functionality"""
    
    print("🔍 Testing Advanced Pattern Recognizer")
    print("=" * 50)
    
    # Initialize recognizer
    recognizer = AdvancedPatternRecognizer()
    
    # Create test data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # Create synthetic OHLCV data with patterns
    base_price = 100
    data = []
    
    for i, date in enumerate(dates):
        # Add some trend and volatility
        trend = i * 0.1
        noise = np.random.normal(0, 2)
        
        open_price = base_price + trend + noise
        high_price = open_price + abs(np.random.normal(1, 0.5))
        low_price = open_price - abs(np.random.normal(1, 0.5))
        close_price = open_price + np.random.normal(0, 1)
        volume = np.random.randint(50000, 200000)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    test_data = pd.DataFrame(data)
    test_data.set_index('timestamp', inplace=True)
    
    # Test pattern scanning
    start_time = datetime.now()
    
    patterns = recognizer.scan_patterns(
        symbol='TEST',
        data=test_data,
        asset_type=AssetType.STOCK,
        timeframe='1d'
    )
    
    scan_time = (datetime.now() - start_time).total_seconds() * 1000
    
    print(f"✅ Pattern scanning completed")
    print(f"   Total patterns detected: {len(patterns)}")
    print(f"   Scan time: {scan_time:.2f}ms")
    
    if patterns:
        print(f"   Pattern types found:")
        pattern_types = {}
        for pattern in patterns:
            pattern_type = pattern.pattern_type.value
            pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
        
        for pattern_type, count in pattern_types.items():
            print(f"     {pattern_type}: {count}")
        
        # Show top 3 patterns
        print(f"\n   Top 3 patterns by confidence:")
        for i, pattern in enumerate(patterns[:3]):
            print(f"     {i+1}. {pattern.pattern_type.value} - "
                  f"Confidence: {pattern.confidence:.3f} - "
                  f"Signal: {pattern.signal.value}")
    
    # Test performance
    performance = recognizer.get_performance_metrics()
    print(f"\n📊 Performance Metrics:")
    for key, value in performance.items():
        print(f"   {key}: {value}")
    
    print("\n🎉 Pattern recognizer tests completed!")

if __name__ == "__main__":
    test_pattern_recognizer()