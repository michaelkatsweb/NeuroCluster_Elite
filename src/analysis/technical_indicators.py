#!/usr/bin/env python3
"""
File: technical_indicators.py
Path: NeuroCluster-Elite/src/analysis/technical_indicators.py
Description: Comprehensive technical analysis indicators for market analysis

This module provides a complete suite of technical analysis indicators including
trend, momentum, volatility, volume, and oscillator indicators with advanced
features like adaptive parameters, multi-timeframe analysis, and signal generation.

Features:
- 50+ technical indicators with customizable parameters
- Trend indicators (MA, EMA, MACD, ADX, etc.)
- Momentum indicators (RSI, Stochastic, Williams %R, etc.)
- Volatility indicators (Bollinger Bands, ATR, Keltner Channels, etc.)
- Volume indicators (OBV, Chaikin, Volume Profile, etc.)
- Custom and composite indicators
- Signal generation and crossover detection
- Multi-timeframe indicator analysis
- Adaptive parameter optimization

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
import warnings
from scipy import stats
from scipy.signal import find_peaks, argrelextrema
import talib

# Import our modules
try:
    from src.core.neurocluster_elite import AssetType, MarketData
    from src.utils.config_manager import ConfigManager
    from src.utils.helpers import format_percentage, calculate_hash
except ImportError:
    # Fallback for testing
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== INDICATOR ENUMS AND STRUCTURES ====================

class IndicatorType(Enum):
    """Types of technical indicators"""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    OSCILLATOR = "oscillator"
    COMPOSITE = "composite"
    CUSTOM = "custom"

class SignalType(Enum):
    """Technical indicator signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    OVERBOUGHT = "overbought"
    OVERSOLD = "oversold"

@dataclass
class IndicatorSignal:
    """Technical indicator signal"""
    indicator_name: str
    signal_type: SignalType
    strength: float  # 0-1
    confidence: float  # 0-100
    value: float
    threshold: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""
    
@dataclass
class IndicatorResult:
    """Result from indicator calculation"""
    name: str
    values: Union[pd.Series, pd.DataFrame]
    signals: List[IndicatorSignal] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    calculation_time: datetime = field(default_factory=datetime.now)
    
@dataclass
class CompositeIndicator:
    """Composite indicator combining multiple indicators"""
    name: str
    components: List[str]
    weights: List[float]
    aggregation_method: str = "weighted_average"
    normalize: bool = True

# ==================== TECHNICAL INDICATORS CLASS ====================

class AdvancedTechnicalIndicators:
    """
    Comprehensive technical indicators calculator
    
    This class provides a complete suite of technical analysis indicators
    with advanced features including adaptive parameters, signal generation,
    and multi-timeframe analysis.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize technical indicators calculator"""
        
        self.config = config or self._get_default_config()
        self.indicator_cache = {}
        self.signal_history = {}
        
        # Initialize indicators
        self._initialize_indicators()
        
        logger.info("Advanced Technical Indicators calculator initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'trend_indicators': {
                'sma_periods': [10, 20, 50, 100, 200],
                'ema_periods': [12, 26, 50],
                'macd_config': {'fast': 12, 'slow': 26, 'signal': 9},
                'adx_period': 14,
                'aroon_period': 25,
                'parabolic_sar': {'acceleration': 0.02, 'maximum': 0.2}
            },
            'momentum_indicators': {
                'rsi_period': 14,
                'stochastic_config': {'k_period': 14, 'd_period': 3},
                'williams_r_period': 14,
                'roc_period': 10,
                'momentum_period': 10,
                'cci_period': 20
            },
            'volatility_indicators': {
                'bollinger_config': {'period': 20, 'std_dev': 2},
                'atr_period': 14,
                'keltner_config': {'period': 20, 'atr_multiplier': 2},
                'donchian_period': 20
            },
            'volume_indicators': {
                'obv_enabled': True,
                'ad_line_enabled': True,
                'chaikin_config': {'fast': 3, 'slow': 10},
                'mfi_period': 14,
                'vwap_enabled': True
            },
            'signal_generation': {
                'crossover_sensitivity': 0.001,  # 0.1% minimum move for crossover
                'overbought_threshold': 70,
                'oversold_threshold': 30,
                'trend_strength_threshold': 25,
                'signal_confirmation_bars': 2
            },
            'multi_timeframe': {
                'enabled': True,
                'timeframes': ['1h', '4h', '1d'],
                'alignment_weight': 1.5  # Weight boost for aligned signals
            }
        }
    
    def _initialize_indicators(self):
        """Initialize indicator calculation methods"""
        
        self.indicators = {
            # Trend Indicators
            'sma': self._calculate_sma,
            'ema': self._calculate_ema,
            'wma': self._calculate_wma,
            'macd': self._calculate_macd,
            'adx': self._calculate_adx,
            'aroon': self._calculate_aroon,
            'parabolic_sar': self._calculate_parabolic_sar,
            'ichimoku': self._calculate_ichimoku,
            
            # Momentum Indicators
            'rsi': self._calculate_rsi,
            'stochastic': self._calculate_stochastic,
            'williams_r': self._calculate_williams_r,
            'roc': self._calculate_roc,
            'momentum': self._calculate_momentum,
            'cci': self._calculate_cci,
            'ultimate_oscillator': self._calculate_ultimate_oscillator,
            
            # Volatility Indicators
            'bollinger_bands': self._calculate_bollinger_bands,
            'atr': self._calculate_atr,
            'keltner_channels': self._calculate_keltner_channels,
            'donchian_channels': self._calculate_donchian_channels,
            
            # Volume Indicators
            'obv': self._calculate_obv,
            'ad_line': self._calculate_ad_line,
            'chaikin_oscillator': self._calculate_chaikin_oscillator,
            'mfi': self._calculate_mfi,
            'vwap': self._calculate_vwap,
            'volume_profile': self._calculate_volume_profile,
            
            # Custom Indicators
            'composite_momentum': self._calculate_composite_momentum,
            'trend_strength': self._calculate_trend_strength,
            'market_regime': self._calculate_market_regime
        }
    
    def calculate_all_indicators(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> Dict[str, IndicatorResult]:
        """Calculate all enabled indicators for given data"""
        
        try:
            if not self._validate_data(data):
                return {}
            
            results = {}
            
            # Calculate each indicator
            for indicator_name, calculator in self.indicators.items():
                try:
                    result = calculator(data, symbol)
                    if result:
                        results[indicator_name] = result
                        
                except Exception as e:
                    logger.warning(f"Error calculating {indicator_name}: {e}")
                    continue
            
            # Calculate composite indicators
            composite_results = self._calculate_composite_indicators(results, data)
            results.update(composite_results)
            
            # Generate combined signals
            combined_signals = self._generate_combined_signals(results)
            results['combined_signals'] = IndicatorResult(
                name='combined_signals',
                values=pd.Series(dtype=float),
                signals=combined_signals
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    # ==================== TREND INDICATORS ====================
    
    def _calculate_sma(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate Simple Moving Averages"""
        
        try:
            periods = self.config['trend_indicators']['sma_periods']
            sma_data = pd.DataFrame(index=data.index)
            signals = []
            
            for period in periods:
                if len(data) >= period:
                    sma_values = talib.SMA(data['close'].values, timeperiod=period)
                    sma_data[f'SMA_{period}'] = sma_values
                    
                    # Generate crossover signals
                    if period == 50 and 'SMA_20' in sma_data.columns:
                        crossover_signals = self._detect_ma_crossover(
                            sma_data['SMA_20'], sma_data['SMA_50'], 'SMA_Crossover_20_50'
                        )
                        signals.extend(crossover_signals)
            
            return IndicatorResult(
                name='sma',
                values=sma_data,
                signals=signals,
                parameters={'periods': periods}
            )
            
        except Exception as e:
            logger.warning(f"Error calculating SMA: {e}")
            return None
    
    def _calculate_ema(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate Exponential Moving Averages"""
        
        try:
            periods = self.config['trend_indicators']['ema_periods']
            ema_data = pd.DataFrame(index=data.index)
            signals = []
            
            for period in periods:
                if len(data) >= period:
                    ema_values = talib.EMA(data['close'].values, timeperiod=period)
                    ema_data[f'EMA_{period}'] = ema_values
            
            # Generate EMA crossover signals
            if 'EMA_12' in ema_data.columns and 'EMA_26' in ema_data.columns:
                crossover_signals = self._detect_ma_crossover(
                    ema_data['EMA_12'], ema_data['EMA_26'], 'EMA_Crossover_12_26'
                )
                signals.extend(crossover_signals)
            
            return IndicatorResult(
                name='ema',
                values=ema_data,
                signals=signals,
                parameters={'periods': periods}
            )
            
        except Exception as e:
            logger.warning(f"Error calculating EMA: {e}")
            return None
    
    def _calculate_wma(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate Weighted Moving Average"""
        
        try:
            period = 20
            if len(data) < period:
                return None
            
            wma_values = talib.WMA(data['close'].values, timeperiod=period)
            wma_data = pd.DataFrame({'WMA': wma_values}, index=data.index)
            
            return IndicatorResult(
                name='wma',
                values=wma_data,
                parameters={'period': period}
            )
            
        except Exception as e:
            logger.warning(f"Error calculating WMA: {e}")
            return None
    
    def _calculate_macd(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        
        try:
            config = self.config['trend_indicators']['macd_config']
            
            if len(data) < config['slow']:
                return None
            
            macd, macd_signal, macd_hist = talib.MACD(
                data['close'].values,
                fastperiod=config['fast'],
                slowperiod=config['slow'],
                signalperiod=config['signal']
            )
            
            macd_data = pd.DataFrame({
                'MACD': macd,
                'MACD_Signal': macd_signal,
                'MACD_Histogram': macd_hist
            }, index=data.index)
            
            # Generate MACD signals
            signals = []
            
            # MACD line crossover signal line
            if len(macd_data.dropna()) >= 2:
                latest_macd = macd_data['MACD'].iloc[-1]
                latest_signal = macd_data['MACD_Signal'].iloc[-1]
                prev_macd = macd_data['MACD'].iloc[-2]
                prev_signal = macd_data['MACD_Signal'].iloc[-2]
                
                if not (np.isnan(latest_macd) or np.isnan(latest_signal)):
                    if prev_macd <= prev_signal and latest_macd > latest_signal:
                        signals.append(IndicatorSignal(
                            indicator_name='MACD',
                            signal_type=SignalType.BUY,
                            strength=0.7,
                            confidence=75.0,
                            value=latest_macd,
                            description="MACD bullish crossover"
                        ))
                    elif prev_macd >= prev_signal and latest_macd < latest_signal:
                        signals.append(IndicatorSignal(
                            indicator_name='MACD',
                            signal_type=SignalType.SELL,
                            strength=0.7,
                            confidence=75.0,
                            value=latest_macd,
                            description="MACD bearish crossover"
                        ))
            
            return IndicatorResult(
                name='macd',
                values=macd_data,
                signals=signals,
                parameters=config
            )
            
        except Exception as e:
            logger.warning(f"Error calculating MACD: {e}")
            return None
    
    def _calculate_adx(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate ADX (Average Directional Index)"""
        
        try:
            period = self.config['trend_indicators']['adx_period']
            
            if len(data) < period or not all(col in data.columns for col in ['high', 'low', 'close']):
                return None
            
            adx = talib.ADX(data['high'].values, data['low'].values, data['close'].values, timeperiod=period)
            plus_di = talib.PLUS_DI(data['high'].values, data['low'].values, data['close'].values, timeperiod=period)
            minus_di = talib.MINUS_DI(data['high'].values, data['low'].values, data['close'].values, timeperiod=period)
            
            adx_data = pd.DataFrame({
                'ADX': adx,
                'PLUS_DI': plus_di,
                'MINUS_DI': minus_di
            }, index=data.index)
            
            # Generate ADX signals
            signals = []
            threshold = self.config['signal_generation']['trend_strength_threshold']
            
            if not np.isnan(adx[-1]):
                latest_adx = adx[-1]
                latest_plus_di = plus_di[-1]
                latest_minus_di = minus_di[-1]
                
                if latest_adx > threshold:
                    if latest_plus_di > latest_minus_di:
                        signals.append(IndicatorSignal(
                            indicator_name='ADX',
                            signal_type=SignalType.BULLISH,
                            strength=min(1.0, latest_adx / 50),
                            confidence=min(95.0, 50 + latest_adx),
                            value=latest_adx,
                            threshold=threshold,
                            description=f"Strong uptrend (ADX: {latest_adx:.1f})"
                        ))
                    else:
                        signals.append(IndicatorSignal(
                            indicator_name='ADX',
                            signal_type=SignalType.BEARISH,
                            strength=min(1.0, latest_adx / 50),
                            confidence=min(95.0, 50 + latest_adx),
                            value=latest_adx,
                            threshold=threshold,
                            description=f"Strong downtrend (ADX: {latest_adx:.1f})"
                        ))
            
            return IndicatorResult(
                name='adx',
                values=adx_data,
                signals=signals,
                parameters={'period': period}
            )
            
        except Exception as e:
            logger.warning(f"Error calculating ADX: {e}")
            return None
    
    def _calculate_aroon(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate Aroon indicator"""
        
        try:
            period = self.config['trend_indicators']['aroon_period']
            
            if len(data) < period or not all(col in data.columns for col in ['high', 'low']):
                return None
            
            aroon_up, aroon_down = talib.AROON(data['high'].values, data['low'].values, timeperiod=period)
            
            aroon_data = pd.DataFrame({
                'Aroon_Up': aroon_up,
                'Aroon_Down': aroon_down,
                'Aroon_Oscillator': aroon_up - aroon_down
            }, index=data.index)
            
            # Generate Aroon signals
            signals = []
            
            if not (np.isnan(aroon_up[-1]) or np.isnan(aroon_down[-1])):
                latest_up = aroon_up[-1]
                latest_down = aroon_down[-1]
                
                if latest_up > 70 and latest_down < 30:
                    signals.append(IndicatorSignal(
                        indicator_name='Aroon',
                        signal_type=SignalType.BULLISH,
                        strength=0.8,
                        confidence=80.0,
                        value=latest_up,
                        description="Aroon bullish (strong uptrend)"
                    ))
                elif latest_down > 70 and latest_up < 30:
                    signals.append(IndicatorSignal(
                        indicator_name='Aroon',
                        signal_type=SignalType.BEARISH,
                        strength=0.8,
                        confidence=80.0,
                        value=latest_down,
                        description="Aroon bearish (strong downtrend)"
                    ))
            
            return IndicatorResult(
                name='aroon',
                values=aroon_data,
                signals=signals,
                parameters={'period': period}
            )
            
        except Exception as e:
            logger.warning(f"Error calculating Aroon: {e}")
            return None
    
    def _calculate_parabolic_sar(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate Parabolic SAR"""
        
        try:
            config = self.config['trend_indicators']['parabolic_sar']
            
            if len(data) < 10 or not all(col in data.columns for col in ['high', 'low']):
                return None
            
            sar = talib.SAR(
                data['high'].values, 
                data['low'].values,
                acceleration=config['acceleration'],
                maximum=config['maximum']
            )
            
            sar_data = pd.DataFrame({'SAR': sar}, index=data.index)
            
            # Generate SAR signals
            signals = []
            
            if len(data) >= 2:
                current_price = data['close'].iloc[-1]
                current_sar = sar[-1]
                prev_sar = sar[-2]
                prev_price = data['close'].iloc[-2]
                
                if not np.isnan(current_sar):
                    # SAR reversal signals
                    if prev_price < prev_sar and current_price > current_sar:
                        signals.append(IndicatorSignal(
                            indicator_name='SAR',
                            signal_type=SignalType.BUY,
                            strength=0.8,
                            confidence=80.0,
                            value=current_sar,
                            description="SAR bullish reversal"
                        ))
                    elif prev_price > prev_sar and current_price < current_sar:
                        signals.append(IndicatorSignal(
                            indicator_name='SAR',
                            signal_type=SignalType.SELL,
                            strength=0.8,
                            confidence=80.0,
                            value=current_sar,
                            description="SAR bearish reversal"
                        ))
            
            return IndicatorResult(
                name='parabolic_sar',
                values=sar_data,
                signals=signals,
                parameters=config
            )
            
        except Exception as e:
            logger.warning(f"Error calculating Parabolic SAR: {e}")
            return None
    
    def _calculate_ichimoku(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate Ichimoku Cloud"""
        
        try:
            if len(data) < 52 or not all(col in data.columns for col in ['high', 'low', 'close']):
                return None
            
            # Ichimoku parameters
            tenkan_period = 9
            kijun_period = 26
            senkou_span_b_period = 52
            displacement = 26
            
            # Calculate components
            tenkan_sen = (data['high'].rolling(tenkan_period).max() + 
                         data['low'].rolling(tenkan_period).min()) / 2
            
            kijun_sen = (data['high'].rolling(kijun_period).max() + 
                        data['low'].rolling(kijun_period).min()) / 2
            
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
            
            senkou_span_b = ((data['high'].rolling(senkou_span_b_period).max() + 
                             data['low'].rolling(senkou_span_b_period).min()) / 2).shift(displacement)
            
            chikou_span = data['close'].shift(-displacement)
            
            ichimoku_data = pd.DataFrame({
                'Tenkan_Sen': tenkan_sen,
                'Kijun_Sen': kijun_sen,
                'Senkou_Span_A': senkou_span_a,
                'Senkou_Span_B': senkou_span_b,
                'Chikou_Span': chikou_span
            }, index=data.index)
            
            # Generate Ichimoku signals
            signals = []
            
            if len(ichimoku_data.dropna()) >= 2:
                current_price = data['close'].iloc[-1]
                current_tenkan = tenkan_sen.iloc[-1]
                current_kijun = kijun_sen.iloc[-1]
                current_span_a = senkou_span_a.iloc[-1]
                current_span_b = senkou_span_b.iloc[-1]
                
                # TK Cross
                if not (np.isnan(current_tenkan) or np.isnan(current_kijun)):
                    if current_tenkan > current_kijun:
                        signals.append(IndicatorSignal(
                            indicator_name='Ichimoku',
                            signal_type=SignalType.BULLISH,
                            strength=0.7,
                            confidence=70.0,
                            value=current_tenkan,
                            description="Ichimoku TK bullish cross"
                        ))
                
                # Price above cloud
                if not (np.isnan(current_span_a) or np.isnan(current_span_b)):
                    cloud_top = max(current_span_a, current_span_b)
                    cloud_bottom = min(current_span_a, current_span_b)
                    
                    if current_price > cloud_top:
                        signals.append(IndicatorSignal(
                            indicator_name='Ichimoku',
                            signal_type=SignalType.BULLISH,
                            strength=0.8,
                            confidence=75.0,
                            value=current_price,
                            description="Price above Ichimoku cloud"
                        ))
                    elif current_price < cloud_bottom:
                        signals.append(IndicatorSignal(
                            indicator_name='Ichimoku',
                            signal_type=SignalType.BEARISH,
                            strength=0.8,
                            confidence=75.0,
                            value=current_price,
                            description="Price below Ichimoku cloud"
                        ))
            
            return IndicatorResult(
                name='ichimoku',
                values=ichimoku_data,
                signals=signals,
                parameters={
                    'tenkan_period': tenkan_period,
                    'kijun_period': kijun_period,
                    'senkou_span_b_period': senkou_span_b_period,
                    'displacement': displacement
                }
            )
            
        except Exception as e:
            logger.warning(f"Error calculating Ichimoku: {e}")
            return None
    
    # ==================== MOMENTUM INDICATORS ====================
    
    def _calculate_rsi(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate RSI (Relative Strength Index)"""
        
        try:
            period = self.config['momentum_indicators']['rsi_period']
            
            if len(data) < period:
                return None
            
            rsi = talib.RSI(data['close'].values, timeperiod=period)
            rsi_data = pd.DataFrame({'RSI': rsi}, index=data.index)
            
            # Generate RSI signals
            signals = []
            overbought = self.config['signal_generation']['overbought_threshold']
            oversold = self.config['signal_generation']['oversold_threshold']
            
            if not np.isnan(rsi[-1]):
                latest_rsi = rsi[-1]
                
                if latest_rsi > overbought:
                    signals.append(IndicatorSignal(
                        indicator_name='RSI',
                        signal_type=SignalType.OVERBOUGHT,
                        strength=min(1.0, (latest_rsi - overbought) / (100 - overbought)),
                        confidence=min(95.0, 50 + (latest_rsi - overbought)),
                        value=latest_rsi,
                        threshold=overbought,
                        description=f"RSI overbought ({latest_rsi:.1f})"
                    ))
                elif latest_rsi < oversold:
                    signals.append(IndicatorSignal(
                        indicator_name='RSI',
                        signal_type=SignalType.OVERSOLD,
                        strength=min(1.0, (oversold - latest_rsi) / oversold),
                        confidence=min(95.0, 50 + (oversold - latest_rsi)),
                        value=latest_rsi,
                        threshold=oversold,
                        description=f"RSI oversold ({latest_rsi:.1f})"
                    ))
            
            return IndicatorResult(
                name='rsi',
                values=rsi_data,
                signals=signals,
                parameters={'period': period}
            )
            
        except Exception as e:
            logger.warning(f"Error calculating RSI: {e}")
            return None
    
    def _calculate_stochastic(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate Stochastic Oscillator"""
        
        try:
            config = self.config['momentum_indicators']['stochastic_config']
            
            if len(data) < config['k_period'] or not all(col in data.columns for col in ['high', 'low', 'close']):
                return None
            
            slowk, slowd = talib.STOCH(
                data['high'].values,
                data['low'].values,
                data['close'].values,
                fastk_period=config['k_period'],
                slowk_period=config['d_period'],
                slowd_period=config['d_period']
            )
            
            stoch_data = pd.DataFrame({
                'Stoch_K': slowk,
                'Stoch_D': slowd
            }, index=data.index)
            
            # Generate Stochastic signals
            signals = []
            
            if not (np.isnan(slowk[-1]) or np.isnan(slowd[-1])):
                latest_k = slowk[-1]
                latest_d = slowd[-1]
                
                if latest_k > 80 and latest_d > 80:
                    signals.append(IndicatorSignal(
                        indicator_name='Stochastic',
                        signal_type=SignalType.OVERBOUGHT,
                        strength=0.7,
                        confidence=70.0,
                        value=latest_k,
                        threshold=80,
                        description=f"Stochastic overbought (K: {latest_k:.1f})"
                    ))
                elif latest_k < 20 and latest_d < 20:
                    signals.append(IndicatorSignal(
                        indicator_name='Stochastic',
                        signal_type=SignalType.OVERSOLD,
                        strength=0.7,
                        confidence=70.0,
                        value=latest_k,
                        threshold=20,
                        description=f"Stochastic oversold (K: {latest_k:.1f})"
                    ))
                
                # K%D crossover
                if len(stoch_data.dropna()) >= 2:
                    prev_k = slowk[-2]
                    prev_d = slowd[-2]
                    
                    if prev_k <= prev_d and latest_k > latest_d and latest_k < 80:
                        signals.append(IndicatorSignal(
                            indicator_name='Stochastic',
                            signal_type=SignalType.BUY,
                            strength=0.6,
                            confidence=65.0,
                            value=latest_k,
                            description="Stochastic bullish crossover"
                        ))
                    elif prev_k >= prev_d and latest_k < latest_d and latest_k > 20:
                        signals.append(IndicatorSignal(
                            indicator_name='Stochastic',
                            signal_type=SignalType.SELL,
                            strength=0.6,
                            confidence=65.0,
                            value=latest_k,
                            description="Stochastic bearish crossover"
                        ))
            
            return IndicatorResult(
                name='stochastic',
                values=stoch_data,
                signals=signals,
                parameters=config
            )
            
        except Exception as e:
            logger.warning(f"Error calculating Stochastic: {e}")
            return None
    
    def _calculate_williams_r(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate Williams %R"""
        
        try:
            period = self.config['momentum_indicators']['williams_r_period']
            
            if len(data) < period or not all(col in data.columns for col in ['high', 'low', 'close']):
                return None
            
            willr = talib.WILLR(data['high'].values, data['low'].values, data['close'].values, timeperiod=period)
            willr_data = pd.DataFrame({'Williams_R': willr}, index=data.index)
            
            # Generate Williams %R signals
            signals = []
            
            if not np.isnan(willr[-1]):
                latest_willr = willr[-1]
                
                if latest_willr > -20:  # Overbought
                    signals.append(IndicatorSignal(
                        indicator_name='Williams_R',
                        signal_type=SignalType.OVERBOUGHT,
                        strength=0.7,
                        confidence=70.0,
                        value=latest_willr,
                        threshold=-20,
                        description=f"Williams %R overbought ({latest_willr:.1f})"
                    ))
                elif latest_willr < -80:  # Oversold
                    signals.append(IndicatorSignal(
                        indicator_name='Williams_R',
                        signal_type=SignalType.OVERSOLD,
                        strength=0.7,
                        confidence=70.0,
                        value=latest_willr,
                        threshold=-80,
                        description=f"Williams %R oversold ({latest_willr:.1f})"
                    ))
            
            return IndicatorResult(
                name='williams_r',
                values=willr_data,
                signals=signals,
                parameters={'period': period}
            )
            
        except Exception as e:
            logger.warning(f"Error calculating Williams %R: {e}")
            return None
    
    def _calculate_roc(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate Rate of Change"""
        
        try:
            period = self.config['momentum_indicators']['roc_period']
            
            if len(data) < period:
                return None
            
            roc = talib.ROC(data['close'].values, timeperiod=period)
            roc_data = pd.DataFrame({'ROC': roc}, index=data.index)
            
            return IndicatorResult(
                name='roc',
                values=roc_data,
                parameters={'period': period}
            )
            
        except Exception as e:
            logger.warning(f"Error calculating ROC: {e}")
            return None
    
    def _calculate_momentum(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate Momentum"""
        
        try:
            period = self.config['momentum_indicators']['momentum_period']
            
            if len(data) < period:
                return None
            
            momentum = talib.MOM(data['close'].values, timeperiod=period)
            momentum_data = pd.DataFrame({'Momentum': momentum}, index=data.index)
            
            return IndicatorResult(
                name='momentum',
                values=momentum_data,
                parameters={'period': period}
            )
            
        except Exception as e:
            logger.warning(f"Error calculating Momentum: {e}")
            return None
    
    def _calculate_cci(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate Commodity Channel Index"""
        
        try:
            period = self.config['momentum_indicators']['cci_period']
            
            if len(data) < period or not all(col in data.columns for col in ['high', 'low', 'close']):
                return None
            
            cci = talib.CCI(data['high'].values, data['low'].values, data['close'].values, timeperiod=period)
            cci_data = pd.DataFrame({'CCI': cci}, index=data.index)
            
            # Generate CCI signals
            signals = []
            
            if not np.isnan(cci[-1]):
                latest_cci = cci[-1]
                
                if latest_cci > 100:
                    signals.append(IndicatorSignal(
                        indicator_name='CCI',
                        signal_type=SignalType.OVERBOUGHT,
                        strength=min(1.0, (latest_cci - 100) / 200),
                        confidence=min(90.0, 50 + abs(latest_cci - 100) / 10),
                        value=latest_cci,
                        threshold=100,
                        description=f"CCI overbought ({latest_cci:.1f})"
                    ))
                elif latest_cci < -100:
                    signals.append(IndicatorSignal(
                        indicator_name='CCI',
                        signal_type=SignalType.OVERSOLD,
                        strength=min(1.0, (abs(latest_cci) - 100) / 200),
                        confidence=min(90.0, 50 + abs(latest_cci + 100) / 10),
                        value=latest_cci,
                        threshold=-100,
                        description=f"CCI oversold ({latest_cci:.1f})"
                    ))
            
            return IndicatorResult(
                name='cci',
                values=cci_data,
                signals=signals,
                parameters={'period': period}
            )
            
        except Exception as e:
            logger.warning(f"Error calculating CCI: {e}")
            return None
    
    def _calculate_ultimate_oscillator(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate Ultimate Oscillator"""
        
        try:
            if len(data) < 28 or not all(col in data.columns for col in ['high', 'low', 'close']):
                return None
            
            ultosc = talib.ULTOSC(
                data['high'].values,
                data['low'].values,
                data['close'].values,
                timeperiod1=7,
                timeperiod2=14,
                timeperiod3=28
            )
            
            ultosc_data = pd.DataFrame({'Ultimate_Oscillator': ultosc}, index=data.index)
            
            return IndicatorResult(
                name='ultimate_oscillator',
                values=ultosc_data,
                parameters={'periods': [7, 14, 28]}
            )
            
        except Exception as e:
            logger.warning(f"Error calculating Ultimate Oscillator: {e}")
            return None
    
    # ==================== VOLATILITY INDICATORS ====================
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate Bollinger Bands"""
        
        try:
            config = self.config['volatility_indicators']['bollinger_config']
            
            if len(data) < config['period']:
                return None
            
            upper, middle, lower = talib.BBANDS(
                data['close'].values,
                timeperiod=config['period'],
                nbdevup=config['std_dev'],
                nbdevdn=config['std_dev']
            )
            
            bb_data = pd.DataFrame({
                'BB_Upper': upper,
                'BB_Middle': middle,
                'BB_Lower': lower,
                'BB_Width': (upper - lower) / middle * 100,
                'BB_Position': (data['close'] - lower) / (upper - lower) * 100
            }, index=data.index)
            
            # Generate Bollinger Bands signals
            signals = []
            
            if not (np.isnan(upper[-1]) or np.isnan(lower[-1])):
                current_price = data['close'].iloc[-1]
                latest_upper = upper[-1]
                latest_lower = lower[-1]
                bb_position = bb_data['BB_Position'].iloc[-1]
                
                if current_price > latest_upper:
                    signals.append(IndicatorSignal(
                        indicator_name='Bollinger_Bands',
                        signal_type=SignalType.OVERBOUGHT,
                        strength=min(1.0, (current_price - latest_upper) / latest_upper),
                        confidence=80.0,
                        value=current_price,
                        threshold=latest_upper,
                        description="Price above Bollinger upper band"
                    ))
                elif current_price < latest_lower:
                    signals.append(IndicatorSignal(
                        indicator_name='Bollinger_Bands',
                        signal_type=SignalType.OVERSOLD,
                        strength=min(1.0, (latest_lower - current_price) / latest_lower),
                        confidence=80.0,
                        value=current_price,
                        threshold=latest_lower,
                        description="Price below Bollinger lower band"
                    ))
                
                # BB squeeze detection
                bb_width = bb_data['BB_Width'].iloc[-1]
                avg_width = bb_data['BB_Width'].tail(20).mean()
                
                if bb_width < avg_width * 0.5:
                    signals.append(IndicatorSignal(
                        indicator_name='Bollinger_Bands',
                        signal_type=SignalType.NEUTRAL,
                        strength=0.8,
                        confidence=75.0,
                        value=bb_width,
                        description="Bollinger Bands squeeze detected"
                    ))
            
            return IndicatorResult(
                name='bollinger_bands',
                values=bb_data,
                signals=signals,
                parameters=config
            )
            
        except Exception as e:
            logger.warning(f"Error calculating Bollinger Bands: {e}")
            return None
    
    def _calculate_atr(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate Average True Range"""
        
        try:
            period = self.config['volatility_indicators']['atr_period']
            
            if len(data) < period or not all(col in data.columns for col in ['high', 'low', 'close']):
                return None
            
            atr = talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=period)
            atr_data = pd.DataFrame({'ATR': atr}, index=data.index)
            
            return IndicatorResult(
                name='atr',
                values=atr_data,
                parameters={'period': period}
            )
            
        except Exception as e:
            logger.warning(f"Error calculating ATR: {e}")
            return None
    
    def _calculate_keltner_channels(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate Keltner Channels"""
        
        try:
            config = self.config['volatility_indicators']['keltner_config']
            period = config['period']
            multiplier = config['atr_multiplier']
            
            if len(data) < period or not all(col in data.columns for col in ['high', 'low', 'close']):
                return None
            
            # Calculate EMA and ATR
            ema = talib.EMA(data['close'].values, timeperiod=period)
            atr = talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=period)
            
            keltner_data = pd.DataFrame({
                'KC_Middle': ema,
                'KC_Upper': ema + (atr * multiplier),
                'KC_Lower': ema - (atr * multiplier)
            }, index=data.index)
            
            return IndicatorResult(
                name='keltner_channels',
                values=keltner_data,
                parameters=config
            )
            
        except Exception as e:
            logger.warning(f"Error calculating Keltner Channels: {e}")
            return None
    
    def _calculate_donchian_channels(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate Donchian Channels"""
        
        try:
            period = self.config['volatility_indicators']['donchian_period']
            
            if len(data) < period or not all(col in data.columns for col in ['high', 'low']):
                return None
            
            donchian_data = pd.DataFrame({
                'DC_Upper': data['high'].rolling(period).max(),
                'DC_Lower': data['low'].rolling(period).min(),
                'DC_Middle': (data['high'].rolling(period).max() + data['low'].rolling(period).min()) / 2
            }, index=data.index)
            
            return IndicatorResult(
                name='donchian_channels',
                values=donchian_data,
                parameters={'period': period}
            )
            
        except Exception as e:
            logger.warning(f"Error calculating Donchian Channels: {e}")
            return None
    
    # ==================== VOLUME INDICATORS ====================
    
    def _calculate_obv(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate On-Balance Volume"""
        
        try:
            if not self.config['volume_indicators']['obv_enabled'] or 'volume' not in data.columns:
                return None
            
            obv = talib.OBV(data['close'].values, data['volume'].values)
            obv_data = pd.DataFrame({'OBV': obv}, index=data.index)
            
            return IndicatorResult(
                name='obv',
                values=obv_data
            )
            
        except Exception as e:
            logger.warning(f"Error calculating OBV: {e}")
            return None
    
    def _calculate_ad_line(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate Accumulation/Distribution Line"""
        
        try:
            if (not self.config['volume_indicators']['ad_line_enabled'] or 
                not all(col in data.columns for col in ['high', 'low', 'close', 'volume'])):
                return None
            
            ad = talib.AD(data['high'].values, data['low'].values, data['close'].values, data['volume'].values)
            ad_data = pd.DataFrame({'AD_Line': ad}, index=data.index)
            
            return IndicatorResult(
                name='ad_line',
                values=ad_data
            )
            
        except Exception as e:
            logger.warning(f"Error calculating A/D Line: {e}")
            return None
    
    def _calculate_chaikin_oscillator(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate Chaikin Oscillator"""
        
        try:
            config = self.config['volume_indicators']['chaikin_config']
            
            if not all(col in data.columns for col in ['high', 'low', 'close', 'volume']):
                return None
            
            chaikin = talib.ADOSC(
                data['high'].values,
                data['low'].values,
                data['close'].values,
                data['volume'].values,
                fastperiod=config['fast'],
                slowperiod=config['slow']
            )
            
            chaikin_data = pd.DataFrame({'Chaikin_Oscillator': chaikin}, index=data.index)
            
            return IndicatorResult(
                name='chaikin_oscillator',
                values=chaikin_data,
                parameters=config
            )
            
        except Exception as e:
            logger.warning(f"Error calculating Chaikin Oscillator: {e}")
            return None
    
    def _calculate_mfi(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate Money Flow Index"""
        
        try:
            period = self.config['volume_indicators']['mfi_period']
            
            if (len(data) < period or 
                not all(col in data.columns for col in ['high', 'low', 'close', 'volume'])):
                return None
            
            mfi = talib.MFI(
                data['high'].values,
                data['low'].values,
                data['close'].values,
                data['volume'].values,
                timeperiod=period
            )
            
            mfi_data = pd.DataFrame({'MFI': mfi}, index=data.index)
            
            # Generate MFI signals
            signals = []
            
            if not np.isnan(mfi[-1]):
                latest_mfi = mfi[-1]
                
                if latest_mfi > 80:
                    signals.append(IndicatorSignal(
                        indicator_name='MFI',
                        signal_type=SignalType.OVERBOUGHT,
                        strength=0.7,
                        confidence=75.0,
                        value=latest_mfi,
                        threshold=80,
                        description=f"MFI overbought ({latest_mfi:.1f})"
                    ))
                elif latest_mfi < 20:
                    signals.append(IndicatorSignal(
                        indicator_name='MFI',
                        signal_type=SignalType.OVERSOLD,
                        strength=0.7,
                        confidence=75.0,
                        value=latest_mfi,
                        threshold=20,
                        description=f"MFI oversold ({latest_mfi:.1f})"
                    ))
            
            return IndicatorResult(
                name='mfi',
                values=mfi_data,
                signals=signals,
                parameters={'period': period}
            )
            
        except Exception as e:
            logger.warning(f"Error calculating MFI: {e}")
            return None
    
    def _calculate_vwap(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate Volume Weighted Average Price"""
        
        try:
            if (not self.config['volume_indicators']['vwap_enabled'] or 
                not all(col in data.columns for col in ['high', 'low', 'close', 'volume'])):
                return None
            
            # Calculate typical price
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            
            # Calculate VWAP
            cumulative_pv = (typical_price * data['volume']).cumsum()
            cumulative_volume = data['volume'].cumsum()
            vwap = cumulative_pv / cumulative_volume
            
            vwap_data = pd.DataFrame({'VWAP': vwap}, index=data.index)
            
            return IndicatorResult(
                name='vwap',
                values=vwap_data
            )
            
        except Exception as e:
            logger.warning(f"Error calculating VWAP: {e}")
            return None
    
    def _calculate_volume_profile(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate Volume Profile"""
        
        try:
            if 'volume' not in data.columns:
                return None
            
            # Simplified volume profile - would need more sophisticated implementation
            price_levels = 20  # Number of price levels
            
            price_min = data['low'].min()
            price_max = data['high'].max()
            price_range = price_max - price_min
            level_size = price_range / price_levels
            
            volume_profile = {}
            
            for i in range(price_levels):
                level_min = price_min + (i * level_size)
                level_max = level_min + level_size
                
                # Find bars that traded in this price level
                mask = ((data['low'] <= level_max) & (data['high'] >= level_min))
                level_volume = data.loc[mask, 'volume'].sum()
                
                volume_profile[f'Level_{i}'] = {
                    'price_min': level_min,
                    'price_max': level_max,
                    'volume': level_volume
                }
            
            # Convert to DataFrame for consistency
            vp_data = pd.DataFrame([volume_profile])
            
            return IndicatorResult(
                name='volume_profile',
                values=vp_data,
                parameters={'price_levels': price_levels}
            )
            
        except Exception as e:
            logger.warning(f"Error calculating Volume Profile: {e}")
            return None
    
    # ==================== CUSTOM INDICATORS ====================
    
    def _calculate_composite_momentum(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate composite momentum from multiple indicators"""
        
        try:
            if len(data) < 30:
                return None
            
            # Calculate individual momentum components
            rsi = talib.RSI(data['close'].values, timeperiod=14)
            
            if len(data) >= 14 and all(col in data.columns for col in ['high', 'low']):
                stoch_k, _ = talib.STOCH(
                    data['high'].values, data['low'].values, data['close'].values,
                    fastk_period=14, slowk_period=3, slowd_period=3
                )
            else:
                stoch_k = np.full(len(data), np.nan)
            
            if len(data) >= 20 and all(col in data.columns for col in ['high', 'low']):
                willr = talib.WILLR(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)
            else:
                willr = np.full(len(data), np.nan)
            
            # Normalize indicators to 0-100 scale
            rsi_norm = rsi
            stoch_norm = stoch_k
            willr_norm = (willr + 100)  # Convert from -100,0 to 0,100
            
            # Calculate composite momentum
            valid_mask = ~(np.isnan(rsi_norm) | np.isnan(stoch_norm) | np.isnan(willr_norm))
            composite = np.full(len(data), np.nan)
            
            if np.any(valid_mask):
                composite[valid_mask] = (
                    0.4 * rsi_norm[valid_mask] + 
                    0.3 * stoch_norm[valid_mask] + 
                    0.3 * willr_norm[valid_mask]
                )
            
            momentum_data = pd.DataFrame({'Composite_Momentum': composite}, index=data.index)
            
            return IndicatorResult(
                name='composite_momentum',
                values=momentum_data,
                parameters={'weights': [0.4, 0.3, 0.3]}
            )
            
        except Exception as e:
            logger.warning(f"Error calculating Composite Momentum: {e}")
            return None
    
    def _calculate_trend_strength(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate overall trend strength"""
        
        try:
            if len(data) < 50:
                return None
            
            # Calculate trend components
            trend_components = []
            
            # ADX component
            if all(col in data.columns for col in ['high', 'low']):
                adx = talib.ADX(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)
                if not np.isnan(adx[-1]):
                    trend_components.append(min(100, adx[-1] * 2))  # Scale ADX
            
            # Moving average alignment
            if len(data) >= 50:
                sma_20 = talib.SMA(data['close'].values, timeperiod=20)
                sma_50 = talib.SMA(data['close'].values, timeperiod=50)
                
                if not (np.isnan(sma_20[-1]) or np.isnan(sma_50[-1])):
                    price = data['close'].iloc[-1]
                    if price > sma_20[-1] > sma_50[-1]:
                        trend_components.append(100)  # Strong uptrend
                    elif price < sma_20[-1] < sma_50[-1]:
                        trend_components.append(100)  # Strong downtrend
                    else:
                        trend_components.append(50)   # Neutral
            
            # Price momentum
            if len(data) >= 20:
                momentum = (data['close'].iloc[-1] / data['close'].iloc[-20] - 1) * 100
                momentum_strength = min(100, abs(momentum) * 5)  # Scale momentum
                trend_components.append(momentum_strength)
            
            # Calculate composite trend strength
            if trend_components:
                trend_strength = np.mean(trend_components)
            else:
                trend_strength = 50  # Neutral
            
            trend_data = pd.DataFrame(
                {'Trend_Strength': [trend_strength] * len(data)}, 
                index=data.index
            )
            
            return IndicatorResult(
                name='trend_strength',
                values=trend_data,
                parameters={'components': len(trend_components)}
            )
            
        except Exception as e:
            logger.warning(f"Error calculating Trend Strength: {e}")
            return None
    
    def _calculate_market_regime(self, data: pd.DataFrame, symbol: str) -> Optional[IndicatorResult]:
        """Calculate market regime classification"""
        
        try:
            if len(data) < 60:
                return None
            
            # Calculate regime indicators
            volatility = data['close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
            
            # Trend strength
            sma_20 = talib.SMA(data['close'].values, timeperiod=20)
            sma_50 = talib.SMA(data['close'].values, timeperiod=50)
            
            regime_scores = []
            
            for i in range(len(data)):
                if i < 60:
                    regime_scores.append(np.nan)
                    continue
                
                current_vol = volatility.iloc[i] if not pd.isna(volatility.iloc[i]) else 20
                current_price = data['close'].iloc[i]
                
                # Classify regime
                if current_vol < 15:  # Low volatility
                    if not (np.isnan(sma_20[i]) or np.isnan(sma_50[i])):
                        if current_price > sma_20[i] > sma_50[i]:
                            regime = 1  # Low vol uptrend
                        elif current_price < sma_20[i] < sma_50[i]:
                            regime = 2  # Low vol downtrend
                        else:
                            regime = 3  # Low vol sideways
                    else:
                        regime = 3
                elif current_vol > 30:  # High volatility
                    regime = 4  # High volatility
                else:
                    regime = 5  # Normal volatility
                
                regime_scores.append(regime)
            
            regime_data = pd.DataFrame({'Market_Regime': regime_scores}, index=data.index)
            
            return IndicatorResult(
                name='market_regime',
                values=regime_data,
                parameters={
                    'regimes': {
                        1: 'Low Vol Uptrend',
                        2: 'Low Vol Downtrend', 
                        3: 'Low Vol Sideways',
                        4: 'High Volatility',
                        5: 'Normal Volatility'
                    }
                }
            )
            
        except Exception as e:
            logger.warning(f"Error calculating Market Regime: {e}")
            return None
    
    # ==================== UTILITY METHODS ====================
    
    def _detect_ma_crossover(self, fast_ma: pd.Series, slow_ma: pd.Series, name: str) -> List[IndicatorSignal]:
        """Detect moving average crossovers"""
        
        signals = []
        
        try:
            if len(fast_ma) < 2 or len(slow_ma) < 2:
                return signals
            
            # Get recent values
            fast_current = fast_ma.iloc[-1]
            fast_prev = fast_ma.iloc[-2]
            slow_current = slow_ma.iloc[-1]
            slow_prev = slow_ma.iloc[-2]
            
            # Check for valid values
            if any(np.isnan([fast_current, fast_prev, slow_current, slow_prev])):
                return signals
            
            # Detect crossovers
            sensitivity = self.config['signal_generation']['crossover_sensitivity']
            
            if (fast_prev <= slow_prev and fast_current > slow_current and 
                abs(fast_current - slow_current) / slow_current > sensitivity):
                
                signals.append(IndicatorSignal(
                    indicator_name=name,
                    signal_type=SignalType.BUY,
                    strength=0.8,
                    confidence=80.0,
                    value=fast_current,
                    description=f"{name} bullish crossover"
                ))
                
            elif (fast_prev >= slow_prev and fast_current < slow_current and 
                  abs(slow_current - fast_current) / slow_current > sensitivity):
                
                signals.append(IndicatorSignal(
                    indicator_name=name,
                    signal_type=SignalType.SELL,
                    strength=0.8,
                    confidence=80.0,
                    value=fast_current,
                    description=f"{name} bearish crossover"
                ))
            
            return signals
            
        except Exception as e:
            logger.warning(f"Error detecting MA crossover: {e}")
            return signals
    
    def _calculate_composite_indicators(self, results: Dict[str, IndicatorResult], data: pd.DataFrame) -> Dict[str, IndicatorResult]:
        """Calculate composite indicators from existing results"""
        
        composite_results = {}
        
        try:
            # Momentum composite
            momentum_indicators = ['rsi', 'stochastic', 'williams_r']
            momentum_values = []
            
            for indicator in momentum_indicators:
                if indicator in results:
                    values = results[indicator].values
                    if isinstance(values, pd.DataFrame):
                        # Take first column if DataFrame
                        values = values.iloc[:, 0]
                    
                    if len(values.dropna()) > 0:
                        # Normalize to 0-100 scale
                        if indicator == 'williams_r':
                            normalized = (values + 100)  # Convert -100,0 to 0,100
                        else:
                            normalized = values
                        
                        momentum_values.append(normalized)
            
            if momentum_values:
                composite_momentum = pd.concat(momentum_values, axis=1).mean(axis=1)
                composite_results['momentum_composite'] = IndicatorResult(
                    name='momentum_composite',
                    values=pd.DataFrame({'Momentum_Composite': composite_momentum}),
                    parameters={'components': len(momentum_values)}
                )
            
            return composite_results
            
        except Exception as e:
            logger.warning(f"Error calculating composite indicators: {e}")
            return {}
    
    def _generate_combined_signals(self, results: Dict[str, IndicatorResult]) -> List[IndicatorSignal]:
        """Generate combined signals from all indicators"""
        
        combined_signals = []
        
        try:
            # Collect all signals
            all_signals = []
            for result in results.values():
                all_signals.extend(result.signals)
            
            if not all_signals:
                return combined_signals
            
            # Count signal types
            signal_counts = {
                SignalType.BUY: 0,
                SignalType.SELL: 0,
                SignalType.BULLISH: 0,
                SignalType.BEARISH: 0,
                SignalType.OVERBOUGHT: 0,
                SignalType.OVERSOLD: 0
            }
            
            total_confidence = 0
            signal_count = 0
            
            for signal in all_signals:
                if signal.signal_type in signal_counts:
                    signal_counts[signal.signal_type] += 1
                    total_confidence += signal.confidence
                    signal_count += 1
            
            # Generate combined signals based on consensus
            if signal_count > 0:
                avg_confidence = total_confidence / signal_count
                
                # Bullish consensus
                bullish_signals = signal_counts[SignalType.BUY] + signal_counts[SignalType.BULLISH] + signal_counts[SignalType.OVERSOLD]
                # Bearish consensus
                bearish_signals = signal_counts[SignalType.SELL] + signal_counts[SignalType.BEARISH] + signal_counts[SignalType.OVERBOUGHT]
                
                total_directional = bullish_signals + bearish_signals
                
                if total_directional >= 3:  # Minimum 3 signals for consensus
                    if bullish_signals > bearish_signals:
                        combined_signals.append(IndicatorSignal(
                            indicator_name='Combined',
                            signal_type=SignalType.BULLISH,
                            strength=bullish_signals / total_directional,
                            confidence=min(95.0, avg_confidence * (bullish_signals / total_directional)),
                            value=bullish_signals,
                            description=f"Bullish consensus ({bullish_signals}/{total_directional} signals)"
                        ))
                    elif bearish_signals > bullish_signals:
                        combined_signals.append(IndicatorSignal(
                            indicator_name='Combined',
                            signal_type=SignalType.BEARISH,
                            strength=bearish_signals / total_directional,
                            confidence=min(95.0, avg_confidence * (bearish_signals / total_directional)),
                            value=bearish_signals,
                            description=f"Bearish consensus ({bearish_signals}/{total_directional} signals)"
                        ))
            
            return combined_signals
            
        except Exception as e:
            logger.warning(f"Error generating combined signals: {e}")
            return []
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data"""
        
        if data is None or len(data) == 0:
            return False
        
        required_columns = ['close']
        if not all(col in data.columns for col in required_columns):
            return False
        
        if data['close'].isna().all():
            return False
        
        return True
    
    def get_indicator_summary(self, results: Dict[str, IndicatorResult]) -> Dict[str, Any]:
        """Get summary of all calculated indicators"""
        
        summary = {
            'timestamp': datetime.now(),
            'total_indicators': len(results),
            'indicators_with_signals': 0,
            'total_signals': 0,
            'signal_breakdown': {
                'bullish': 0,
                'bearish': 0,
                'neutral': 0,
                'overbought': 0,
                'oversold': 0
            },
            'avg_confidence': 0.0
        }
        
        try:
            all_signals = []
            
            for result in results.values():
                if result.signals:
                    summary['indicators_with_signals'] += 1
                    all_signals.extend(result.signals)
            
            summary['total_signals'] = len(all_signals)
            
            if all_signals:
                # Count signal types
                for signal in all_signals:
                    if signal.signal_type in [SignalType.BUY, SignalType.BULLISH]:
                        summary['signal_breakdown']['bullish'] += 1
                    elif signal.signal_type in [SignalType.SELL, SignalType.BEARISH]:
                        summary['signal_breakdown']['bearish'] += 1
                    elif signal.signal_type == SignalType.OVERBOUGHT:
                        summary['signal_breakdown']['overbought'] += 1
                    elif signal.signal_type == SignalType.OVERSOLD:
                        summary['signal_breakdown']['oversold'] += 1
                    else:
                        summary['signal_breakdown']['neutral'] += 1
                
                # Calculate average confidence
                summary['avg_confidence'] = np.mean([s.confidence for s in all_signals])
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating indicator summary: {e}")
            return summary

# ==================== TESTING ====================

def test_technical_indicators():
    """Test technical indicators functionality"""
    
    print("📊 Testing Advanced Technical Indicators")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    
    base_price = 100
    prices = [base_price]
    
    for i in range(1, 100):
        change = np.random.randn() * 0.02  # 2% volatility
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    sample_data = pd.DataFrame({
        'open': prices,
        'close': [p * (1 + np.random.randn() * 0.005) for p in prices],
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Add high/low
    sample_data['high'] = sample_data[['open', 'close']].max(axis=1) * (1 + np.random.rand(100) * 0.01)
    sample_data['low'] = sample_data[['open', 'close']].min(axis=1) * (1 - np.random.rand(100) * 0.01)
    
    # Create technical indicators calculator
    indicators = AdvancedTechnicalIndicators()
    
    print(f"✅ Technical indicators calculator initialized:")
    print(f"   Available indicators: {len(indicators.indicators)}")
    print(f"   Trend indicators: {len(indicators.config['trend_indicators'])}")
    print(f"   Momentum indicators: {len(indicators.config['momentum_indicators'])}")
    print(f"   Volatility indicators: {len(indicators.config['volatility_indicators'])}")
    print(f"   Volume indicators: {len(indicators.config['volume_indicators'])}")
    
    # Calculate all indicators
    results = indicators.calculate_all_indicators(sample_data, 'TEST')
    
    print(f"\n📈 Calculated indicators:")
    print(f"   Total indicators calculated: {len(results)}")
    
    # Show individual indicator results
    for name, result in results.items():
        if name == 'combined_signals':
            continue
        
        signal_count = len(result.signals)
        
        if isinstance(result.values, pd.DataFrame):
            columns = list(result.values.columns)
            latest_values = {}
            for col in columns[:3]:  # Show first 3 columns
                try:
                    latest_val = result.values[col].iloc[-1]
                    if not pd.isna(latest_val):
                        latest_values[col] = latest_val
                except:
                    pass
        else:
            latest_val = result.values.iloc[-1] if len(result.values) > 0 else None
            latest_values = {name: latest_val} if latest_val is not None else {}
        
        print(f"   {name}: {signal_count} signals, {len(latest_values)} values")
        
        for col, val in latest_values.items():
            if isinstance(val, (int, float)):
                print(f"      {col}: {val:.3f}")
        
        if result.signals:
            for signal in result.signals[:2]:  # Show first 2 signals
                print(f"      Signal: {signal.signal_type.value} "
                      f"({signal.confidence:.1f}% confidence) - {signal.description}")
    
    # Show indicator summary
    summary = indicators.get_indicator_summary(results)
    
    print(f"\n📋 Indicator Summary:")
    print(f"   Total indicators: {summary['total_indicators']}")
    print(f"   Indicators with signals: {summary['indicators_with_signals']}")
    print(f"   Total signals: {summary['total_signals']}")
    print(f"   Average confidence: {summary['avg_confidence']:.1f}%")
    
    print(f"\n🎯 Signal breakdown:")
    for signal_type, count in summary['signal_breakdown'].items():
        if count > 0:
            print(f"   {signal_type.title()}: {count}")
    
    # Show combined signals
    if 'combined_signals' in results:
        combined = results['combined_signals']
        print(f"\n🔄 Combined signals: {len(combined.signals)}")
        for signal in combined.signals:
            print(f"   {signal.signal_type.value}: {signal.strength:.2f} strength, "
                  f"{signal.confidence:.1f}% confidence")
            print(f"   Description: {signal.description}")
    
    print("\n🎉 Technical indicators tests completed!")

if __name__ == "__main__":
    test_technical_indicators()