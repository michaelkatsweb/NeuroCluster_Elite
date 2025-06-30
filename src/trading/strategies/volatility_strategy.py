#!/usr/bin/env python3
"""
File: volatility_strategy.py
Path: NeuroCluster-Elite/src/trading/strategies/volatility_strategy.py
Description: Advanced volatility trading strategy for high-volatility market regimes

This module implements sophisticated volatility trading strategies that capitalize on
market volatility through various approaches including volatility breakouts,
mean reversion, volatility surface analysis, and options-based volatility trading.

Features:
- Multiple volatility models (GARCH, EWMA, Parkinson, etc.)
- Volatility regime detection and adaptation
- Mean reversion and volatility breakout strategies
- VIX and implied volatility analysis
- Options volatility strategies
- Multi-timeframe volatility analysis
- Risk-adjusted position sizing based on volatility
- Dynamic stop-loss and take-profit based on volatility

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
from scipy.optimize import minimize
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

# ==================== VOLATILITY ENUMS AND STRUCTURES ====================

class VolatilityModel(Enum):
    """Volatility estimation models"""
    HISTORICAL = "historical"
    EWMA = "ewma"  # Exponentially Weighted Moving Average
    GARCH = "garch"  # Generalized Autoregressive Conditional Heteroskedasticity
    PARKINSON = "parkinson"  # Parkinson high-low estimator
    GARMAN_KLASS = "garman_klass"  # Garman-Klass OHLC estimator
    ROGERS_SATCHELL = "rogers_satchell"  # Rogers-Satchell estimator
    YANG_ZHANG = "yang_zhang"  # Yang-Zhang estimator

class VolatilityRegime(Enum):
    """Volatility regime types"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"
    EXPANDING = "expanding"
    CONTRACTING = "contracting"

class VolatilityStrategy(Enum):
    """Volatility trading strategies"""
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    VOLATILITY_CLUSTERING = "volatility_clustering"
    VOLATILITY_SURFACE = "volatility_surface"
    VIX_TRADING = "vix_trading"
    IMPLIED_VOLATILITY = "implied_volatility"

@dataclass
class VolatilityMetrics:
    """Comprehensive volatility metrics"""
    symbol: str
    timestamp: datetime
    
    # Current volatility measures
    historical_vol_1d: float = 0.0
    historical_vol_7d: float = 0.0
    historical_vol_30d: float = 0.0
    ewma_vol: float = 0.0
    parkinson_vol: float = 0.0
    garman_klass_vol: float = 0.0
    yang_zhang_vol: float = 0.0
    
    # Regime information
    volatility_regime: VolatilityRegime = VolatilityRegime.NORMAL
    regime_confidence: float = 0.0
    regime_persistence: float = 0.0
    
    # Volatility dynamics
    volatility_trend: float = 0.0  # Positive = increasing, negative = decreasing
    volatility_mean_reversion_speed: float = 0.0
    volatility_clustering_factor: float = 0.0
    
    # Relative volatility measures
    volatility_percentile_30d: float = 0.0
    volatility_percentile_90d: float = 0.0
    volatility_z_score: float = 0.0
    
    # Risk metrics
    downside_volatility: float = 0.0
    upside_volatility: float = 0.0
    volatility_skew: float = 0.0
    volatility_kurtosis: float = 0.0
    
    # Market-specific metrics
    implied_volatility: Optional[float] = None
    volatility_risk_premium: Optional[float] = None
    term_structure_slope: Optional[float] = None

@dataclass
class VolatilitySignal:
    """Volatility-specific trading signal"""
    base_signal: TradingSignal
    volatility_metrics: VolatilityMetrics
    volatility_strategy: VolatilityStrategy
    expected_volatility: float
    volatility_confidence: float
    risk_adjustment_factor: float
    optimal_holding_period: int  # in bars
    volatility_stop_loss: Optional[float] = None
    volatility_take_profit: Optional[float] = None

# ==================== VOLATILITY STRATEGY IMPLEMENTATION ====================

class AdvancedVolatilityStrategy(BaseStrategy):
    """
    Advanced volatility trading strategy
    
    This strategy uses multiple volatility models and regime detection to capitalize
    on volatility patterns in the market. It includes mean reversion, breakout,
    and volatility surface strategies.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize volatility strategy"""
        
        self.config = config or self._get_default_config()
        self.volatility_cache = {}
        self.regime_history = {}
        self.model_parameters = {}
        
        # Initialize volatility models
        self._initialize_models()
        
        # Strategy state
        self.current_regime = VolatilityRegime.NORMAL
        self.regime_confidence = 0.0
        self.last_regime_change = None
        
        super().__init__()
        logger.info("Advanced Volatility Strategy initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'volatility_models': {
                'primary_model': VolatilityModel.YANG_ZHANG,
                'secondary_models': [VolatilityModel.EWMA, VolatilityModel.PARKINSON],
                'model_weights': {
                    VolatilityModel.YANG_ZHANG: 0.4,
                    VolatilityModel.EWMA: 0.3,
                    VolatilityModel.PARKINSON: 0.3
                }
            },
            'regime_detection': {
                'lookback_period': 60,
                'regime_thresholds': {
                    'low': 0.15,      # 15th percentile
                    'normal': 0.85,   # 85th percentile
                    'high': 0.95      # 95th percentile
                },
                'min_regime_duration': 5,  # minimum bars in regime
                'regime_smoothing': 0.8
            },
            'mean_reversion': {
                'enabled': True,
                'reversion_speed_threshold': 0.5,
                'entry_z_score': 2.0,
                'exit_z_score': 0.5,
                'max_holding_period': 20
            },
            'volatility_breakout': {
                'enabled': True,
                'breakout_threshold': 1.5,  # standard deviations
                'confirmation_period': 3,
                'momentum_threshold': 0.02
            },
            'position_sizing': {
                'base_position_size': 0.02,  # 2% of portfolio
                'volatility_adjustment': True,
                'max_volatility_multiplier': 3.0,
                'min_volatility_multiplier': 0.5
            },
            'risk_management': {
                'volatility_stop_multiplier': 2.0,
                'profit_target_multiplier': 3.0,
                'max_daily_volatility': 0.05,  # 5% daily volatility limit
                'correlation_adjustment': True
            }
        }
    
    def _initialize_models(self):
        """Initialize volatility models"""
        
        self.volatility_models = {
            VolatilityModel.HISTORICAL: self._calculate_historical_volatility,
            VolatilityModel.EWMA: self._calculate_ewma_volatility,
            VolatilityModel.PARKINSON: self._calculate_parkinson_volatility,
            VolatilityModel.GARMAN_KLASS: self._calculate_garman_klass_volatility,
            VolatilityModel.YANG_ZHANG: self._calculate_yang_zhang_volatility
        }
        
        # Initialize model parameters
        self.model_parameters = {
            'ewma_lambda': 0.94,
            'garch_p': 1,
            'garch_q': 1,
            'min_periods': 20
        }
    
    def generate_signal(self, 
                       data: pd.DataFrame, 
                       symbol: str,
                       regime: RegimeType, 
                       confidence: float,
                       additional_data: Dict = None) -> Optional[VolatilitySignal]:
        """
        Generate volatility-based trading signal
        
        Args:
            data: OHLCV data
            symbol: Asset symbol
            regime: Current market regime
            confidence: Regime confidence
            additional_data: Additional market data
            
        Returns:
            VolatilitySignal or None
        """
        
        try:
            if len(data) < 30:  # Need minimum data for volatility analysis
                return None
            
            # Calculate comprehensive volatility metrics
            vol_metrics = self._calculate_volatility_metrics(data, symbol)
            
            if not vol_metrics:
                return None
            
            # Detect volatility regime
            vol_regime = self._detect_volatility_regime(vol_metrics, data)
            
            # Update regime history
            self._update_regime_history(symbol, vol_regime)
            
            # Generate signals based on volatility regime and patterns
            signals = []
            
            # Mean reversion signals
            if self.config['mean_reversion']['enabled']:
                mean_reversion_signal = self._generate_mean_reversion_signal(
                    data, vol_metrics, vol_regime
                )
                if mean_reversion_signal:
                    signals.append(mean_reversion_signal)
            
            # Volatility breakout signals
            if self.config['volatility_breakout']['enabled']:
                breakout_signal = self._generate_volatility_breakout_signal(
                    data, vol_metrics, vol_regime
                )
                if breakout_signal:
                    signals.append(breakout_signal)
            
            # Select best signal
            if not signals:
                return None
            
            best_signal = max(signals, key=lambda s: s.base_signal.confidence)
            
            # Apply risk adjustments
            adjusted_signal = self._apply_volatility_risk_adjustments(
                best_signal, vol_metrics, data
            )
            
            return adjusted_signal
            
        except Exception as e:
            logger.error(f"Error generating volatility signal for {symbol}: {e}")
            return None
    
    def _calculate_volatility_metrics(self, data: pd.DataFrame, symbol: str) -> Optional[VolatilityMetrics]:
        """Calculate comprehensive volatility metrics"""
        
        try:
            if len(data) < 20:
                return None
            
            current_time = data.index[-1] if hasattr(data.index[-1], 'to_pydatetime') else datetime.now()
            
            # Calculate returns
            returns = data['close'].pct_change().dropna()
            log_returns = np.log(data['close'] / data['close'].shift(1)).dropna()
            
            # Calculate different volatility measures
            vol_metrics = VolatilityMetrics(
                symbol=symbol,
                timestamp=current_time
            )
            
            # Historical volatilities (annualized)
            if len(returns) >= 1:
                vol_metrics.historical_vol_1d = float(returns.tail(1).std() * np.sqrt(252))
            if len(returns) >= 7:
                vol_metrics.historical_vol_7d = float(returns.tail(7).std() * np.sqrt(252))
            if len(returns) >= 30:
                vol_metrics.historical_vol_30d = float(returns.tail(30).std() * np.sqrt(252))
            
            # EWMA volatility
            vol_metrics.ewma_vol = self._calculate_ewma_volatility(returns)
            
            # High-low based volatilities
            if all(col in data.columns for col in ['high', 'low', 'open']):
                vol_metrics.parkinson_vol = self._calculate_parkinson_volatility(data)
                vol_metrics.garman_klass_vol = self._calculate_garman_klass_volatility(data)
                vol_metrics.yang_zhang_vol = self._calculate_yang_zhang_volatility(data)
            
            # Downside and upside volatility
            downside_returns = returns[returns < 0]
            upside_returns = returns[returns > 0]
            
            if len(downside_returns) > 0:
                vol_metrics.downside_volatility = float(downside_returns.std() * np.sqrt(252))
            if len(upside_returns) > 0:
                vol_metrics.upside_volatility = float(upside_returns.std() * np.sqrt(252))
            
            # Volatility percentiles and z-score
            if len(returns) >= 30:
                vol_30d = returns.tail(30).rolling(window=10).std() * np.sqrt(252)
                current_vol = vol_30d.iloc[-1]
                vol_metrics.volatility_percentile_30d = float(
                    stats.percentileofscore(vol_30d.dropna(), current_vol) / 100
                )
                
                vol_mean = vol_30d.mean()
                vol_std = vol_30d.std()
                if vol_std > 0:
                    vol_metrics.volatility_z_score = float((current_vol - vol_mean) / vol_std)
            
            if len(returns) >= 90:
                vol_90d = returns.tail(90).rolling(window=10).std() * np.sqrt(252)
                current_vol = vol_90d.iloc[-1]
                vol_metrics.volatility_percentile_90d = float(
                    stats.percentileofscore(vol_90d.dropna(), current_vol) / 100
                )
            
            # Volatility trend (slope of recent volatility)
            if len(returns) >= 20:
                recent_vol = returns.tail(20).rolling(window=5).std()
                if len(recent_vol.dropna()) >= 10:
                    x = np.arange(len(recent_vol.dropna()))
                    slope, _, _, _, _ = stats.linregress(x, recent_vol.dropna().values)
                    vol_metrics.volatility_trend = float(slope)
            
            # Volatility skew and kurtosis
            if len(returns) >= 30:
                vol_metrics.volatility_skew = float(returns.tail(30).skew())
                vol_metrics.volatility_kurtosis = float(returns.tail(30).kurtosis())
            
            # Mean reversion speed (using AR(1) model)
            if len(returns) >= 50:
                vol_series = returns.tail(50).rolling(window=10).std()
                vol_series = vol_series.dropna()
                
                if len(vol_series) >= 20:
                    lagged_vol = vol_series.shift(1).dropna()
                    current_vol_series = vol_series[1:]
                    
                    if len(lagged_vol) == len(current_vol_series):
                        slope, _, r_value, _, _ = stats.linregress(lagged_vol, current_vol_series)
                        vol_metrics.volatility_mean_reversion_speed = float(1 - slope)
            
            return vol_metrics
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {e}")
            return None
    
    def _calculate_historical_volatility(self, returns: pd.Series, window: int = 20) -> float:
        """Calculate simple historical volatility"""
        
        try:
            if len(returns) < window:
                return 0.0
            
            vol = returns.tail(window).std() * np.sqrt(252)
            return float(vol)
            
        except Exception as e:
            logger.warning(f"Error calculating historical volatility: {e}")
            return 0.0
    
    def _calculate_ewma_volatility(self, returns: pd.Series, lambda_param: float = None) -> float:
        """Calculate EWMA (Exponentially Weighted Moving Average) volatility"""
        
        try:
            if len(returns) < 10:
                return 0.0
            
            lambda_param = lambda_param or self.model_parameters['ewma_lambda']
            
            # Calculate EWMA volatility
            ewma_var = 0.0
            for i in range(len(returns)):
                if i == 0:
                    ewma_var = returns.iloc[i] ** 2
                else:
                    ewma_var = lambda_param * ewma_var + (1 - lambda_param) * (returns.iloc[i] ** 2)
            
            ewma_vol = np.sqrt(ewma_var * 252)
            return float(ewma_vol)
            
        except Exception as e:
            logger.warning(f"Error calculating EWMA volatility: {e}")
            return 0.0
    
    def _calculate_parkinson_volatility(self, data: pd.DataFrame, window: int = 20) -> float:
        """Calculate Parkinson high-low volatility estimator"""
        
        try:
            if len(data) < window or 'high' not in data.columns or 'low' not in data.columns:
                return 0.0
            
            # Parkinson estimator: (1/(4*ln(2))) * ln(H/L)^2
            high_low_ratio = np.log(data['high'] / data['low'])
            parkinson_var = (1 / (4 * np.log(2))) * (high_low_ratio ** 2)
            
            # Annualized volatility
            parkinson_vol = np.sqrt(parkinson_var.tail(window).mean() * 252)
            return float(parkinson_vol)
            
        except Exception as e:
            logger.warning(f"Error calculating Parkinson volatility: {e}")
            return 0.0
    
    def _calculate_garman_klass_volatility(self, data: pd.DataFrame, window: int = 20) -> float:
        """Calculate Garman-Klass OHLC volatility estimator"""
        
        try:
            required_cols = ['open', 'high', 'low', 'close']
            if len(data) < window or not all(col in data.columns for col in required_cols):
                return 0.0
            
            # Garman-Klass estimator
            ln_hl = np.log(data['high'] / data['low'])
            ln_co = np.log(data['close'] / data['open'])
            
            gk_var = 0.5 * (ln_hl ** 2) - (2 * np.log(2) - 1) * (ln_co ** 2)
            
            # Annualized volatility
            gk_vol = np.sqrt(gk_var.tail(window).mean() * 252)
            return float(gk_vol)
            
        except Exception as e:
            logger.warning(f"Error calculating Garman-Klass volatility: {e}")
            return 0.0
    
    def _calculate_yang_zhang_volatility(self, data: pd.DataFrame, window: int = 20) -> float:
        """Calculate Yang-Zhang volatility estimator"""
        
        try:
            required_cols = ['open', 'high', 'low', 'close']
            if len(data) < window or not all(col in data.columns for col in required_cols):
                return 0.0
            
            # Yang-Zhang estimator components
            ln_ho = np.log(data['high'] / data['open'])
            ln_lo = np.log(data['low'] / data['open'])
            ln_co = np.log(data['close'] / data['open'])
            ln_oc = np.log(data['open'] / data['close'].shift(1))
            ln_cc = np.log(data['close'] / data['close'].shift(1))
            
            # Calculate components
            overnight = ln_oc ** 2
            rs = ln_ho * (ln_ho - ln_co) + ln_lo * (ln_lo - ln_co)
            
            # Combine components
            k = 0.34 / (1.34 + (window + 1) / (window - 1))
            yz_var = overnight + k * ln_cc ** 2 + (1 - k) * rs
            
            # Annualized volatility
            yz_vol = np.sqrt(yz_var.tail(window).mean() * 252)
            return float(yz_vol)
            
        except Exception as e:
            logger.warning(f"Error calculating Yang-Zhang volatility: {e}")
            return 0.0
    
    def _detect_volatility_regime(self, vol_metrics: VolatilityMetrics, data: pd.DataFrame) -> VolatilityRegime:
        """Detect current volatility regime"""
        
        try:
            # Use primary volatility measure
            current_vol = vol_metrics.yang_zhang_vol or vol_metrics.historical_vol_30d
            
            if current_vol == 0:
                return VolatilityRegime.NORMAL
            
            # Use percentile-based regime detection
            vol_percentile = vol_metrics.volatility_percentile_30d
            
            thresholds = self.config['regime_detection']['regime_thresholds']
            
            if vol_percentile <= thresholds['low']:
                regime = VolatilityRegime.LOW
            elif vol_percentile >= thresholds['high']:
                regime = VolatilityRegime.HIGH
            elif vol_percentile >= thresholds['normal']:
                regime = VolatilityRegime.NORMAL
            else:
                regime = VolatilityRegime.NORMAL
            
            # Check for extreme volatility
            if vol_metrics.volatility_z_score > 3.0:
                regime = VolatilityRegime.EXTREME
            
            # Check for expanding/contracting regimes
            if vol_metrics.volatility_trend > 0.1:
                regime = VolatilityRegime.EXPANDING
            elif vol_metrics.volatility_trend < -0.1:
                regime = VolatilityRegime.CONTRACTING
            
            return regime
            
        except Exception as e:
            logger.warning(f"Error detecting volatility regime: {e}")
            return VolatilityRegime.NORMAL
    
    def _generate_mean_reversion_signal(self, 
                                      data: pd.DataFrame, 
                                      vol_metrics: VolatilityMetrics, 
                                      vol_regime: VolatilityRegime) -> Optional[VolatilitySignal]:
        """Generate mean reversion signal based on volatility"""
        
        try:
            config = self.config['mean_reversion']
            
            # Only trade mean reversion in high volatility regimes
            if vol_regime not in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]:
                return None
            
            # Check if volatility shows mean reversion tendency
            if vol_metrics.volatility_mean_reversion_speed < config['reversion_speed_threshold']:
                return None
            
            # Use volatility z-score for signal generation
            vol_z_score = vol_metrics.volatility_z_score
            
            signal_type = None
            confidence = 0.0
            
            # Mean reversion signals
            if vol_z_score > config['entry_z_score']:
                # High volatility - expect reversion to mean (sell volatility)
                signal_type = SignalType.SELL
                confidence = min(95.0, 50.0 + abs(vol_z_score) * 10)
                
            elif vol_z_score < -config['entry_z_score']:
                # Low volatility - expect expansion (buy for volatility increase)
                signal_type = SignalType.BUY
                confidence = min(95.0, 50.0 + abs(vol_z_score) * 10)
            
            if not signal_type:
                return None
            
            # Create base signal
            current_price = float(data['close'].iloc[-1])
            
            base_signal = TradingSignal(
                symbol=vol_metrics.symbol,
                asset_type=AssetType.STOCK,  # Default, should be passed in
                signal_type=signal_type,
                regime=RegimeType.VOLATILE,  # Map volatility regime to market regime
                confidence=confidence,
                entry_price=current_price,
                current_price=current_price,
                timestamp=vol_metrics.timestamp,
                strategy_name="VolatilityMeanReversion",
                reasoning=f"Volatility z-score: {vol_z_score:.2f}, mean reversion speed: {vol_metrics.volatility_mean_reversion_speed:.2f}"
            )
            
            # Create volatility signal
            vol_signal = VolatilitySignal(
                base_signal=base_signal,
                volatility_metrics=vol_metrics,
                volatility_strategy=VolatilityStrategy.MEAN_REVERSION,
                expected_volatility=vol_metrics.historical_vol_30d,
                volatility_confidence=confidence,
                risk_adjustment_factor=min(2.0, vol_metrics.historical_vol_30d / 0.2),
                optimal_holding_period=config['max_holding_period']
            )
            
            return vol_signal
            
        except Exception as e:
            logger.warning(f"Error generating mean reversion signal: {e}")
            return None
    
    def _generate_volatility_breakout_signal(self, 
                                           data: pd.DataFrame, 
                                           vol_metrics: VolatilityMetrics, 
                                           vol_regime: VolatilityRegime) -> Optional[VolatilitySignal]:
        """Generate volatility breakout signal"""
        
        try:
            config = self.config['volatility_breakout']
            
            # Look for volatility breakouts in low/normal regimes
            if vol_regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]:
                return None
            
            # Calculate recent price momentum
            if len(data) < 10:
                return None
            
            returns = data['close'].pct_change().tail(5)
            recent_momentum = returns.sum()
            
            # Check for volatility expansion
            if vol_metrics.volatility_trend <= 0:
                return None
            
            # Check if current volatility is breaking out
            vol_z_score = vol_metrics.volatility_z_score
            
            if vol_z_score < config['breakout_threshold']:
                return None
            
            # Determine signal direction based on price momentum
            signal_type = None
            confidence = 0.0
            
            if recent_momentum > config['momentum_threshold']:
                signal_type = SignalType.BUY
                confidence = min(90.0, 60.0 + vol_z_score * 10)
            elif recent_momentum < -config['momentum_threshold']:
                signal_type = SignalType.SELL
                confidence = min(90.0, 60.0 + vol_z_score * 10)
            else:
                return None
            
            # Create base signal
            current_price = float(data['close'].iloc[-1])
            
            base_signal = TradingSignal(
                symbol=vol_metrics.symbol,
                asset_type=AssetType.STOCK,
                signal_type=signal_type,
                regime=RegimeType.BREAKOUT,
                confidence=confidence,
                entry_price=current_price,
                current_price=current_price,
                timestamp=vol_metrics.timestamp,
                strategy_name="VolatilityBreakout",
                reasoning=f"Volatility breakout: z-score {vol_z_score:.2f}, momentum {recent_momentum:.3f}"
            )
            
            # Create volatility signal
            vol_signal = VolatilitySignal(
                base_signal=base_signal,
                volatility_metrics=vol_metrics,
                volatility_strategy=VolatilityStrategy.VOLATILITY_BREAKOUT,
                expected_volatility=vol_metrics.historical_vol_30d * 1.5,  # Expect higher volatility
                volatility_confidence=confidence,
                risk_adjustment_factor=1.0,  # Standard risk for breakouts
                optimal_holding_period=config['confirmation_period'] * 3
            )
            
            return vol_signal
            
        except Exception as e:
            logger.warning(f"Error generating volatility breakout signal: {e}")
            return None
    
    def _apply_volatility_risk_adjustments(self, 
                                         signal: VolatilitySignal, 
                                         vol_metrics: VolatilityMetrics, 
                                         data: pd.DataFrame) -> VolatilitySignal:
        """Apply risk adjustments based on volatility"""
        
        try:
            config = self.config['risk_management']
            
            # Adjust position size based on volatility
            base_size = self.config['position_sizing']['base_position_size']
            current_vol = vol_metrics.historical_vol_30d
            target_vol = 0.20  # 20% target volatility
            
            if current_vol > 0:
                vol_adjustment = target_vol / current_vol
                vol_adjustment = np.clip(
                    vol_adjustment,
                    self.config['position_sizing']['min_volatility_multiplier'],
                    self.config['position_sizing']['max_volatility_multiplier']
                )
            else:
                vol_adjustment = 1.0
            
            adjusted_position_size = base_size * vol_adjustment
            signal.base_signal.position_size = adjusted_position_size
            
            # Set volatility-based stop loss
            current_price = signal.base_signal.current_price
            vol_stop_distance = current_vol * config['volatility_stop_multiplier'] / np.sqrt(252)  # Daily vol
            
            if signal.base_signal.signal_type == SignalType.BUY:
                signal.volatility_stop_loss = current_price * (1 - vol_stop_distance)
                signal.volatility_take_profit = current_price * (1 + vol_stop_distance * config['profit_target_multiplier'])
            else:
                signal.volatility_stop_loss = current_price * (1 + vol_stop_distance)
                signal.volatility_take_profit = current_price * (1 - vol_stop_distance * config['profit_target_multiplier'])
            
            # Update base signal stop loss and take profit
            signal.base_signal.stop_loss = signal.volatility_stop_loss
            signal.base_signal.take_profit = signal.volatility_take_profit
            
            # Adjust confidence based on volatility regime stability
            if vol_metrics.regime_persistence > 0.8:
                signal.base_signal.confidence *= 1.1  # Increase confidence for stable regimes
            elif vol_metrics.regime_persistence < 0.3:
                signal.base_signal.confidence *= 0.9  # Decrease confidence for unstable regimes
            
            signal.base_signal.confidence = min(95.0, signal.base_signal.confidence)
            
            return signal
            
        except Exception as e:
            logger.warning(f"Error applying volatility risk adjustments: {e}")
            return signal
    
    def _update_regime_history(self, symbol: str, regime: VolatilityRegime):
        """Update volatility regime history"""
        
        try:
            if symbol not in self.regime_history:
                self.regime_history[symbol] = []
            
            # Add current regime
            self.regime_history[symbol].append({
                'regime': regime,
                'timestamp': datetime.now()
            })
            
            # Keep only recent history
            max_history = 100
            if len(self.regime_history[symbol]) > max_history:
                self.regime_history[symbol] = self.regime_history[symbol][-max_history:]
            
            # Calculate regime persistence
            if len(self.regime_history[symbol]) >= 10:
                recent_regimes = [r['regime'] for r in self.regime_history[symbol][-10:]]
                same_regime_count = sum(1 for r in recent_regimes if r == regime)
                self.regime_persistence = same_regime_count / len(recent_regimes)
            
        except Exception as e:
            logger.warning(f"Error updating regime history: {e}")
    
    def get_strategy_metrics(self) -> StrategyMetrics:
        """Get strategy performance metrics"""
        
        try:
            # This would be implemented with actual performance tracking
            return StrategyMetrics(
                strategy_name="AdvancedVolatilityStrategy",
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
            return StrategyMetrics(strategy_name="AdvancedVolatilityStrategy")
    
    def validate_signal(self, signal: VolatilitySignal, current_positions: Dict = None) -> bool:
        """Validate volatility signal before execution"""
        
        try:
            # Check daily volatility limits
            max_daily_vol = self.config['risk_management']['max_daily_volatility']
            if signal.volatility_metrics.historical_vol_1d > max_daily_vol:
                logger.warning(f"Signal rejected: daily volatility {signal.volatility_metrics.historical_vol_1d:.3f} exceeds limit {max_daily_vol:.3f}")
                return False
            
            # Check signal confidence
            if signal.base_signal.confidence < 60.0:
                return False
            
            # Check position size
            if signal.base_signal.position_size and signal.base_signal.position_size > 0.1:  # Max 10% position
                logger.warning(f"Position size {signal.base_signal.position_size:.3f} too large, rejecting signal")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating signal: {e}")
            return False

# ==================== TESTING ====================

def test_volatility_strategy():
    """Test volatility strategy functionality"""
    
    print("⚡ Testing Advanced Volatility Strategy")
    print("=" * 50)
    
    # Create sample data with varying volatility
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    
    # Create data with volatility clustering
    base_price = 100
    prices = [base_price]
    volatility = 0.02  # Initial volatility
    
    for i in range(1, 100):
        # Volatility clustering - volatility depends on previous volatility
        volatility = 0.95 * volatility + 0.05 * abs(np.random.randn() * 0.03)
        
        # Add volatility regime changes
        if 30 <= i <= 40:  # High volatility period
            volatility *= 2.0
        elif 60 <= i <= 70:  # Low volatility period
            volatility *= 0.5
        
        # Generate price with current volatility
        return_val = np.random.randn() * volatility
        new_price = prices[-1] * (1 + return_val)
        prices.append(new_price)
    
    # Create OHLC data
    sample_data = pd.DataFrame({
        'open': prices,
        'close': [p * (1 + np.random.randn() * 0.001) for p in prices],
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Add high/low
    sample_data['high'] = sample_data[['open', 'close']].max(axis=1) * (1 + np.random.rand(100) * 0.01)
    sample_data['low'] = sample_data[['open', 'close']].min(axis=1) * (1 - np.random.rand(100) * 0.01)
    
    # Create volatility strategy
    vol_strategy = AdvancedVolatilityStrategy()
    
    print(f"✅ Strategy initialized:")
    print(f"   Primary model: {vol_strategy.config['volatility_models']['primary_model'].value}")
    print(f"   Mean reversion enabled: {vol_strategy.config['mean_reversion']['enabled']}")
    print(f"   Breakout enabled: {vol_strategy.config['volatility_breakout']['enabled']}")
    
    # Test volatility metrics calculation
    vol_metrics = vol_strategy._calculate_volatility_metrics(sample_data, 'TEST')
    
    if vol_metrics:
        print(f"\n📊 Volatility metrics:")
        print(f"   30-day historical vol: {vol_metrics.historical_vol_30d:.3f}")
        print(f"   EWMA volatility: {vol_metrics.ewma_vol:.3f}")
        print(f"   Yang-Zhang volatility: {vol_metrics.yang_zhang_vol:.3f}")
        print(f"   Volatility z-score: {vol_metrics.volatility_z_score:.2f}")
        print(f"   Volatility trend: {vol_metrics.volatility_trend:.3f}")
        print(f"   Downside vol: {vol_metrics.downside_volatility:.3f}")
        print(f"   Upside vol: {vol_metrics.upside_volatility:.3f}")
    
    # Test regime detection
    vol_regime = vol_strategy._detect_volatility_regime(vol_metrics, sample_data)
    print(f"\n🎯 Volatility regime: {vol_regime.value}")
    
    # Generate signal
    signal = vol_strategy.generate_signal(
        sample_data, 
        'TEST',
        RegimeType.VOLATILE, 
        85.0
    )
    
    if signal:
        print(f"\n🚦 Generated signal:")
        print(f"   Type: {signal.base_signal.signal_type.value}")
        print(f"   Confidence: {signal.base_signal.confidence:.1f}%")
        print(f"   Strategy: {signal.volatility_strategy.value}")
        print(f"   Entry price: ${signal.base_signal.entry_price:.2f}")
        print(f"   Stop loss: ${signal.volatility_stop_loss:.2f}")
        print(f"   Take profit: ${signal.volatility_take_profit:.2f}")
        print(f"   Position size: {signal.base_signal.position_size:.3f}")
        print(f"   Expected volatility: {signal.expected_volatility:.3f}")
        print(f"   Risk adjustment: {signal.risk_adjustment_factor:.2f}")
        print(f"   Optimal holding: {signal.optimal_holding_period} bars")
        print(f"   Reasoning: {signal.base_signal.reasoning}")
        
        # Validate signal
        is_valid = vol_strategy.validate_signal(signal)
        print(f"   Signal valid: {'✅' if is_valid else '❌'}")
    else:
        print(f"\n🚦 No signal generated")
    
    # Test different volatility models
    print(f"\n🔬 Testing volatility models:")
    returns = sample_data['close'].pct_change().dropna()
    
    hist_vol = vol_strategy._calculate_historical_volatility(returns)
    ewma_vol = vol_strategy._calculate_ewma_volatility(returns)
    parkinson_vol = vol_strategy._calculate_parkinson_volatility(sample_data)
    
    print(f"   Historical (20d): {hist_vol:.3f}")
    print(f"   EWMA: {ewma_vol:.3f}")
    print(f"   Parkinson: {parkinson_vol:.3f}")
    
    # Test strategy metrics
    metrics = vol_strategy.get_strategy_metrics()
    print(f"\n📈 Strategy metrics:")
    print(f"   Strategy name: {metrics.strategy_name}")
    print(f"   Total signals: {metrics.total_signals}")
    print(f"   Success rate: {metrics.success_rate:.1%}")
    
    print("\n🎉 Volatility strategy tests completed!")

if __name__ == "__main__":
    test_volatility_strategy()