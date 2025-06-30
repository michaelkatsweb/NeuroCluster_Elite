#!/usr/bin/env python3
"""
File: strategy_selector.py
Path: NeuroCluster-Elite/src/trading/strategy_selector.py
Description: Intelligent strategy selection based on market conditions and regime analysis

This module implements sophisticated strategy selection logic that analyzes market
conditions, asset characteristics, volatility regimes, and performance metrics to
dynamically select the most appropriate trading strategy for current conditions.

Features:
- Multi-factor strategy selection based on market regime
- Asset-specific strategy adaptation
- Performance-based strategy ranking and selection
- Dynamic strategy switching with hysteresis
- Risk-adjusted strategy allocation
- Strategy ensemble and blending capabilities
- Backtesting-based strategy validation
- Real-time strategy performance monitoring

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
from collections import defaultdict, deque
import math

# Import our modules
try:
    from src.core.neurocluster_elite import RegimeType, AssetType, MarketData
    from src.trading.strategies.base_strategy import BaseStrategy, TradingSignal, SignalType, StrategyMetrics
    from src.trading.strategies.bull_strategy import BullMarketStrategy
    from src.trading.strategies.bear_strategy import BearMarketStrategy
    from src.trading.strategies.volatility_strategy import AdvancedVolatilityStrategy
    from src.trading.strategies.breakout_strategy import AdvancedBreakoutStrategy
    from src.utils.config_manager import ConfigManager
    from src.utils.helpers import calculate_sharpe_ratio, calculate_max_drawdown, format_percentage
except ImportError:
    # Fallback for testing
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== STRATEGY SELECTION ENUMS AND STRUCTURES ====================

class SelectionMethod(Enum):
    """Strategy selection methods"""
    BEST_PERFORMER = "best_performer"
    REGIME_BASED = "regime_based"
    ENSEMBLE = "ensemble"
    ADAPTIVE = "adaptive"
    RISK_ADJUSTED = "risk_adjusted"
    MOMENTUM_BASED = "momentum_based"

class StrategyWeight(Enum):
    """Strategy weighting schemes"""
    EQUAL = "equal"
    PERFORMANCE = "performance"
    SHARPE_RATIO = "sharpe_ratio"
    RECENT_PERFORMANCE = "recent_performance"
    CONFIDENCE_BASED = "confidence_based"

@dataclass
class StrategyScore:
    """Strategy performance and suitability score"""
    strategy_name: str
    suitability_score: float  # 0-100, how suitable for current conditions
    performance_score: float  # 0-100, recent performance quality
    risk_score: float        # 0-100, risk-adjusted performance
    confidence_score: float  # 0-100, strategy confidence
    composite_score: float   # 0-100, overall weighted score
    
    # Performance metrics
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_return: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    
    # Selection factors
    regime_alignment: float = 0.0
    asset_fit: float = 0.0
    market_conditions_fit: float = 0.0
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    sample_size: int = 0

@dataclass
class StrategyAllocation:
    """Strategy allocation in ensemble"""
    strategy_name: str
    weight: float  # 0-1
    confidence: float
    expected_return: float
    expected_risk: float
    allocation_reason: str = ""

@dataclass
class SelectionDecision:
    """Strategy selection decision"""
    selected_strategies: List[str]
    selection_method: SelectionMethod
    allocations: List[StrategyAllocation]
    decision_confidence: float
    expected_portfolio_return: float
    expected_portfolio_risk: float
    
    # Decision context
    market_regime: RegimeType
    asset_type: AssetType
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Selection rationale
    selection_reasoning: str = ""
    risk_factors: List[str] = field(default_factory=list)
    
    # Metadata
    decision_time: datetime = field(default_factory=datetime.now)
    review_time: Optional[datetime] = None

# ==================== STRATEGY SELECTOR CLASS ====================

class IntelligentStrategySelector:
    """
    Intelligent strategy selector for dynamic trading strategy selection
    
    This class analyzes market conditions, strategy performance, and risk factors
    to select the most appropriate trading strategy or ensemble of strategies
    for current market conditions.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize strategy selector"""
        
        self.config = config or self._get_default_config()
        self.strategies = {}
        self.strategy_scores = {}
        self.performance_history = defaultdict(list)
        self.allocation_history = []
        self.current_allocation = None
        
        # Initialize strategies
        self._initialize_strategies()
        
        # Selection state
        self.last_selection_time = None
        self.selection_stability_buffer = deque(maxlen=10)
        
        logger.info("Intelligent Strategy Selector initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'selection_method': SelectionMethod.ADAPTIVE,
            'weighting_scheme': StrategyWeight.PERFORMANCE,
            'performance_lookback_days': 30,
            'min_sample_size': 10,
            'selection_frequency_minutes': 60,  # Re-evaluate every hour
            
            'regime_strategy_mapping': {
                RegimeType.BULL: ['BullMarketStrategy', 'AdvancedBreakoutStrategy'],
                RegimeType.BEAR: ['BearMarketStrategy', 'AdvancedVolatilityStrategy'],
                RegimeType.SIDEWAYS: ['AdvancedVolatilityStrategy', 'RangeTradingStrategy'],
                RegimeType.VOLATILE: ['AdvancedVolatilityStrategy', 'AdvancedBreakoutStrategy'],
                RegimeType.BREAKOUT: ['AdvancedBreakoutStrategy', 'BullMarketStrategy'],
                RegimeType.BREAKDOWN: ['BearMarketStrategy', 'AdvancedVolatilityStrategy']
            },
            
            'asset_strategy_preferences': {
                AssetType.STOCK: {
                    'preferred': ['BullMarketStrategy', 'BearMarketStrategy'],
                    'secondary': ['AdvancedBreakoutStrategy', 'AdvancedVolatilityStrategy']
                },
                AssetType.CRYPTO: {
                    'preferred': ['AdvancedVolatilityStrategy', 'AdvancedBreakoutStrategy'],
                    'secondary': ['BullMarketStrategy', 'BearMarketStrategy']
                },
                AssetType.FOREX: {
                    'preferred': ['AdvancedVolatilityStrategy', 'RangeTradingStrategy'],
                    'secondary': ['AdvancedBreakoutStrategy']
                }
            },
            
            'scoring_weights': {
                'suitability': 0.3,
                'performance': 0.25,
                'risk_adjusted': 0.25,
                'confidence': 0.2
            },
            
            'ensemble_settings': {
                'max_strategies': 3,
                'min_weight': 0.1,
                'rebalance_threshold': 0.1,
                'correlation_penalty': 0.15
            },
            
            'switching_rules': {
                'min_improvement': 0.05,  # 5% improvement needed to switch
                'stability_period': 5,    # Require 5 consecutive selections
                'hysteresis_factor': 0.1  # 10% hysteresis to prevent oscillation
            },
            
            'performance_thresholds': {
                'excellent': 80.0,
                'good': 60.0,
                'acceptable': 40.0,
                'poor': 20.0
            }
        }
    
    def _initialize_strategies(self):
        """Initialize available trading strategies"""
        
        try:
            # Initialize available strategies
            self.strategies = {
                'BullMarketStrategy': BullMarketStrategy(),
                'BearMarketStrategy': BearMarketStrategy(),
                'AdvancedVolatilityStrategy': AdvancedVolatilityStrategy(),
                'AdvancedBreakoutStrategy': AdvancedBreakoutStrategy(),
                # Add more strategies as they become available
            }
            
            # Initialize strategy scores
            for strategy_name in self.strategies.keys():
                self.strategy_scores[strategy_name] = StrategyScore(
                    strategy_name=strategy_name,
                    suitability_score=50.0,
                    performance_score=50.0,
                    risk_score=50.0,
                    confidence_score=50.0,
                    composite_score=50.0
                )
            
            logger.info(f"Initialized {len(self.strategies)} trading strategies")
            
        except Exception as e:
            logger.error(f"Error initializing strategies: {e}")
            self.strategies = {}
    
    def select_strategy(self, 
                       market_data: pd.DataFrame,
                       symbol: str,
                       regime: RegimeType,
                       asset_type: AssetType,
                       additional_context: Dict = None) -> SelectionDecision:
        """
        Select optimal trading strategy based on current conditions
        
        Args:
            market_data: Recent market data for analysis
            symbol: Asset symbol
            regime: Current market regime
            asset_type: Type of asset
            additional_context: Additional market context
            
        Returns:
            SelectionDecision with selected strategy and rationale
        """
        
        try:
            # Check if we need to re-evaluate
            if not self._should_reevaluate():
                if self.current_allocation:
                    return self.current_allocation
            
            # Update strategy performance scores
            self._update_strategy_scores(market_data, symbol, regime, asset_type)
            
            # Get market conditions analysis
            market_conditions = self._analyze_market_conditions(market_data, additional_context)
            
            # Select strategy based on configured method
            selection_method = self.config['selection_method']
            
            if selection_method == SelectionMethod.REGIME_BASED:
                decision = self._select_by_regime(regime, asset_type, market_conditions)
            elif selection_method == SelectionMethod.BEST_PERFORMER:
                decision = self._select_best_performer(market_conditions)
            elif selection_method == SelectionMethod.ENSEMBLE:
                decision = self._select_ensemble(regime, asset_type, market_conditions)
            elif selection_method == SelectionMethod.ADAPTIVE:
                decision = self._select_adaptive(regime, asset_type, market_conditions)
            elif selection_method == SelectionMethod.RISK_ADJUSTED:
                decision = self._select_risk_adjusted(market_conditions)
            else:
                decision = self._select_best_performer(market_conditions)
            
            # Apply stability checks and hysteresis
            decision = self._apply_stability_checks(decision)
            
            # Update allocation history
            self.current_allocation = decision
            self.allocation_history.append(decision)
            self.last_selection_time = datetime.now()
            
            # Limit history size
            if len(self.allocation_history) > 100:
                self.allocation_history = self.allocation_history[-100:]
            
            logger.info(f"Selected strategy: {decision.selected_strategies} "
                       f"(method: {selection_method.value}, confidence: {decision.decision_confidence:.1f}%)")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error selecting strategy: {e}")
            
            # Return fallback decision
            return self._create_fallback_decision(regime, asset_type)
    
    def _update_strategy_scores(self, 
                              market_data: pd.DataFrame,
                              symbol: str,
                              regime: RegimeType,
                              asset_type: AssetType):
        """Update strategy performance and suitability scores"""
        
        try:
            for strategy_name, strategy in self.strategies.items():
                score = self.strategy_scores[strategy_name]
                
                # Update suitability score
                score.suitability_score = self._calculate_suitability_score(
                    strategy_name, regime, asset_type, market_data
                )
                
                # Update performance score
                score.performance_score = self._calculate_performance_score(
                    strategy_name, market_data
                )
                
                # Update risk score
                score.risk_score = self._calculate_risk_score(
                    strategy_name, market_data
                )
                
                # Update confidence score
                score.confidence_score = self._calculate_confidence_score(
                    strategy_name, market_data
                )
                
                # Calculate composite score
                weights = self.config['scoring_weights']
                score.composite_score = (
                    score.suitability_score * weights['suitability'] +
                    score.performance_score * weights['performance'] +
                    score.risk_score * weights['risk_adjusted'] +
                    score.confidence_score * weights['confidence']
                )
                
                score.last_updated = datetime.now()
                
        except Exception as e:
            logger.warning(f"Error updating strategy scores: {e}")
    
    def _calculate_suitability_score(self, 
                                   strategy_name: str,
                                   regime: RegimeType,
                                   asset_type: AssetType,
                                   market_data: pd.DataFrame) -> float:
        """Calculate how suitable a strategy is for current conditions"""
        
        try:
            score = 50.0  # Base score
            
            # Regime alignment score
            regime_preferences = self.config['regime_strategy_mapping'].get(regime, [])
            if strategy_name in regime_preferences:
                regime_rank = regime_preferences.index(strategy_name)
                score += 30.0 * (1 - regime_rank / len(regime_preferences))
            
            # Asset type alignment
            asset_prefs = self.config['asset_strategy_preferences'].get(asset_type, {})
            if strategy_name in asset_prefs.get('preferred', []):
                score += 15.0
            elif strategy_name in asset_prefs.get('secondary', []):
                score += 8.0
            
            # Market condition alignment
            volatility = market_data['close'].pct_change().std() * np.sqrt(252) if len(market_data) > 1 else 0.2
            
            if 'Volatility' in strategy_name:
                # Volatility strategies work better in high volatility
                if volatility > 0.3:
                    score += 10.0
                elif volatility < 0.15:
                    score -= 10.0
            elif 'Breakout' in strategy_name:
                # Breakout strategies work better with medium volatility
                if 0.2 <= volatility <= 0.4:
                    score += 10.0
            elif 'Bull' in strategy_name or 'Bear' in strategy_name:
                # Trend strategies work better in low-medium volatility
                if volatility < 0.25:
                    score += 10.0
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.warning(f"Error calculating suitability score for {strategy_name}: {e}")
            return 50.0
    
    def _calculate_performance_score(self, strategy_name: str, market_data: pd.DataFrame) -> float:
        """Calculate recent performance score for strategy"""
        
        try:
            # Get recent performance history
            lookback_days = self.config['performance_lookback_days']
            cutoff_time = datetime.now() - timedelta(days=lookback_days)
            
            recent_performance = [
                perf for perf in self.performance_history[strategy_name]
                if perf['timestamp'] >= cutoff_time
            ]
            
            if not recent_performance or len(recent_performance) < self.config['min_sample_size']:
                return 50.0  # Neutral score for insufficient data
            
            # Calculate performance metrics
            returns = [perf['return'] for perf in recent_performance]
            win_rate = sum(1 for r in returns if r > 0) / len(returns)
            avg_return = np.mean(returns)
            
            # Score based on win rate and average return
            score = 50.0
            score += win_rate * 30.0  # 0-30 points for win rate
            score += max(-20.0, min(20.0, avg_return * 1000))  # -20 to +20 for returns
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.warning(f"Error calculating performance score for {strategy_name}: {e}")
            return 50.0
    
    def _calculate_risk_score(self, strategy_name: str, market_data: pd.DataFrame) -> float:
        """Calculate risk-adjusted score for strategy"""
        
        try:
            # Get recent performance history
            lookback_days = self.config['performance_lookback_days']
            cutoff_time = datetime.now() - timedelta(days=lookback_days)
            
            recent_performance = [
                perf for perf in self.performance_history[strategy_name]
                if perf['timestamp'] >= cutoff_time
            ]
            
            if not recent_performance or len(recent_performance) < self.config['min_sample_size']:
                return 50.0
            
            returns = [perf['return'] for perf in recent_performance]
            
            if not returns:
                return 50.0
            
            # Calculate risk metrics
            volatility = np.std(returns)
            sharpe_ratio = np.mean(returns) / volatility if volatility > 0 else 0
            max_dd = self._calculate_max_drawdown(returns)
            
            # Score based on risk-adjusted metrics
            score = 50.0
            score += max(-25.0, min(25.0, sharpe_ratio * 25))  # Sharpe ratio component
            score += max(-15.0, min(15.0, -max_dd * 100))      # Max drawdown component
            score += max(-10.0, min(10.0, -volatility * 200))  # Volatility component
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.warning(f"Error calculating risk score for {strategy_name}: {e}")
            return 50.0
    
    def _calculate_confidence_score(self, strategy_name: str, market_data: pd.DataFrame) -> float:
        """Calculate confidence score based on consistency and sample size"""
        
        try:
            # Get recent performance history
            lookback_days = self.config['performance_lookback_days']
            cutoff_time = datetime.now() - timedelta(days=lookback_days)
            
            recent_performance = [
                perf for perf in self.performance_history[strategy_name]
                if perf['timestamp'] >= cutoff_time
            ]
            
            if not recent_performance:
                return 20.0  # Low confidence for no data
            
            sample_size = len(recent_performance)
            
            # Sample size component (0-40 points)
            min_sample = self.config['min_sample_size']
            sample_score = min(40.0, (sample_size / min_sample) * 20.0)
            
            # Consistency component (0-40 points)
            if sample_size >= 5:
                returns = [perf['return'] for perf in recent_performance]
                consistency = 1.0 - (np.std(returns) / (abs(np.mean(returns)) + 0.01))
                consistency_score = max(0.0, min(40.0, consistency * 40.0))
            else:
                consistency_score = 20.0
            
            # Recent activity component (0-20 points)
            if recent_performance:
                last_activity = recent_performance[-1]['timestamp']
                days_since = (datetime.now() - last_activity).days
                activity_score = max(0.0, 20.0 - days_since)
            else:
                activity_score = 0.0
            
            total_score = sample_score + consistency_score + activity_score
            return max(0.0, min(100.0, total_score))
            
        except Exception as e:
            logger.warning(f"Error calculating confidence score for {strategy_name}: {e}")
            return 50.0
    
    def _analyze_market_conditions(self, market_data: pd.DataFrame, additional_context: Dict = None) -> Dict[str, Any]:
        """Analyze current market conditions"""
        
        try:
            conditions = {}
            
            if len(market_data) > 1:
                # Volatility
                returns = market_data['close'].pct_change().dropna()
                conditions['volatility'] = returns.std() * np.sqrt(252)
                
                # Trend strength
                if len(market_data) >= 20:
                    sma_20 = market_data['close'].rolling(20).mean()
                    trend_strength = abs(market_data['close'].iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1]
                    conditions['trend_strength'] = trend_strength
                
                # Volume trend
                if 'volume' in market_data.columns and len(market_data) >= 10:
                    avg_volume = market_data['volume'].rolling(10).mean()
                    volume_trend = market_data['volume'].iloc[-1] / avg_volume.iloc[-1]
                    conditions['volume_trend'] = volume_trend
                
                # Price momentum
                if len(market_data) >= 5:
                    momentum = (market_data['close'].iloc[-1] / market_data['close'].iloc[-5]) - 1
                    conditions['momentum'] = momentum
            
            # Add additional context
            if additional_context:
                conditions.update(additional_context)
            
            return conditions
            
        except Exception as e:
            logger.warning(f"Error analyzing market conditions: {e}")
            return {}
    
    # ==================== SELECTION METHODS ====================
    
    def _select_by_regime(self, 
                         regime: RegimeType,
                         asset_type: AssetType,
                         market_conditions: Dict) -> SelectionDecision:
        """Select strategy based on market regime"""
        
        try:
            # Get preferred strategies for regime
            preferred_strategies = self.config['regime_strategy_mapping'].get(regime, [])
            
            if not preferred_strategies:
                return self._select_best_performer(market_conditions)
            
            # Filter by available strategies
            available_strategies = [s for s in preferred_strategies if s in self.strategies]
            
            if not available_strategies:
                return self._select_best_performer(market_conditions)
            
            # Select best scoring strategy from preferred list
            best_strategy = max(
                available_strategies,
                key=lambda s: self.strategy_scores[s].composite_score
            )
            
            allocation = StrategyAllocation(
                strategy_name=best_strategy,
                weight=1.0,
                confidence=self.strategy_scores[best_strategy].confidence_score,
                expected_return=self.strategy_scores[best_strategy].avg_return,
                expected_risk=self.strategy_scores[best_strategy].volatility,
                allocation_reason=f"Best performer for {regime.value} regime"
            )
            
            decision = SelectionDecision(
                selected_strategies=[best_strategy],
                selection_method=SelectionMethod.REGIME_BASED,
                allocations=[allocation],
                decision_confidence=self.strategy_scores[best_strategy].composite_score,
                expected_portfolio_return=allocation.expected_return,
                expected_portfolio_risk=allocation.expected_risk,
                market_regime=regime,
                asset_type=asset_type,
                market_conditions=market_conditions,
                selection_reasoning=f"Selected {best_strategy} as best strategy for {regime.value} regime"
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in regime-based selection: {e}")
            return self._create_fallback_decision(regime, asset_type)
    
    def _select_best_performer(self, market_conditions: Dict) -> SelectionDecision:
        """Select single best performing strategy"""
        
        try:
            if not self.strategy_scores:
                return self._create_fallback_decision(RegimeType.BULL, AssetType.STOCK)
            
            # Find best scoring strategy
            best_strategy = max(
                self.strategy_scores.keys(),
                key=lambda s: self.strategy_scores[s].composite_score
            )
            
            best_score = self.strategy_scores[best_strategy]
            
            allocation = StrategyAllocation(
                strategy_name=best_strategy,
                weight=1.0,
                confidence=best_score.confidence_score,
                expected_return=best_score.avg_return,
                expected_risk=best_score.volatility,
                allocation_reason="Highest composite score"
            )
            
            decision = SelectionDecision(
                selected_strategies=[best_strategy],
                selection_method=SelectionMethod.BEST_PERFORMER,
                allocations=[allocation],
                decision_confidence=best_score.composite_score,
                expected_portfolio_return=allocation.expected_return,
                expected_portfolio_risk=allocation.expected_risk,
                market_conditions=market_conditions,
                selection_reasoning=f"Selected {best_strategy} with highest composite score ({best_score.composite_score:.1f})"
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in best performer selection: {e}")
            return self._create_fallback_decision(RegimeType.BULL, AssetType.STOCK)
    
    def _select_ensemble(self, 
                        regime: RegimeType,
                        asset_type: AssetType,
                        market_conditions: Dict) -> SelectionDecision:
        """Select ensemble of strategies with optimal weights"""
        
        try:
            # Get top strategies
            max_strategies = self.config['ensemble_settings']['max_strategies']
            min_weight = self.config['ensemble_settings']['min_weight']
            
            # Sort strategies by composite score
            sorted_strategies = sorted(
                self.strategy_scores.items(),
                key=lambda x: x[1].composite_score,
                reverse=True
            )
            
            # Select top strategies that meet minimum score
            selected_strategies = []
            for strategy_name, score in sorted_strategies[:max_strategies]:
                if score.composite_score >= 40.0:  # Minimum acceptable score
                    selected_strategies.append((strategy_name, score))
            
            if not selected_strategies:
                return self._select_best_performer(market_conditions)
            
            # Calculate weights based on scores
            total_score = sum(score.composite_score for _, score in selected_strategies)
            
            allocations = []
            total_weight = 0.0
            
            for strategy_name, score in selected_strategies:
                weight = (score.composite_score / total_score)
                
                # Ensure minimum weight
                weight = max(min_weight, weight)
                total_weight += weight
                
                allocation = StrategyAllocation(
                    strategy_name=strategy_name,
                    weight=weight,
                    confidence=score.confidence_score,
                    expected_return=score.avg_return,
                    expected_risk=score.volatility,
                    allocation_reason=f"Ensemble member (score: {score.composite_score:.1f})"
                )
                allocations.append(allocation)
            
            # Normalize weights
            for allocation in allocations:
                allocation.weight /= total_weight
            
            # Calculate portfolio metrics
            portfolio_return = sum(alloc.weight * alloc.expected_return for alloc in allocations)
            portfolio_risk = np.sqrt(sum(alloc.weight**2 * alloc.expected_risk**2 for alloc in allocations))
            
            decision = SelectionDecision(
                selected_strategies=[alloc.strategy_name for alloc in allocations],
                selection_method=SelectionMethod.ENSEMBLE,
                allocations=allocations,
                decision_confidence=sum(alloc.weight * alloc.confidence for alloc in allocations),
                expected_portfolio_return=portfolio_return,
                expected_portfolio_risk=portfolio_risk,
                market_regime=regime,
                asset_type=asset_type,
                market_conditions=market_conditions,
                selection_reasoning=f"Ensemble of {len(allocations)} strategies for diversification"
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in ensemble selection: {e}")
            return self._select_best_performer(market_conditions)
    
    def _select_adaptive(self, 
                        regime: RegimeType,
                        asset_type: AssetType,
                        market_conditions: Dict) -> SelectionDecision:
        """Adaptive selection based on market conditions"""
        
        try:
            # Start with regime-based selection
            regime_decision = self._select_by_regime(regime, asset_type, market_conditions)
            
            # Check if ensemble would be better
            ensemble_decision = self._select_ensemble(regime, asset_type, market_conditions)
            
            # Use ensemble if it has significantly higher expected return with acceptable risk
            if (len(ensemble_decision.allocations) > 1 and
                ensemble_decision.expected_portfolio_return > regime_decision.expected_portfolio_return * 1.1 and
                ensemble_decision.expected_portfolio_risk < regime_decision.expected_portfolio_risk * 1.2):
                
                ensemble_decision.selection_method = SelectionMethod.ADAPTIVE
                ensemble_decision.selection_reasoning = "Adaptive: Ensemble selected for better risk-return profile"
                return ensemble_decision
            
            # Otherwise use regime-based selection
            regime_decision.selection_method = SelectionMethod.ADAPTIVE
            regime_decision.selection_reasoning = "Adaptive: Regime-based selection preferred"
            return regime_decision
            
        except Exception as e:
            logger.error(f"Error in adaptive selection: {e}")
            return self._select_best_performer(market_conditions)
    
    def _select_risk_adjusted(self, market_conditions: Dict) -> SelectionDecision:
        """Select strategy based on risk-adjusted performance"""
        
        try:
            if not self.strategy_scores:
                return self._create_fallback_decision(RegimeType.BULL, AssetType.STOCK)
            
            # Find strategy with best risk score
            best_strategy = max(
                self.strategy_scores.keys(),
                key=lambda s: self.strategy_scores[s].risk_score
            )
            
            best_score = self.strategy_scores[best_strategy]
            
            allocation = StrategyAllocation(
                strategy_name=best_strategy,
                weight=1.0,
                confidence=best_score.confidence_score,
                expected_return=best_score.avg_return,
                expected_risk=best_score.volatility,
                allocation_reason="Best risk-adjusted performance"
            )
            
            decision = SelectionDecision(
                selected_strategies=[best_strategy],
                selection_method=SelectionMethod.RISK_ADJUSTED,
                allocations=[allocation],
                decision_confidence=best_score.risk_score,
                expected_portfolio_return=allocation.expected_return,
                expected_portfolio_risk=allocation.expected_risk,
                market_conditions=market_conditions,
                selection_reasoning=f"Selected {best_strategy} for best risk-adjusted performance"
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in risk-adjusted selection: {e}")
            return self._create_fallback_decision(RegimeType.BULL, AssetType.STOCK)
    
    # ==================== UTILITY METHODS ====================
    
    def _should_reevaluate(self) -> bool:
        """Check if strategy selection should be re-evaluated"""
        
        try:
            if not self.last_selection_time:
                return True
            
            frequency_minutes = self.config['selection_frequency_minutes']
            time_since_last = (datetime.now() - self.last_selection_time).total_seconds() / 60
            
            return time_since_last >= frequency_minutes
            
        except Exception as e:
            logger.warning(f"Error checking re-evaluation: {e}")
            return True
    
    def _apply_stability_checks(self, decision: SelectionDecision) -> SelectionDecision:
        """Apply stability checks and hysteresis to prevent excessive switching"""
        
        try:
            # Add current decision to stability buffer
            self.selection_stability_buffer.append(decision.selected_strategies)
            
            # Check for stability
            stability_period = self.config['switching_rules']['stability_period']
            min_improvement = self.config['switching_rules']['min_improvement']
            
            if len(self.selection_stability_buffer) >= stability_period:
                # Check if same strategy has been selected consistently
                recent_selections = list(self.selection_stability_buffer)[-stability_period:]
                
                # Find most common selection
                selection_counts = {}
                for selection in recent_selections:
                    key = tuple(sorted(selection))
                    selection_counts[key] = selection_counts.get(key, 0) + 1
                
                most_common = max(selection_counts.items(), key=lambda x: x[1])
                
                # If current selection is stable, use it
                if most_common[1] >= stability_period * 0.6:  # 60% consistency
                    stable_strategies = list(most_common[0])
                    
                    if set(stable_strategies) != set(decision.selected_strategies):
                        # Override with stable selection
                        decision.selected_strategies = stable_strategies
                        decision.selection_reasoning += " (Stability override applied)"
            
            # Apply hysteresis if there's a current allocation
            if self.current_allocation:
                hysteresis_factor = self.config['switching_rules']['hysteresis_factor']
                current_strategies = set(self.current_allocation.selected_strategies)
                new_strategies = set(decision.selected_strategies)
                
                # If strategies are different, check if improvement is significant
                if current_strategies != new_strategies:
                    current_return = self.current_allocation.expected_portfolio_return
                    new_return = decision.expected_portfolio_return
                    
                    improvement = (new_return - current_return) / abs(current_return) if current_return != 0 else 0
                    
                    if improvement < min_improvement + hysteresis_factor:
                        # Not enough improvement, stick with current
                        return self.current_allocation
            
            return decision
            
        except Exception as e:
            logger.warning(f"Error applying stability checks: {e}")
            return decision
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns"""
        
        try:
            if not returns:
                return 0.0
            
            cumulative = np.cumprod(1 + np.array(returns))
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            
            return abs(np.min(drawdown))
            
        except Exception as e:
            logger.warning(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _create_fallback_decision(self, regime: RegimeType, asset_type: AssetType) -> SelectionDecision:
        """Create fallback decision when selection fails"""
        
        try:
            # Default to first available strategy
            fallback_strategy = list(self.strategies.keys())[0] if self.strategies else "BullMarketStrategy"
            
            allocation = StrategyAllocation(
                strategy_name=fallback_strategy,
                weight=1.0,
                confidence=50.0,
                expected_return=0.0,
                expected_risk=0.2,
                allocation_reason="Fallback selection"
            )
            
            return SelectionDecision(
                selected_strategies=[fallback_strategy],
                selection_method=SelectionMethod.BEST_PERFORMER,
                allocations=[allocation],
                decision_confidence=50.0,
                expected_portfolio_return=0.0,
                expected_portfolio_risk=0.2,
                market_regime=regime,
                asset_type=asset_type,
                selection_reasoning="Fallback selection due to error"
            )
            
        except Exception as e:
            logger.error(f"Error creating fallback decision: {e}")
            # Return minimal fallback
            return SelectionDecision(
                selected_strategies=["BullMarketStrategy"],
                selection_method=SelectionMethod.BEST_PERFORMER,
                allocations=[],
                decision_confidence=0.0,
                expected_portfolio_return=0.0,
                expected_portfolio_risk=0.0,
                market_regime=regime,
                asset_type=asset_type
            )
    
    def update_strategy_performance(self, 
                                  strategy_name: str,
                                  return_pct: float,
                                  timestamp: datetime = None,
                                  additional_metrics: Dict = None):
        """Update strategy performance with new trade result"""
        
        try:
            timestamp = timestamp or datetime.now()
            
            performance_record = {
                'timestamp': timestamp,
                'return': return_pct,
                'strategy': strategy_name
            }
            
            if additional_metrics:
                performance_record.update(additional_metrics)
            
            self.performance_history[strategy_name].append(performance_record)
            
            # Limit history size
            max_records = 1000
            if len(self.performance_history[strategy_name]) > max_records:
                self.performance_history[strategy_name] = self.performance_history[strategy_name][-max_records:]
            
            logger.debug(f"Updated performance for {strategy_name}: {return_pct:.3f}%")
            
        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")
    
    def get_strategy_rankings(self) -> List[Tuple[str, StrategyScore]]:
        """Get current strategy rankings"""
        
        try:
            return sorted(
                self.strategy_scores.items(),
                key=lambda x: x[1].composite_score,
                reverse=True
            )
            
        except Exception as e:
            logger.error(f"Error getting strategy rankings: {e}")
            return []
    
    def get_selection_history(self, lookback_days: int = 30) -> List[SelectionDecision]:
        """Get recent selection history"""
        
        try:
            cutoff_time = datetime.now() - timedelta(days=lookback_days)
            
            return [
                decision for decision in self.allocation_history
                if decision.decision_time >= cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error getting selection history: {e}")
            return []

# ==================== TESTING ====================

def test_strategy_selector():
    """Test strategy selector functionality"""
    
    print("🎯 Testing Intelligent Strategy Selector")
    print("=" * 50)
    
    # Create sample market data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=50, freq='1H')
    
    base_price = 100
    prices = []
    
    for i in range(50):
        change = np.random.randn() * 0.02
        new_price = (prices[-1] if prices else base_price) * (1 + change)
        prices.append(new_price)
    
    sample_data = pd.DataFrame({
        'open': prices,
        'close': [p * (1 + np.random.randn() * 0.005) for p in prices],
        'volume': np.random.randint(1000, 10000, 50)
    }, index=dates)
    
    sample_data['high'] = sample_data[['open', 'close']].max(axis=1) * (1 + np.random.rand(50) * 0.01)
    sample_data['low'] = sample_data[['open', 'close']].min(axis=1) * (1 - np.random.rand(50) * 0.01)
    
    # Create strategy selector
    selector = IntelligentStrategySelector()
    
    print(f"✅ Strategy selector initialized:")
    print(f"   Available strategies: {len(selector.strategies)}")
    print(f"   Selection method: {selector.config['selection_method'].value}")
    print(f"   Weighting scheme: {selector.config['weighting_scheme'].value}")
    
    # Add some mock performance data
    strategies = list(selector.strategies.keys())
    for i, strategy in enumerate(strategies):
        for j in range(20):  # 20 mock trades
            return_pct = np.random.randn() * 0.02 + (i * 0.005)  # Different performance per strategy
            timestamp = datetime.now() - timedelta(days=j)
            selector.update_strategy_performance(strategy, return_pct, timestamp)
    
    print(f"\n📊 Added mock performance data for testing")
    
    # Test different selection methods
    test_regimes = [RegimeType.BULL, RegimeType.BEAR, RegimeType.VOLATILE, RegimeType.BREAKOUT]
    
    for regime in test_regimes:
        print(f"\n🎯 Testing {regime.value} regime:")
        
        decision = selector.select_strategy(
            market_data=sample_data,
            symbol='TEST',
            regime=regime,
            asset_type=AssetType.STOCK
        )
        
        print(f"   Selected strategies: {decision.selected_strategies}")
        print(f"   Selection method: {decision.selection_method.value}")
        print(f"   Decision confidence: {decision.decision_confidence:.1f}%")
        print(f"   Expected return: {decision.expected_portfolio_return:.3f}")
        print(f"   Expected risk: {decision.expected_portfolio_risk:.3f}")
        print(f"   Reasoning: {decision.selection_reasoning}")
        
        print(f"   Allocations:")
        for alloc in decision.allocations:
            print(f"      {alloc.strategy_name}: {alloc.weight:.2f} ({alloc.allocation_reason})")
    
    # Test strategy rankings
    print(f"\n🏆 Strategy Rankings:")
    rankings = selector.get_strategy_rankings()
    
    for i, (strategy_name, score) in enumerate(rankings):
        print(f"   {i+1}. {strategy_name}")
        print(f"      Composite Score: {score.composite_score:.1f}")
        print(f"      Suitability: {score.suitability_score:.1f}")
        print(f"      Performance: {score.performance_score:.1f}")
        print(f"      Risk Score: {score.risk_score:.1f}")
        print(f"      Confidence: {score.confidence_score:.1f}")
    
    # Test ensemble selection
    print(f"\n🎛️ Testing ensemble selection:")
    original_method = selector.config['selection_method']
    selector.config['selection_method'] = SelectionMethod.ENSEMBLE
    
    ensemble_decision = selector.select_strategy(
        market_data=sample_data,
        symbol='TEST',
        regime=RegimeType.VOLATILE,
        asset_type=AssetType.STOCK
    )
    
    print(f"   Ensemble strategies: {len(ensemble_decision.selected_strategies)}")
    print(f"   Portfolio return: {ensemble_decision.expected_portfolio_return:.3f}")
    print(f"   Portfolio risk: {ensemble_decision.expected_portfolio_risk:.3f}")
    
    print(f"   Detailed allocations:")
    for alloc in ensemble_decision.allocations:
        print(f"      {alloc.strategy_name}: {alloc.weight:.1%} weight, {alloc.confidence:.1f}% confidence")
    
    # Restore original method
    selector.config['selection_method'] = original_method
    
    # Test selection history
    print(f"\n📜 Selection history:")
    history = selector.get_selection_history(lookback_days=1)
    print(f"   Total decisions: {len(history)}")
    
    for decision in history[-3:]:  # Show last 3
        print(f"   {decision.decision_time.strftime('%H:%M:%S')}: {decision.selected_strategies} "
              f"({decision.selection_method.value})")
    
    print("\n🎉 Strategy selector tests completed!")

if __name__ == "__main__":
    test_strategy_selector()