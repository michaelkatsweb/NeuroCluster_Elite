#!/usr/bin/env python3
"""
File: risk_manager.py
Path: NeuroCluster-Elite/src/trading/risk_manager.py
Description: Advanced risk management system for NeuroCluster Elite

This module implements comprehensive risk management including position sizing,
portfolio risk monitoring, dynamic stop losses, and Kelly Criterion optimization.

Features:
- Kelly Criterion position sizing
- Dynamic stop loss and take profit levels
- Portfolio heat mapping and correlation analysis
- Real-time risk metrics monitoring
- Drawdown protection and circuit breakers
- Multi-asset risk allocation
- Volatility-adjusted position sizing
- Risk-adjusted performance metrics

Author: Your Name
Created: 2025-06-28
Version: 1.0.0
License: MIT
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading
import math
from collections import defaultdict, deque
import warnings

# Import our modules
try:
    from src.core.neurocluster_elite import RegimeType, AssetType, MarketData
    from src.trading.strategies.base_strategy import TradingSignal, SignalType
    from src.utils.logger import get_enhanced_logger, LogCategory
    from src.utils.helpers import calculate_sharpe_ratio, calculate_max_drawdown, format_currency, format_percentage
except ImportError:
    # Fallback for testing
    pass

# Configure logging
logger = get_enhanced_logger(__name__, LogCategory.RISK)

# ==================== ENUMS AND DATA STRUCTURES ====================

class RiskLevel(Enum):
    """Risk level categories"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"

class RiskEventType(Enum):
    """Types of risk events"""
    POSITION_LIMIT_BREACH = "position_limit_breach"
    PORTFOLIO_LIMIT_BREACH = "portfolio_limit_breach"
    DRAWDOWN_LIMIT_BREACH = "drawdown_limit_breach"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_RISK = "correlation_risk"
    CONCENTRATION_RISK = "concentration_risk"
    MARGIN_CALL = "margin_call"
    CIRCUIT_BREAKER = "circuit_breaker"

@dataclass
class RiskMetrics:
    """Risk metrics container"""
    portfolio_value: float
    total_exposure: float
    cash_balance: float
    leverage: float
    portfolio_beta: float
    portfolio_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk (95% confidence)
    var_99: float  # Value at Risk (99% confidence)
    expected_shortfall: float
    risk_adjusted_return: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PositionRisk:
    """Position-level risk metrics"""
    symbol: str
    asset_type: AssetType
    position_size: float
    market_value: float
    unrealized_pnl: float
    position_weight: float  # % of portfolio
    asset_beta: float
    asset_volatility: float
    correlation_to_portfolio: float
    var_contribution: float
    stop_loss_level: Optional[float] = None
    take_profit_level: Optional[float] = None
    risk_score: float = 0.0

@dataclass
class RiskEvent:
    """Risk event notification"""
    event_type: RiskEventType
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    message: str
    symbol: Optional[str] = None
    current_value: Optional[float] = None
    limit_value: Optional[float] = None
    recommended_action: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

# ==================== KELLY CRITERION CALCULATOR ====================

class KellyCriterionCalculator:
    """
    Kelly Criterion position sizing calculator
    
    Calculates optimal position sizes based on historical win rate,
    average win/loss ratios, and account for transaction costs.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.min_trades_for_kelly = self.config.get('min_trades_for_kelly', 20)
        self.max_kelly_fraction = self.config.get('max_kelly_fraction', 0.25)
        self.transaction_cost_pct = self.config.get('transaction_cost_pct', 0.001)
        
        # Historical performance tracking
        self.strategy_performance: Dict[str, Dict] = defaultdict(lambda: {
            'wins': 0,
            'losses': 0, 
            'total_win_amount': 0.0,
            'total_loss_amount': 0.0,
            'trades': []
        })
    
    def update_performance(self, strategy_name: str, trade_pnl: float, 
                          win_amount: float = None, loss_amount: float = None):
        """Update strategy performance for Kelly calculation"""
        
        perf = self.strategy_performance[strategy_name]
        
        if trade_pnl > 0:
            perf['wins'] += 1
            perf['total_win_amount'] += win_amount or trade_pnl
        else:
            perf['losses'] += 1
            perf['total_loss_amount'] += abs(loss_amount or trade_pnl)
        
        # Keep rolling window of recent trades
        perf['trades'].append({
            'pnl': trade_pnl,
            'timestamp': datetime.now()
        })
        
        # Keep only recent trades (last 100)
        if len(perf['trades']) > 100:
            perf['trades'] = perf['trades'][-100:]
    
    def calculate_kelly_fraction(self, strategy_name: str, 
                                 confidence_adjustment: float = 1.0) -> float:
        """
        Calculate Kelly fraction for position sizing
        
        Formula: f* = (bp - q) / b
        Where:
        - f* = fraction of capital to wager
        - b = odds received on the wager (reward/risk ratio)
        - p = probability of winning
        - q = probability of losing (1 - p)
        
        Args:
            strategy_name: Name of strategy
            confidence_adjustment: Adjustment factor based on regime confidence
            
        Returns:
            Kelly fraction (0.0 to max_kelly_fraction)
        """
        
        perf = self.strategy_performance[strategy_name]
        total_trades = perf['wins'] + perf['losses']
        
        # Need minimum number of trades for reliable Kelly calculation
        if total_trades < self.min_trades_for_kelly:
            logger.warning(f"Insufficient trades ({total_trades}) for Kelly calculation, using conservative sizing")
            return 0.02  # Conservative 2% sizing
        
        # Calculate win probability
        win_probability = perf['wins'] / total_trades
        loss_probability = perf['losses'] / total_trades
        
        # Calculate average win and loss amounts
        avg_win = perf['total_win_amount'] / perf['wins'] if perf['wins'] > 0 else 0
        avg_loss = perf['total_loss_amount'] / perf['losses'] if perf['losses'] > 0 else 1
        
        # Avoid division by zero
        if avg_loss == 0:
            return 0.01
        
        # Calculate odds (reward/risk ratio)
        odds = avg_win / avg_loss
        
        # Kelly fraction calculation
        kelly_fraction = (odds * win_probability - loss_probability) / odds
        
        # Apply confidence adjustment
        kelly_fraction *= confidence_adjustment
        
        # Apply transaction cost adjustment
        kelly_fraction *= (1 - self.transaction_cost_pct)
        
        # Ensure positive and within limits
        kelly_fraction = max(0.0, min(kelly_fraction, self.max_kelly_fraction))
        
        logger.debug(f"Kelly calculation for {strategy_name}: "
                    f"Win rate: {win_probability:.2%}, "
                    f"Avg win/loss: {odds:.2f}, "
                    f"Kelly fraction: {kelly_fraction:.3f}")
        
        return kelly_fraction

# ==================== RISK MANAGER ====================

class RiskManager:
    """
    Advanced risk management system
    
    Features:
    - Position sizing with Kelly Criterion
    - Portfolio risk monitoring
    - Dynamic stop losses
    - Correlation and concentration risk analysis
    - Real-time risk metrics calculation
    """
    
    def __init__(self, config: Dict = None):
        """Initialize risk manager"""
        
        self.config = config or self._default_config()
        
        # Risk parameters
        self.risk_level = RiskLevel(self.config.get('risk_level', 'moderate'))
        self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.02)
        self.max_position_size = self.config.get('max_position_size', 0.10)
        self.max_correlation = self.config.get('max_correlation', 0.7)
        self.max_sector_concentration = self.config.get('max_sector_concentration', 0.3)
        self.max_drawdown_limit = self.config.get('max_drawdown_limit', 0.15)
        
        # Kelly Criterion calculator
        self.kelly_calculator = KellyCriterionCalculator(self.config.get('kelly_config', {}))
        
        # Risk tracking
        self.positions: Dict[str, PositionRisk] = {}
        self.risk_events: List[RiskEvent] = []
        self.portfolio_metrics_history: List[RiskMetrics] = []
        
        # State management
        self.risk_lock = threading.RLock()
        self.circuit_breaker_active = False
        self.last_risk_calculation = None
        
        logger.info(f"🛡️ Risk Manager initialized with {self.risk_level.value} risk level")
    
    def _default_config(self) -> Dict:
        """Default risk management configuration"""
        return {
            'risk_level': 'moderate',
            'max_portfolio_risk': 0.02,     # 2% max portfolio risk per trade
            'max_position_size': 0.10,       # 10% max position size
            'max_correlation': 0.7,          # Max correlation between positions
            'max_sector_concentration': 0.3, # Max 30% in any sector
            'max_drawdown_limit': 0.15,      # 15% max drawdown before circuit breaker
            'var_confidence_levels': [0.95, 0.99],
            'lookback_days': 252,            # 1 year of trading days
            'rebalance_threshold': 0.05,     # 5% drift before rebalancing
            'kelly_config': {
                'max_kelly_fraction': 0.25,
                'min_trades_for_kelly': 20,
                'transaction_cost_pct': 0.001
            }
        }
    
    def calculate_position_size(self, signal: TradingSignal, portfolio_value: float,
                               current_volatility: float = None) -> Tuple[float, Dict]:
        """
        Calculate optimal position size using Kelly Criterion and risk constraints
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            current_volatility: Current asset volatility
            
        Returns:
            Tuple of (position_size, risk_analysis)
        """
        
        with self.risk_lock:
            # Get Kelly fraction
            kelly_fraction = self.kelly_calculator.calculate_kelly_fraction(
                signal.strategy_name,
                confidence_adjustment=signal.confidence / 100.0
            )
            
            # Calculate base position size from Kelly
            kelly_position_value = kelly_fraction * portfolio_value
            
            # Apply volatility adjustment
            if current_volatility:
                # Reduce position size for high volatility assets
                volatility_adjustment = min(1.0, 20.0 / max(current_volatility, 5.0))
                kelly_position_value *= volatility_adjustment
            
            # Apply maximum position size constraint
            max_position_value = self.max_position_size * portfolio_value
            
            # Apply portfolio risk constraint
            risk_adjusted_value = self._calculate_risk_adjusted_size(
                signal, portfolio_value, kelly_position_value
            )
            
            # Take the most conservative sizing
            final_position_value = min(
                kelly_position_value,
                max_position_value,
                risk_adjusted_value
            )
            
            # Convert to shares/units
            position_size = final_position_value / signal.entry_price
            
            # Risk analysis
            risk_analysis = {
                'kelly_fraction': kelly_fraction,
                'kelly_position_value': kelly_position_value,
                'max_position_constraint': max_position_value,
                'risk_adjusted_value': risk_adjusted_value,
                'final_position_value': final_position_value,
                'position_size': position_size,
                'position_weight': final_position_value / portfolio_value,
                'volatility_adjustment': volatility_adjustment if current_volatility else 1.0
            }
            
            logger.info(f"Position sizing for {signal.symbol}: "
                       f"Kelly: {format_currency(kelly_position_value)}, "
                       f"Final: {format_currency(final_position_value)} "
                       f"({risk_analysis['position_weight']:.1%} of portfolio)")
            
            return position_size, risk_analysis
    
    def _calculate_risk_adjusted_size(self, signal: TradingSignal, 
                                    portfolio_value: float,
                                    base_position_value: float) -> float:
        """Calculate risk-adjusted position size"""
        
        # Estimate potential loss (stop loss distance)
        if signal.stop_loss:
            potential_loss_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
        else:
            # Use default 5% stop loss if not specified
            potential_loss_pct = 0.05
        
        # Calculate position size based on max portfolio risk
        max_loss_amount = self.max_portfolio_risk * portfolio_value
        risk_adjusted_value = max_loss_amount / potential_loss_pct
        
        return min(base_position_value, risk_adjusted_value)
    
    def update_position_risk(self, symbol: str, position_data: Dict):
        """Update position risk metrics"""
        
        with self.risk_lock:
            position_risk = PositionRisk(
                symbol=symbol,
                asset_type=position_data.get('asset_type', AssetType.STOCK),
                position_size=position_data['position_size'],
                market_value=position_data['market_value'],
                unrealized_pnl=position_data['unrealized_pnl'],
                position_weight=position_data['position_weight'],
                asset_beta=position_data.get('beta', 1.0),
                asset_volatility=position_data.get('volatility', 20.0),
                correlation_to_portfolio=position_data.get('correlation', 0.5),
                var_contribution=position_data.get('var_contribution', 0.0),
                stop_loss_level=position_data.get('stop_loss'),
                take_profit_level=position_data.get('take_profit')
            )
            
            # Calculate risk score
            position_risk.risk_score = self._calculate_position_risk_score(position_risk)
            
            self.positions[symbol] = position_risk
    
    def _calculate_position_risk_score(self, position: PositionRisk) -> float:
        """Calculate composite risk score for position"""
        
        # Weight components
        size_risk = min(position.position_weight / self.max_position_size, 1.0) * 0.3
        volatility_risk = min(position.asset_volatility / 50.0, 1.0) * 0.3
        correlation_risk = min(position.correlation_to_portfolio / self.max_correlation, 1.0) * 0.2
        concentration_risk = min(position.position_weight / self.max_sector_concentration, 1.0) * 0.2
        
        risk_score = size_risk + volatility_risk + correlation_risk + concentration_risk
        return min(risk_score, 1.0)  # Cap at 1.0
    
    def calculate_portfolio_metrics(self, portfolio_data: Dict) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        
        portfolio_value = portfolio_data['total_value']
        positions = portfolio_data['positions']
        returns_history = portfolio_data.get('returns_history', [])
        
        # Basic metrics
        total_exposure = sum(abs(pos['market_value']) for pos in positions.values())
        cash_balance = portfolio_data.get('cash_balance', 0)
        leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0
        
        # Calculate portfolio beta and volatility
        portfolio_beta = self._calculate_portfolio_beta(positions, portfolio_value)
        portfolio_volatility = self._calculate_portfolio_volatility(returns_history)
        
        # Performance metrics
        sharpe_ratio = calculate_sharpe_ratio(returns_history) if len(returns_history) >= 30 else 0
        max_drawdown = calculate_max_drawdown(returns_history) if len(returns_history) >= 10 else 0
        
        # Value at Risk calculations
        var_95, var_99 = self._calculate_var(returns_history, portfolio_value)
        expected_shortfall = self._calculate_expected_shortfall(returns_history, portfolio_value)
        
        # Risk-adjusted return
        risk_adjusted_return = sharpe_ratio * np.sqrt(252) if sharpe_ratio > 0 else 0
        
        metrics = RiskMetrics(
            portfolio_value=portfolio_value,
            total_exposure=total_exposure,
            cash_balance=cash_balance,
            leverage=leverage,
            portfolio_beta=portfolio_beta,
            portfolio_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            risk_adjusted_return=risk_adjusted_return
        )
        
        # Store metrics history
        self.portfolio_metrics_history.append(metrics)
        if len(self.portfolio_metrics_history) > 1000:  # Keep last 1000 calculations
            self.portfolio_metrics_history = self.portfolio_metrics_history[-1000:]
        
        # Check for risk events
        self._check_risk_limits(metrics)
        
        return metrics
    
    def _calculate_portfolio_beta(self, positions: Dict, portfolio_value: float) -> float:
        """Calculate portfolio beta"""
        
        if portfolio_value <= 0:
            return 1.0
        
        weighted_beta = 0.0
        for symbol, position in positions.items():
            weight = abs(position['market_value']) / portfolio_value
            beta = position.get('beta', 1.0)
            weighted_beta += weight * beta
        
        return weighted_beta
    
    def _calculate_portfolio_volatility(self, returns_history: List[float]) -> float:
        """Calculate portfolio volatility (annualized)"""
        
        if len(returns_history) < 10:
            return 0.2  # Default 20% volatility
        
        returns_array = np.array(returns_history[-252:])  # Last year
        daily_volatility = np.std(returns_array)
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        return annualized_volatility
    
    def _calculate_var(self, returns_history: List[float], 
                      portfolio_value: float) -> Tuple[float, float]:
        """Calculate Value at Risk at 95% and 99% confidence levels"""
        
        if len(returns_history) < 30:
            return 0.0, 0.0
        
        returns_array = np.array(returns_history[-252:])  # Last year
        
        # Calculate percentiles
        var_95_pct = np.percentile(returns_array, 5)   # 5th percentile
        var_99_pct = np.percentile(returns_array, 1)   # 1st percentile
        
        # Convert to dollar amounts
        var_95 = abs(var_95_pct * portfolio_value)
        var_99 = abs(var_99_pct * portfolio_value)
        
        return var_95, var_99
    
    def _calculate_expected_shortfall(self, returns_history: List[float],
                                    portfolio_value: float) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        
        if len(returns_history) < 30:
            return 0.0
        
        returns_array = np.array(returns_history[-252:])
        var_95_pct = np.percentile(returns_array, 5)
        
        # Expected Shortfall is the average of returns below VaR
        tail_returns = returns_array[returns_array <= var_95_pct]
        if len(tail_returns) > 0:
            expected_shortfall_pct = np.mean(tail_returns)
            return abs(expected_shortfall_pct * portfolio_value)
        
        return 0.0
    
    def _check_risk_limits(self, metrics: RiskMetrics):
        """Check portfolio against risk limits and generate alerts"""
        
        # Maximum drawdown check
        if metrics.max_drawdown > self.max_drawdown_limit:
            self._create_risk_event(
                RiskEventType.DRAWDOWN_LIMIT_BREACH,
                "HIGH",
                f"Portfolio drawdown ({metrics.max_drawdown:.1%}) exceeds limit ({self.max_drawdown_limit:.1%})",
                current_value=metrics.max_drawdown,
                limit_value=self.max_drawdown_limit,
                recommended_action="Consider reducing position sizes or stopping trading"
            )
        
        # Leverage check
        max_leverage = 2.0 if self.risk_level == RiskLevel.AGGRESSIVE else 1.5
        if metrics.leverage > max_leverage:
            self._create_risk_event(
                RiskEventType.PORTFOLIO_LIMIT_BREACH,
                "MEDIUM",
                f"Portfolio leverage ({metrics.leverage:.1f}x) exceeds limit ({max_leverage:.1f}x)",
                current_value=metrics.leverage,
                limit_value=max_leverage,
                recommended_action="Reduce position sizes to lower leverage"
            )
        
        # Portfolio concentration check
        self._check_concentration_risk()
    
    def _check_concentration_risk(self):
        """Check for concentration risk in portfolio"""
        
        if not self.positions:
            return
        
        total_portfolio_value = sum(pos.market_value for pos in self.positions.values())
        
        for symbol, position in self.positions.items():
            if position.position_weight > self.max_position_size:
                self._create_risk_event(
                    RiskEventType.CONCENTRATION_RISK,
                    "MEDIUM",
                    f"Position {symbol} weight ({position.position_weight:.1%}) exceeds limit",
                    symbol=symbol,
                    current_value=position.position_weight,
                    limit_value=self.max_position_size,
                    recommended_action=f"Consider reducing {symbol} position size"
                )
    
    def _create_risk_event(self, event_type: RiskEventType, severity: str,
                          message: str, symbol: str = None, current_value: float = None,
                          limit_value: float = None, recommended_action: str = None):
        """Create and log risk event"""
        
        event = RiskEvent(
            event_type=event_type,
            severity=severity,
            message=message,
            symbol=symbol,
            current_value=current_value,
            limit_value=limit_value,
            recommended_action=recommended_action
        )
        
        self.risk_events.append(event)
        
        # Keep only recent events
        if len(self.risk_events) > 1000:
            self.risk_events = self.risk_events[-1000:]
        
        # Log event
        log_level = {
            'LOW': logger.info,
            'MEDIUM': logger.warning,
            'HIGH': logger.error,
            'CRITICAL': logger.critical
        }.get(severity, logger.info)
        
        log_level(f"🚨 Risk Event: {message}")
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary"""
        
        if not self.portfolio_metrics_history:
            return {'status': 'No portfolio data available'}
        
        latest_metrics = self.portfolio_metrics_history[-1]
        recent_events = [e for e in self.risk_events[-10:] if e.severity in ['HIGH', 'CRITICAL']]
        
        return {
            'portfolio_metrics': {
                'value': format_currency(latest_metrics.portfolio_value),
                'leverage': f"{latest_metrics.leverage:.2f}x",
                'volatility': f"{latest_metrics.portfolio_volatility:.1%}",
                'sharpe_ratio': f"{latest_metrics.sharpe_ratio:.2f}",
                'max_drawdown': f"{latest_metrics.max_drawdown:.1%}",
                'var_95': format_currency(latest_metrics.var_95),
                'beta': f"{latest_metrics.portfolio_beta:.2f}"
            },
            'risk_status': {
                'circuit_breaker_active': self.circuit_breaker_active,
                'high_risk_positions': len([p for p in self.positions.values() if p.risk_score > 0.7]),
                'recent_critical_events': len(recent_events),
                'overall_risk_level': self._assess_overall_risk_level()
            },
            'recent_events': [
                {
                    'type': e.event_type.value,
                    'severity': e.severity,
                    'message': e.message,
                    'timestamp': e.timestamp.isoformat()
                }
                for e in recent_events
            ]
        }
    
    def _assess_overall_risk_level(self) -> str:
        """Assess overall portfolio risk level"""
        
        if not self.portfolio_metrics_history:
            return "UNKNOWN"
        
        latest = self.portfolio_metrics_history[-1]
        risk_score = 0
        
        # Drawdown risk
        if latest.max_drawdown > 0.10:
            risk_score += 3
        elif latest.max_drawdown > 0.05:
            risk_score += 1
        
        # Leverage risk
        if latest.leverage > 2.0:
            risk_score += 3
        elif latest.leverage > 1.5:
            risk_score += 1
        
        # Volatility risk
        if latest.portfolio_volatility > 0.30:
            risk_score += 2
        elif latest.portfolio_volatility > 0.20:
            risk_score += 1
        
        # Concentration risk
        high_risk_positions = len([p for p in self.positions.values() if p.risk_score > 0.7])
        if high_risk_positions > 3:
            risk_score += 2
        elif high_risk_positions > 1:
            risk_score += 1
        
        # Map to risk levels
        if risk_score >= 6:
            return "CRITICAL"
        elif risk_score >= 4:
            return "HIGH"
        elif risk_score >= 2:
            return "MEDIUM"
        else:
            return "LOW"

# ==================== TESTING ====================

def test_risk_manager():
    """Test risk manager functionality"""
    
    print("🛡️ Testing Risk Manager")
    print("=" * 40)
    
    # Create risk manager
    config = {
        'risk_level': 'moderate',
        'max_portfolio_risk': 0.02,
        'max_position_size': 0.10
    }
    
    risk_manager = RiskManager(config)
    
    # Create mock trading signal
    from src.trading.strategies.base_strategy import TradingSignal
    from src.core.neurocluster_elite import RegimeType, AssetType
    
    signal = TradingSignal(
        symbol='AAPL',
        asset_type=AssetType.STOCK,
        signal_type=SignalType.BUY,
        regime=RegimeType.BULL,
        confidence=85.0,
        entry_price=150.0,
        current_price=150.0,
        strategy_name='BullMomentumStrategy',
        reasoning='Strong momentum signal'
    )
    
    # Test position sizing
    portfolio_value = 100000
    position_size, risk_analysis = risk_manager.calculate_position_size(
        signal, portfolio_value, current_volatility=20.0
    )
    
    print(f"✅ Position sizing calculation:")
    print(f"   Portfolio value: {format_currency(portfolio_value)}")
    print(f"   Recommended position size: {position_size:.0f} shares")
    print(f"   Position value: {format_currency(position_size * signal.entry_price)}")
    print(f"   Portfolio weight: {risk_analysis['position_weight']:.1%}")
    print(f"   Kelly fraction: {risk_analysis['kelly_fraction']:.3f}")
    
    # Test portfolio metrics
    portfolio_data = {
        'total_value': portfolio_value,
        'cash_balance': 20000,
        'positions': {
            'AAPL': {
                'market_value': 15000,
                'position_size': 100,
                'beta': 1.2,
                'volatility': 25.0
            },
            'GOOGL': {
                'market_value': 12000,
                'position_size': 50,
                'beta': 1.1,
                'volatility': 30.0
            }
        },
        'returns_history': list(np.random.normal(0.001, 0.02, 100))  # Mock returns
    }
    
    metrics = risk_manager.calculate_portfolio_metrics(portfolio_data)
    
    print(f"\n✅ Portfolio risk metrics:")
    print(f"   Leverage: {metrics.leverage:.2f}x")
    print(f"   Portfolio beta: {metrics.portfolio_beta:.2f}")
    print(f"   Volatility: {metrics.portfolio_volatility:.1%}")
    print(f"   Max drawdown: {metrics.max_drawdown:.1%}")
    print(f"   VaR (95%): {format_currency(metrics.var_95)}")
    print(f"   Sharpe ratio: {metrics.sharpe_ratio:.2f}")
    
    # Test risk summary
    summary = risk_manager.get_risk_summary()
    print(f"\n✅ Overall risk level: {summary['risk_status']['overall_risk_level']}")
    
    print("\n🎉 Risk manager tests completed!")

if __name__ == "__main__":
    test_risk_manager()