#!/usr/bin/env python3
"""
File: trading_engine.py
Path: NeuroCluster-Elite/src/trading/trading_engine.py
Description: Advanced trading engine with multi-asset support and AI-powered strategies

This module implements the core trading engine that combines the proven NeuroCluster
algorithm with advanced trading strategies, risk management, and multi-asset support.

Features:
- AI-powered strategy selection based on market regimes
- Advanced risk management with Kelly Criterion position sizing
- Multi-asset support (stocks, crypto, forex, commodities)
- Real-time portfolio tracking and P&L calculation
- Paper trading and live trading modes
- Performance analytics and backtesting
- Order management and execution simulation
- Dynamic strategy adaptation

Author: Your Name
Created: 2025-06-28
Version: 1.0.0
License: MIT
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import sqlite3
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid
import math
from collections import defaultdict, deque

# Import our modules
try:
    from src.core.neurocluster_elite import NeuroClusterElite, RegimeType, AssetType, MarketData
    from src.data.multi_asset_manager import MultiAssetDataManager
    from src.trading.strategies.base_strategy import BaseStrategy, TradingSignal, SignalType
    from src.trading.strategies.strategy_factory import StrategyFactory
    from src.trading.risk_manager import RiskManager
    from src.trading.portfolio_manager import PortfolioManager
    from src.trading.order_manager import OrderManager
    from src.utils.config_manager import ConfigManager
    from src.utils.helpers import calculate_sharpe_ratio, calculate_max_drawdown, format_currency
except ImportError:
    # Fallback for testing
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== ENUMS AND DATA STRUCTURES ====================

class OrderType(Enum):
    """Order types for execution"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP = "TRAILING_STOP"

class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class PositionSide(Enum):
    """Position direction"""
    LONG = "LONG"
    SHORT = "SHORT"

class TradingMode(Enum):
    """Trading execution modes"""
    PAPER = "PAPER"
    LIVE = "LIVE"
    BACKTEST = "BACKTEST"

@dataclass
class Position:
    """Trading position structure"""
    id: str
    symbol: str
    asset_type: AssetType
    side: PositionSide
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    opened_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    strategy_name: str = ""
    regime_at_entry: Optional[RegimeType] = None
    confidence_at_entry: float = 0.0
    
    def update_market_value(self, current_price: float):
        """Update position with current market price"""
        self.current_price = current_price
        self.market_value = self.quantity * current_price
        
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
        
        if self.entry_price > 0:
            self.unrealized_pnl_pct = (self.unrealized_pnl / (self.entry_price * self.quantity)) * 100
        
        self.updated_at = datetime.now()

@dataclass
class Trade:
    """Completed trade record"""
    id: str
    symbol: str
    asset_type: AssetType
    side: PositionSide
    signal_type: SignalType
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None
    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0
    fees: float = 0.0
    strategy_name: str = ""
    regime_at_entry: Optional[RegimeType] = None
    regime_at_exit: Optional[RegimeType] = None
    confidence_at_entry: float = 0.0
    confidence_at_exit: float = 0.0
    exit_reason: str = ""
    duration_minutes: int = 0
    
    def close_trade(self, exit_price: float, exit_reason: str = "Manual", 
                   regime: RegimeType = None, confidence: float = 0.0):
        """Close the trade and calculate P&L"""
        self.exit_price = exit_price
        self.exit_time = datetime.now()
        self.exit_reason = exit_reason
        self.regime_at_exit = regime
        self.confidence_at_exit = confidence
        
        # Calculate duration
        self.duration_minutes = int((self.exit_time - self.entry_time).total_seconds() / 60)
        
        # Calculate realized P&L
        if self.side == PositionSide.LONG:
            self.realized_pnl = (exit_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.realized_pnl = (self.entry_price - exit_price) * self.quantity
        
        # Subtract fees
        self.realized_pnl -= self.fees
        
        # Calculate percentage return
        if self.entry_price > 0:
            investment = self.entry_price * self.quantity
            self.realized_pnl_pct = (self.realized_pnl / investment) * 100

@dataclass
class PortfolioState:
    """Portfolio state and metrics"""
    timestamp: datetime
    total_value: float
    cash_balance: float
    invested_value: float
    available_buying_power: float
    
    # Performance metrics
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    
    # Risk metrics
    portfolio_beta: float = 0.0
    var_95: float = 0.0  # Value at Risk (95% confidence)
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Position summary
    num_positions: int = 0
    long_positions: int = 0
    short_positions: int = 0
    largest_position_pct: float = 0.0
    
    # Asset allocation
    asset_allocation: Dict[AssetType, float] = field(default_factory=dict)
    sector_allocation: Dict[str, float] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Trading performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # days
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    calmar_ratio: float = 0.0
    
    def update_metrics(self, trades: List[Trade], portfolio_values: List[float]):
        """Update performance metrics from trade history"""
        
        if not trades:
            return
        
        # Basic trade statistics
        self.total_trades = len(trades)
        winning_trades = [t for t in trades if t.realized_pnl > 0]
        losing_trades = [t for t in trades if t.realized_pnl < 0]
        
        self.winning_trades = len(winning_trades)
        self.losing_trades = len(losing_trades)
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        
        # Average win/loss
        if winning_trades:
            self.average_win = np.mean([t.realized_pnl for t in winning_trades])
        if losing_trades:
            self.average_loss = abs(np.mean([t.realized_pnl for t in losing_trades]))
        
        # Profit factor
        total_wins = sum(t.realized_pnl for t in winning_trades)
        total_losses = abs(sum(t.realized_pnl for t in losing_trades))
        self.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Portfolio-based metrics
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            self.total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
            self.volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized
            
            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            if self.volatility > 0:
                self.sharpe_ratio = (self.annualized_return - risk_free_rate) / (self.volatility / 100)
            
            # Max drawdown
            self.max_drawdown = calculate_max_drawdown(portfolio_values)

# ==================== STRATEGY SELECTION ENGINE ====================

class StrategySelector:
    """Intelligent strategy selection based on market conditions"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.strategy_factory = StrategyFactory()
        self.strategy_performance = defaultdict(lambda: {'trades': 0, 'pnl': 0.0, 'win_rate': 0.0})
        
    def select_strategy(self, regime: RegimeType, asset_type: AssetType, 
                       market_data: MarketData, confidence: float) -> str:
        """Select optimal strategy based on market conditions"""
        
        # Base strategy mapping by regime
        regime_strategies = {
            RegimeType.BULL: 'bull_momentum',
            RegimeType.BEAR: 'bear_defensive',
            RegimeType.BREAKOUT: 'breakout_momentum',
            RegimeType.BREAKDOWN: 'bear_defensive',
            RegimeType.VOLATILE: 'volatility_trading',
            RegimeType.SIDEWAYS: 'range_trading',
            RegimeType.ACCUMULATION: 'accumulation',
            RegimeType.DISTRIBUTION: 'distribution'
        }
        
        base_strategy = regime_strategies.get(regime, 'range_trading')
        
        # Asset-specific adjustments
        if asset_type == AssetType.CRYPTO:
            if market_data.volatility and market_data.volatility > 50:
                return 'crypto_volatility'
            elif regime in [RegimeType.BULL, RegimeType.BREAKOUT]:
                return 'crypto_momentum'
        elif asset_type == AssetType.FOREX:
            return 'forex_carry'
        elif asset_type == AssetType.COMMODITY:
            if regime == RegimeType.VOLATILE:
                return 'commodity_momentum'
        
        # Confidence-based adjustments
        if confidence < 0.6:
            return 'conservative'  # Low confidence strategy
        elif confidence > 0.9:
            return f'{base_strategy}_aggressive'  # High confidence variant
        
        return base_strategy
    
    def update_strategy_performance(self, strategy_name: str, trade: Trade):
        """Update strategy performance tracking"""
        
        perf = self.strategy_performance[strategy_name]
        perf['trades'] += 1
        perf['pnl'] += trade.realized_pnl
        
        # Recalculate win rate
        # This is simplified - in production, track wins/losses separately
        if trade.realized_pnl > 0:
            perf['win_rate'] = (perf['win_rate'] * (perf['trades'] - 1) + 1) / perf['trades']
        else:
            perf['win_rate'] = (perf['win_rate'] * (perf['trades'] - 1)) / perf['trades']
    
    def get_best_strategies(self, limit: int = 5) -> List[Tuple[str, Dict]]:
        """Get best performing strategies"""
        
        strategies = []
        for name, perf in self.strategy_performance.items():
            if perf['trades'] >= 10:  # Minimum sample size
                score = perf['pnl'] * perf['win_rate']  # Combined score
                strategies.append((name, perf, score))
        
        strategies.sort(key=lambda x: x[2], reverse=True)
        return [(name, perf) for name, perf, _ in strategies[:limit]]

# ==================== MAIN TRADING ENGINE ====================

class AdvancedTradingEngine:
    """
    Advanced trading engine combining NeuroCluster algorithm with intelligent trading
    
    Features:
    - Real-time regime detection using proven NCS algorithm
    - AI-powered strategy selection
    - Advanced risk management
    - Multi-asset support
    - Paper and live trading modes
    """
    
    def __init__(self, neurocluster: NeuroClusterElite, data_manager: MultiAssetDataManager, 
                 config: Dict = None):
        
        self.neurocluster = neurocluster
        self.data_manager = data_manager
        self.config = config or self._default_config()
        
        # Trading state
        self.trading_mode = TradingMode(self.config.get('trading_mode', 'PAPER'))
        self.initial_capital = self.config.get('initial_capital', 100000.0)
        self.portfolio_value = self.initial_capital
        self.cash_balance = self.initial_capital
        self.is_trading_active = False
        
        # Core components
        self.strategy_selector = StrategySelector(self.config.get('strategy_selection', {}))
        self.risk_manager = RiskManager(self.config.get('risk_management', {}))
        self.portfolio_manager = PortfolioManager(self.config.get('portfolio', {}))
        self.order_manager = OrderManager(self.config.get('order_management', {}))
        
        # Data storage
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.orders: Dict[str, Dict] = {}
        self.portfolio_history: List[PortfolioState] = []
        self.performance_metrics = PerformanceMetrics()
        
        # Threading and async
        self.engine_lock = threading.RLock()
        self.last_update = datetime.now()
        
        # Database connection
        self.db_path = self.config.get('database_path', 'data/trading_engine.db')
        self._initialize_database()
        
        logger.info(f"🚀 Advanced Trading Engine initialized - Mode: {self.trading_mode.value}")
    
    def _default_config(self) -> Dict:
        """Default trading engine configuration"""
        return {
            'trading_mode': 'PAPER',
            'initial_capital': 100000.0,
            'max_positions': 20,
            'position_sizing': 'kelly_criterion',
            'risk_management': {
                'max_position_size': 0.10,
                'max_portfolio_risk': 0.02,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.15
            },
            'fees': {
                'stock_commission': 0.0,  # Commission-free brokers
                'crypto_fee': 0.001,      # 0.1%
                'forex_spread': 0.0001    # 1 pip
            },
            'execution': {
                'slippage_pct': 0.001,    # 0.1% slippage
                'partial_fill_probability': 0.1
            }
        }
    
    def _initialize_database(self):
        """Initialize SQLite database for persistent storage"""
        
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        
        # Create tables
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS positions (
                id TEXT PRIMARY KEY,
                symbol TEXT,
                asset_type TEXT,
                side TEXT,
                quantity REAL,
                entry_price REAL,
                current_price REAL,
                market_value REAL,
                unrealized_pnl REAL,
                stop_loss REAL,
                take_profit REAL,
                opened_at TEXT,
                updated_at TEXT,
                strategy_name TEXT,
                regime_at_entry TEXT
            );
            
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                symbol TEXT,
                asset_type TEXT,
                side TEXT,
                signal_type TEXT,
                quantity REAL,
                entry_price REAL,
                exit_price REAL,
                entry_time TEXT,
                exit_time TEXT,
                realized_pnl REAL,
                realized_pnl_pct REAL,
                fees REAL,
                strategy_name TEXT,
                regime_at_entry TEXT,
                regime_at_exit TEXT,
                exit_reason TEXT,
                duration_minutes INTEGER
            );
            
            CREATE TABLE IF NOT EXISTS portfolio_history (
                timestamp TEXT,
                total_value REAL,
                cash_balance REAL,
                invested_value REAL,
                total_pnl REAL,
                num_positions INTEGER
            );
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("Trading engine database initialized")
    
    async def initialize(self):
        """Initialize trading engine components"""
        
        # Load existing positions and trades from database
        await self._load_state_from_database()
        
        # Initialize components
        await self.data_manager.initialize()
        
        logger.info("Trading engine components initialized")
    
    async def start_trading(self):
        """Start the trading engine"""
        
        self.is_trading_active = True
        logger.info("🟢 Trading engine started")
    
    def stop_trading(self):
        """Stop the trading engine"""
        
        self.is_trading_active = False
        logger.info("🔴 Trading engine stopped")
    
    async def execute_trading_cycle(self, symbols: List[str], asset_types: Dict[str, AssetType]) -> Dict:
        """Execute complete trading cycle"""
        
        if not self.is_trading_active:
            return {'status': 'inactive'}
        
        start_time = time.time()
        
        try:
            with self.engine_lock:
                # 1. Fetch market data for all assets
                all_market_data = await self._fetch_all_market_data(symbols, asset_types)
                
                if not all_market_data:
                    logger.warning("No market data received")
                    return {'status': 'no_data'}
                
                # 2. Detect market regime using NeuroCluster
                regime, confidence = self.neurocluster.detect_regime(all_market_data)
                
                # 3. Update existing positions with current prices
                await self._update_positions(all_market_data)
                
                # 4. Check exit conditions for existing positions
                exit_signals = await self._check_exit_conditions(all_market_data, regime, confidence)
                
                # 5. Execute exit trades
                exit_trades = await self._execute_exit_signals(exit_signals, all_market_data)
                
                # 6. Generate new entry signals
                entry_signals = await self._generate_entry_signals(all_market_data, regime, confidence)
                
                # 7. Apply risk management filters
                validated_signals = self.risk_manager.validate_signals(
                    entry_signals, self.portfolio_value, self.positions
                )
                
                # 8. Execute entry trades
                entry_trades = await self._execute_entry_signals(validated_signals, all_market_data)
                
                # 9. Update portfolio state
                await self._update_portfolio_state(all_market_data)
                
                # 10. Save state to database
                await self._save_state_to_database()
                
                # Update last update time
                self.last_update = datetime.now()
                
                # Log cycle completion
                cycle_time = (time.time() - start_time) * 1000
                logger.info(
                    f"✅ Trading cycle completed in {cycle_time:.2f}ms | "
                    f"Regime: {regime.value} ({confidence:.1f}%) | "
                    f"Signals: {len(entry_signals)} | "
                    f"Executed: {len(entry_trades + exit_trades)}"
                )
                
                return {
                    'status': 'success',
                    'regime': regime,
                    'confidence': confidence,
                    'signals_generated': len(entry_signals),
                    'signals_executed': len(entry_trades + exit_trades),
                    'portfolio_value': self.portfolio_value,
                    'active_positions': len(self.positions),
                    'cycle_time_ms': cycle_time
                }
                
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _fetch_all_market_data(self, symbols: List[str], 
                                   asset_types: Dict[str, AssetType]) -> Dict[str, MarketData]:
        """Fetch market data for all tracked symbols"""
        
        all_data = {}
        
        # Group symbols by asset type for efficient fetching
        grouped_symbols = defaultdict(list)
        for symbol in symbols:
            asset_type = asset_types.get(symbol, AssetType.STOCK)
            grouped_symbols[asset_type].append(symbol)
        
        # Fetch data for each asset type
        for asset_type, type_symbols in grouped_symbols.items():
            try:
                data = await self.data_manager.fetch_market_data(type_symbols, asset_type)
                all_data.update(data)
            except Exception as e:
                logger.warning(f"Error fetching {asset_type.value} data: {e}")
        
        return all_data
    
    async def _update_positions(self, market_data: Dict[str, MarketData]):
        """Update existing positions with current market prices"""
        
        for position_id, position in self.positions.items():
            if position.symbol in market_data:
                current_price = market_data[position.symbol].price
                position.update_market_value(current_price)
    
    async def _check_exit_conditions(self, market_data: Dict[str, MarketData], 
                                   regime: RegimeType, confidence: float) -> List[Dict]:
        """Check exit conditions for existing positions"""
        
        exit_signals = []
        
        for position_id, position in self.positions.items():
            if position.symbol not in market_data:
                continue
            
            market_data_item = market_data[position.symbol]
            current_price = market_data_item.price
            
            # Check stop loss
            if position.stop_loss and self._check_stop_loss(position, current_price):
                exit_signals.append({
                    'position_id': position_id,
                    'reason': 'stop_loss',
                    'price': current_price,
                    'urgency': 'high'
                })
                continue
            
            # Check take profit
            if position.take_profit and self._check_take_profit(position, current_price):
                exit_signals.append({
                    'position_id': position_id,
                    'reason': 'take_profit',
                    'price': current_price,
                    'urgency': 'medium'
                })
                continue
            
            # Check strategy-specific exit conditions
            strategy_exit = self._check_strategy_exit(position, market_data_item, regime, confidence)
            if strategy_exit:
                exit_signals.append({
                    'position_id': position_id,
                    'reason': strategy_exit['reason'],
                    'price': current_price,
                    'urgency': strategy_exit.get('urgency', 'low')
                })
        
        return exit_signals
    
    def _check_stop_loss(self, position: Position, current_price: float) -> bool:
        """Check if stop loss should be triggered"""
        
        if not position.stop_loss:
            return False
        
        if position.side == PositionSide.LONG:
            return current_price <= position.stop_loss
        else:  # SHORT
            return current_price >= position.stop_loss
    
    def _check_take_profit(self, position: Position, current_price: float) -> bool:
        """Check if take profit should be triggered"""
        
        if not position.take_profit:
            return False
        
        if position.side == PositionSide.LONG:
            return current_price >= position.take_profit
        else:  # SHORT
            return current_price <= position.take_profit
    
    def _check_strategy_exit(self, position: Position, market_data: MarketData, 
                           regime: RegimeType, confidence: float) -> Optional[Dict]:
        """Check strategy-specific exit conditions"""
        
        # Time-based exits
        position_age = datetime.now() - position.opened_at
        
        # Exit if regime has changed significantly from entry
        if position.regime_at_entry and position.regime_at_entry != regime:
            if confidence > 0.8:  # High confidence regime change
                return {'reason': 'regime_change', 'urgency': 'medium'}
        
        # Exit if position is very old (reduce portfolio turnover)
        if position_age.days > 30:
            return {'reason': 'max_hold_period', 'urgency': 'low'}
        
        # Exit if unrealized loss is too large
        if position.unrealized_pnl_pct < -10:  # 10% loss
            return {'reason': 'large_loss', 'urgency': 'high'}
        
        return None
    
    async def _execute_exit_signals(self, exit_signals: List[Dict], 
                                  market_data: Dict[str, MarketData]) -> List[Trade]:
        """Execute exit signals and close positions"""
        
        executed_trades = []
        
        # Sort by urgency (high urgency first)
        urgency_order = {'high': 0, 'medium': 1, 'low': 2}
        exit_signals.sort(key=lambda x: urgency_order.get(x['urgency'], 3))
        
        for signal in exit_signals:
            try:
                position_id = signal['position_id']
                if position_id not in self.positions:
                    continue
                
                position = self.positions[position_id]
                exit_price = signal['price']
                exit_reason = signal['reason']
                
                # Create trade record
                trade = Trade(
                    id=str(uuid.uuid4()),
                    symbol=position.symbol,
                    asset_type=position.asset_type,
                    side=position.side,
                    signal_type=SignalType.SELL if position.side == PositionSide.LONG else SignalType.BUY,
                    quantity=position.quantity,
                    entry_price=position.entry_price,
                    strategy_name=position.strategy_name,
                    regime_at_entry=position.regime_at_entry,
                    confidence_at_entry=position.confidence_at_entry
                )
                
                # Close the trade
                trade.close_trade(exit_price, exit_reason)
                
                # Calculate fees
                trade.fees = self._calculate_fees(trade)
                trade.realized_pnl -= trade.fees
                
                # Update cash balance
                if position.side == PositionSide.LONG:
                    self.cash_balance += exit_price * position.quantity - trade.fees
                else:  # SHORT
                    self.cash_balance += (position.entry_price - exit_price) * position.quantity - trade.fees
                
                # Remove position
                del self.positions[position_id]
                
                # Add to trade history
                self.trades.append(trade)
                executed_trades.append(trade)
                
                # Update strategy performance
                self.strategy_selector.update_strategy_performance(trade.strategy_name, trade)
                
                logger.info(
                    f"🔄 Closed position: {trade.symbol} | "
                    f"P&L: {format_currency(trade.realized_pnl)} ({trade.realized_pnl_pct:.2f}%) | "
                    f"Reason: {exit_reason}"
                )
                
            except Exception as e:
                logger.error(f"Error executing exit signal: {e}")
        
        return executed_trades
    
    async def _generate_entry_signals(self, market_data: Dict[str, MarketData], 
                                    regime: RegimeType, confidence: float) -> List[TradingSignal]:
        """Generate new entry signals based on market conditions"""
        
        signals = []
        
        for symbol, data in market_data.items():
            try:
                # Skip if we already have a position in this symbol
                if any(pos.symbol == symbol for pos in self.positions.values()):
                    continue
                
                # Select appropriate strategy
                strategy_name = self.strategy_selector.select_strategy(
                    regime, data.asset_type, data, confidence
                )
                
                # Get strategy instance and generate signal
                strategy = self.strategy_factory.get_strategy(strategy_name)
                if not strategy:
                    continue
                
                signal = strategy.generate_signal(data, regime, confidence)
                if signal:
                    # Add position sizing
                    position_size = self.risk_manager.calculate_position_size(
                        signal, self.portfolio_value, self.positions
                    )
                    
                    if position_size > 0:
                        signal.position_size = position_size / signal.entry_price
                        signal.position_value = position_size
                        signal.strategy_name = strategy_name
                        signals.append(signal)
                
            except Exception as e:
                logger.warning(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    async def _execute_entry_signals(self, signals: List[TradingSignal], 
                                   market_data: Dict[str, MarketData]) -> List[Trade]:
        """Execute entry signals and open new positions"""
        
        executed_trades = []
        
        for signal in signals:
            try:
                # Check if we have enough cash
                required_cash = signal.position_value + self._calculate_fees_for_signal(signal)
                if required_cash > self.cash_balance:
                    logger.warning(f"Insufficient cash for {signal.symbol}: ${required_cash:.2f} > ${self.cash_balance:.2f}")
                    continue
                
                # Execute the trade
                entry_price = self._apply_slippage(signal.entry_price)
                
                # Create position
                position = Position(
                    id=str(uuid.uuid4()),
                    symbol=signal.symbol,
                    asset_type=signal.asset_type,
                    side=PositionSide.LONG if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else PositionSide.SHORT,
                    quantity=signal.position_size,
                    entry_price=entry_price,
                    current_price=entry_price,
                    market_value=signal.position_size * entry_price,
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    strategy_name=signal.strategy_name,
                    regime_at_entry=signal.regime,
                    confidence_at_entry=signal.confidence
                )
                
                # Create trade record for entry
                trade = Trade(
                    id=str(uuid.uuid4()),
                    symbol=signal.symbol,
                    asset_type=signal.asset_type,
                    side=position.side,
                    signal_type=signal.signal_type,
                    quantity=signal.position_size,
                    entry_price=entry_price,
                    strategy_name=signal.strategy_name,
                    regime_at_entry=signal.regime,
                    confidence_at_entry=signal.confidence
                )
                
                # Calculate fees
                fees = self._calculate_fees_for_signal(signal)
                
                # Update cash balance
                if position.side == PositionSide.LONG:
                    self.cash_balance -= position.market_value + fees
                else:  # SHORT
                    self.cash_balance += position.market_value - fees
                
                # Add position
                self.positions[position.id] = position
                executed_trades.append(trade)
                
                logger.info(
                    f"📈 Opened position: {position.symbol} | "
                    f"Side: {position.side.value} | "
                    f"Size: {position.quantity:.2f} @ ${entry_price:.2f} | "
                    f"Strategy: {signal.strategy_name}"
                )
                
            except Exception as e:
                logger.error(f"Error executing entry signal for {signal.symbol}: {e}")
        
        return executed_trades
    
    def _apply_slippage(self, price: float) -> float:
        """Apply slippage to execution price"""
        
        slippage_pct = self.config.get('execution', {}).get('slippage_pct', 0.001)
        # Random slippage between 0 and max slippage
        slippage = np.random.uniform(0, slippage_pct)
        return price * (1 + slippage)
    
    def _calculate_fees(self, trade: Trade) -> float:
        """Calculate trading fees"""
        
        fees_config = self.config.get('fees', {})
        
        if trade.asset_type == AssetType.STOCK:
            return fees_config.get('stock_commission', 0.0)
        elif trade.asset_type == AssetType.CRYPTO:
            fee_rate = fees_config.get('crypto_fee', 0.001)
            return trade.quantity * trade.entry_price * fee_rate
        elif trade.asset_type == AssetType.FOREX:
            spread = fees_config.get('forex_spread', 0.0001)
            return trade.quantity * spread
        
        return 0.0
    
    def _calculate_fees_for_signal(self, signal: TradingSignal) -> float:
        """Calculate fees for a trading signal"""
        
        fees_config = self.config.get('fees', {})
        
        if signal.asset_type == AssetType.STOCK:
            return fees_config.get('stock_commission', 0.0)
        elif signal.asset_type == AssetType.CRYPTO:
            fee_rate = fees_config.get('crypto_fee', 0.001)
            return signal.position_value * fee_rate
        elif signal.asset_type == AssetType.FOREX:
            spread = fees_config.get('forex_spread', 0.0001)
            return signal.position_size * spread
        
        return 0.0
    
    async def _update_portfolio_state(self, market_data: Dict[str, MarketData]):
        """Update portfolio state and metrics"""
        
        # Calculate total portfolio value
        total_position_value = sum(pos.market_value for pos in self.positions.values())
        self.portfolio_value = self.cash_balance + total_position_value
        
        # Calculate P&L
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized_pnl = sum(trade.realized_pnl for trade in self.trades)
        total_pnl = total_unrealized_pnl + total_realized_pnl
        total_pnl_pct = (total_pnl / self.initial_capital) * 100
        
        # Create portfolio state
        portfolio_state = PortfolioState(
            timestamp=datetime.now(),
            total_value=self.portfolio_value,
            cash_balance=self.cash_balance,
            invested_value=total_position_value,
            available_buying_power=self.cash_balance,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            num_positions=len(self.positions),
            long_positions=len([p for p in self.positions.values() if p.side == PositionSide.LONG]),
            short_positions=len([p for p in self.positions.values() if p.side == PositionSide.SHORT])
        )
        
        # Calculate asset allocation
        asset_allocation = defaultdict(float)
        for position in self.positions.values():
            asset_allocation[position.asset_type] += position.market_value
        
        # Normalize to percentages
        for asset_type in asset_allocation:
            asset_allocation[asset_type] = (asset_allocation[asset_type] / self.portfolio_value) * 100
        
        portfolio_state.asset_allocation = dict(asset_allocation)
        
        # Add to history
        self.portfolio_history.append(portfolio_state)
        
        # Keep only recent history (last 1000 entries)
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-1000:]
        
        # Update performance metrics
        portfolio_values = [state.total_value for state in self.portfolio_history]
        self.performance_metrics.update_metrics(self.trades, portfolio_values)
    
    async def _load_state_from_database(self):
        """Load trading state from database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load positions
            cursor = conn.execute("SELECT * FROM positions")
            for row in cursor.fetchall():
                position = Position(
                    id=row[0],
                    symbol=row[1],
                    asset_type=AssetType(row[2]),
                    side=PositionSide(row[3]),
                    quantity=row[4],
                    entry_price=row[5],
                    current_price=row[6],
                    market_value=row[7],
                    unrealized_pnl=row[8],
                    stop_loss=row[9],
                    take_profit=row[10],
                    opened_at=datetime.fromisoformat(row[11]),
                    updated_at=datetime.fromisoformat(row[12]),
                    strategy_name=row[13] or "",
                    regime_at_entry=RegimeType(row[14]) if row[14] else None
                )
                self.positions[position.id] = position
            
            # Load recent trades (last 100)
            cursor = conn.execute("SELECT * FROM trades ORDER BY entry_time DESC LIMIT 100")
            for row in cursor.fetchall():
                trade = Trade(
                    id=row[0],
                    symbol=row[1],
                    asset_type=AssetType(row[2]),
                    side=PositionSide(row[3]),
                    signal_type=SignalType(row[4]),
                    quantity=row[5],
                    entry_price=row[6],
                    exit_price=row[7],
                    entry_time=datetime.fromisoformat(row[8]),
                    exit_time=datetime.fromisoformat(row[9]) if row[9] else None,
                    realized_pnl=row[10],
                    realized_pnl_pct=row[11],
                    fees=row[12],
                    strategy_name=row[13] or "",
                    regime_at_entry=RegimeType(row[14]) if row[14] else None,
                    regime_at_exit=RegimeType(row[15]) if row[15] else None,
                    exit_reason=row[16] or "",
                    duration_minutes=row[17] or 0
                )
                self.trades.append(trade)
            
            conn.close()
            logger.info(f"Loaded {len(self.positions)} positions and {len(self.trades)} trades from database")
            
        except Exception as e:
            logger.warning(f"Error loading state from database: {e}")
    
    async def _save_state_to_database(self):
        """Save current trading state to database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Save positions
            conn.execute("DELETE FROM positions")
            for position in self.positions.values():
                conn.execute('''
                    INSERT INTO positions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    position.id, position.symbol, position.asset_type.value, position.side.value,
                    position.quantity, position.entry_price, position.current_price, position.market_value,
                    position.unrealized_pnl, position.stop_loss, position.take_profit,
                    position.opened_at.isoformat(), position.updated_at.isoformat(),
                    position.strategy_name, position.regime_at_entry.value if position.regime_at_entry else None
                ))
            
            # Save new trades (only those not already in database)
            for trade in self.trades[-10:]:  # Save last 10 trades
                conn.execute('''
                    INSERT OR REPLACE INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade.id, trade.symbol, trade.asset_type.value, trade.side.value,
                    trade.signal_type.value, trade.quantity, trade.entry_price, trade.exit_price,
                    trade.entry_time.isoformat(), trade.exit_time.isoformat() if trade.exit_time else None,
                    trade.realized_pnl, trade.realized_pnl_pct, trade.fees, trade.strategy_name,
                    trade.regime_at_entry.value if trade.regime_at_entry else None,
                    trade.regime_at_exit.value if trade.regime_at_exit else None,
                    trade.exit_reason, trade.duration_minutes
                ))
            
            # Save portfolio history (last entry)
            if self.portfolio_history:
                latest_state = self.portfolio_history[-1]
                conn.execute('''
                    INSERT INTO portfolio_history VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    latest_state.timestamp.isoformat(), latest_state.total_value,
                    latest_state.cash_balance, latest_state.invested_value,
                    latest_state.total_pnl, latest_state.num_positions
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Error saving state to database: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        
        current_state = self.portfolio_history[-1] if self.portfolio_history else None
        
        return {
            'portfolio_value': self.portfolio_value,
            'cash_balance': self.cash_balance,
            'invested_value': sum(pos.market_value for pos in self.positions.values()),
            'total_pnl': sum(pos.unrealized_pnl for pos in self.positions.values()) + sum(trade.realized_pnl for trade in self.trades),
            'active_positions': len(self.positions),
            'total_trades': len(self.trades),
            'performance_metrics': asdict(self.performance_metrics),
            'current_state': asdict(current_state) if current_state else None,
            'last_update': self.last_update.isoformat(),
            'trading_mode': self.trading_mode.value,
            'is_active': self.is_trading_active
        }
    
    def save_state(self):
        """Save current state (synchronous version)"""
        asyncio.create_task(self._save_state_to_database())

# ==================== MAIN FUNCTION ====================

if __name__ == "__main__":
    # Test trading engine
    print("⚡ Testing NeuroCluster Elite Trading Engine")
    print("=" * 50)
    
    # This would be a more comprehensive test in production
    async def test_trading_engine():
        from src.core.neurocluster_elite import NeuroClusterElite
        from src.data.multi_asset_manager import MultiAssetDataManager
        
        # Initialize components
        neurocluster = NeuroClusterElite()
        data_manager = MultiAssetDataManager()
        trading_engine = AdvancedTradingEngine(neurocluster, data_manager)
        
        await trading_engine.initialize()
        
        # Test data
        symbols = ['AAPL', 'BTC-USD']
        asset_types = {'AAPL': AssetType.STOCK, 'BTC-USD': AssetType.CRYPTO}
        
        # Execute trading cycle
        result = await trading_engine.execute_trading_cycle(symbols, asset_types)
        
        print(f"Trading cycle result: {result}")
        print(f"Portfolio summary: {trading_engine.get_portfolio_summary()}")
        
        await data_manager.cleanup()
    
    # Run test
    try:
        asyncio.run(test_trading_engine())
        print("✅ Trading engine test completed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")