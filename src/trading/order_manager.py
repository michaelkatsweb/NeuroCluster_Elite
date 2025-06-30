#!/usr/bin/env python3
"""
File: order_manager.py
Path: NeuroCluster-Elite/src/trading/order_manager.py
Description: Advanced order management and execution system for NeuroCluster Elite

This module handles order creation, execution, tracking, and management including
advanced order types, execution algorithms, and slippage simulation.

Features:
- Multiple order types (Market, Limit, Stop, etc.)
- Order execution simulation with realistic slippage
- Order tracking and status management
- Execution algorithms (TWAP, VWAP, etc.)
- Fill reporting and transaction cost analysis
- Order routing and smart execution
- Risk checks and order validation

Author: Your Name
Created: 2025-06-28
Version: 1.0.0
License: MIT
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading
import uuid
import json
import sqlite3
from pathlib import Path
import time
import random
from collections import defaultdict, deque

# Import our modules
try:
    from src.core.neurocluster_elite import AssetType, MarketData
    from src.trading.strategies.base_strategy import TradingSignal, SignalType
    from src.utils.logger import get_enhanced_logger, LogCategory
    from src.utils.helpers import format_currency, format_percentage
except ImportError:
    # Fallback for testing
    pass

# Configure logging
logger = get_enhanced_logger(__name__, LogCategory.TRADING)

# ==================== ENUMS AND DATA STRUCTURES ====================

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"

class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class TimeInForce(Enum):
    """Time in force"""
    DAY = "day"          # Good for day
    GTC = "gtc"          # Good till cancelled
    IOC = "ioc"          # Immediate or cancel
    FOK = "fok"          # Fill or kill
    GTD = "gtd"          # Good till date

class ExecutionAlgorithm(Enum):
    """Execution algorithms"""
    IMMEDIATE = "immediate"
    TWAP = "twap"        # Time-weighted average price
    VWAP = "vwap"        # Volume-weighted average price
    ICEBERG = "iceberg"   # Iceberg orders
    SNIPER = "sniper"     # Sniper algorithm

@dataclass
class Order:
    """Order representation"""
    id: str
    symbol: str
    asset_type: AssetType
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    time_in_force: TimeInForce = TimeInForce.DAY
    
    # Execution details
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    avg_fill_price: float = 0.0
    total_commission: float = 0.0
    
    # Status and timing
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Metadata
    strategy_name: Optional[str] = None
    parent_signal_id: Optional[str] = None
    notes: Optional[str] = None
    
    # Advanced features
    execution_algorithm: ExecutionAlgorithm = ExecutionAlgorithm.IMMEDIATE
    iceberg_quantity: Optional[float] = None  # For iceberg orders
    trailing_amount: Optional[float] = None   # For trailing stops
    
    def __post_init__(self):
        """Initialize computed fields"""
        self.remaining_quantity = self.quantity

@dataclass
class Fill:
    """Order fill representation"""
    id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    venue: Optional[str] = None
    liquidity: Optional[str] = None  # "maker" or "taker"

@dataclass
class ExecutionReport:
    """Execution report for order tracking"""
    order_id: str
    symbol: str
    status: OrderStatus
    filled_quantity: float
    remaining_quantity: float
    avg_price: float
    last_fill_price: Optional[float] = None
    last_fill_quantity: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    message: Optional[str] = None

# ==================== MARKET SIMULATOR ====================

class MarketSimulator:
    """
    Realistic market simulation for order execution
    
    Simulates:
    - Bid/ask spreads
    - Market impact and slippage
    - Partial fills
    - Market volatility effects
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.market_data: Dict[str, MarketData] = {}
        self.bid_ask_spreads: Dict[str, float] = {}
        
    def _default_config(self) -> Dict:
        """Default market simulation configuration"""
        return {
            'base_spread_bps': 5,      # Base spread in basis points
            'volatility_multiplier': 2.0,  # Spread multiplier for volatility
            'market_impact_bps': 1,    # Market impact per $1M traded
            'slippage_std_bps': 2,     # Random slippage standard deviation
            'partial_fill_probability': 0.1,  # Probability of partial fills
            'rejection_probability': 0.01,    # Probability of order rejection
            'latency_ms': (1, 10),     # Execution latency range
        }
    
    def update_market_data(self, symbol: str, data: MarketData):
        """Update market data for simulation"""
        self.market_data[symbol] = data
        
        # Calculate bid/ask spread based on volatility
        base_spread = self.config['base_spread_bps'] / 10000  # Convert to decimal
        volatility_factor = getattr(data, 'volatility', 20) / 20  # Normalize to 20% vol
        spread = base_spread * (1 + volatility_factor * self.config['volatility_multiplier'])
        
        self.bid_ask_spreads[symbol] = spread
    
    def simulate_execution(self, order: Order) -> Tuple[Optional[Fill], ExecutionReport]:
        """
        Simulate order execution with realistic market effects
        
        Returns:
            Tuple of (Fill or None, ExecutionReport)
        """
        
        # Check if we have market data
        if order.symbol not in self.market_data:
            return None, ExecutionReport(
                order_id=order.id,
                symbol=order.symbol,
                status=OrderStatus.REJECTED,
                filled_quantity=0,
                remaining_quantity=order.quantity,
                avg_price=0,
                message="No market data available"
            )
        
        market_data = self.market_data[order.symbol]
        
        # Simulate execution latency
        latency_range = self.config['latency_ms']
        execution_delay = random.uniform(latency_range[0], latency_range[1]) / 1000
        time.sleep(execution_delay)  # Simulate network latency
        
        # Check for order rejection
        if random.random() < self.config['rejection_probability']:
            return None, ExecutionReport(
                order_id=order.id,
                symbol=order.symbol,
                status=OrderStatus.REJECTED,
                filled_quantity=0,
                remaining_quantity=order.quantity,
                avg_price=0,
                message="Order rejected by venue"
            )
        
        # Calculate execution price
        execution_price = self._calculate_execution_price(order, market_data)
        
        if execution_price is None:
            # Order not executable at current price (e.g., limit order outside market)
            return None, ExecutionReport(
                order_id=order.id,
                symbol=order.symbol,
                status=OrderStatus.SUBMITTED,
                filled_quantity=0,
                remaining_quantity=order.quantity,
                avg_price=0,
                message="Order awaiting execution"
            )
        
        # Determine fill quantity (simulate partial fills)
        fill_quantity = order.remaining_quantity
        
        if (order.order_type != OrderType.MARKET and 
            random.random() < self.config['partial_fill_probability']):
            # Partial fill
            fill_quantity = random.uniform(0.1, 0.9) * order.remaining_quantity
            fill_quantity = max(1, round(fill_quantity))  # At least 1 share
        
        # Calculate commission
        commission = self._calculate_commission(order.symbol, fill_quantity, execution_price)
        
        # Create fill
        fill = Fill(
            id=f"F{uuid.uuid4().hex[:8]}",
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=execution_price,
            commission=commission,
            timestamp=datetime.now(),
            venue="NEUROCLUSTER_SIM",
            liquidity="taker" if order.order_type == OrderType.MARKET else "maker"
        )
        
        # Update order
        order.filled_quantity += fill_quantity
        order.remaining_quantity -= fill_quantity
        order.total_commission += commission
        
        # Calculate average fill price
        if order.filled_quantity > 0:
            order.avg_fill_price = ((order.avg_fill_price * (order.filled_quantity - fill_quantity)) + 
                                   (execution_price * fill_quantity)) / order.filled_quantity
        
        # Determine order status
        if order.remaining_quantity <= 0:
            status = OrderStatus.FILLED
            order.filled_at = datetime.now()
        else:
            status = OrderStatus.PARTIALLY_FILLED
        
        execution_report = ExecutionReport(
            order_id=order.id,
            symbol=order.symbol,
            status=status,
            filled_quantity=fill_quantity,
            remaining_quantity=order.remaining_quantity,
            avg_price=order.avg_fill_price,
            last_fill_price=execution_price,
            last_fill_quantity=fill_quantity,
            message=f"Filled {fill_quantity} @ {execution_price:.2f}"
        )
        
        return fill, execution_report
    
    def _calculate_execution_price(self, order: Order, market_data: MarketData) -> Optional[float]:
        """Calculate realistic execution price with slippage"""
        
        current_price = market_data.price
        spread = self.bid_ask_spreads.get(order.symbol, 0.001)  # Default 0.1% spread
        
        # Calculate bid/ask prices
        bid_price = current_price * (1 - spread / 2)
        ask_price = current_price * (1 + spread / 2)
        
        if order.order_type == OrderType.MARKET:
            # Market order - execute at bid/ask with slippage
            base_price = ask_price if order.side == OrderSide.BUY else bid_price
            
            # Add market impact based on order size
            order_value = order.remaining_quantity * current_price
            market_impact_bps = self.config['market_impact_bps'] * (order_value / 1000000)  # Per $1M
            market_impact = market_impact_bps / 10000
            
            if order.side == OrderSide.BUY:
                impact_price = base_price * (1 + market_impact)
            else:
                impact_price = base_price * (1 - market_impact)
            
            # Add random slippage
            slippage_std = self.config['slippage_std_bps'] / 10000
            slippage = random.normalvariate(0, slippage_std)
            
            final_price = impact_price * (1 + slippage)
            return max(0.01, final_price)  # Ensure positive price
        
        elif order.order_type == OrderType.LIMIT:
            # Limit order - only execute if price is favorable
            limit_price = order.price
            
            if order.side == OrderSide.BUY:
                # Buy limit: execute if ask <= limit price
                if ask_price <= limit_price:
                    return min(ask_price, limit_price)
            else:
                # Sell limit: execute if bid >= limit price
                if bid_price >= limit_price:
                    return max(bid_price, limit_price)
            
            return None  # Order not executable
        
        elif order.order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]:
            # Stop orders become market orders when triggered
            stop_price = order.stop_price
            
            if order.side == OrderSide.BUY:
                triggered = current_price >= stop_price
            else:
                triggered = current_price <= stop_price
            
            if triggered:
                # Execute as market order
                base_price = ask_price if order.side == OrderSide.BUY else bid_price
                slippage_std = self.config['slippage_std_bps'] / 10000 * 2  # Higher slippage for stops
                slippage = random.normalvariate(0, slippage_std)
                return base_price * (1 + slippage)
            
            return None  # Stop not triggered
        
        else:
            # Other order types - simplified execution
            return current_price
    
    def _calculate_commission(self, symbol: str, quantity: float, price: float) -> float:
        """Calculate trading commission"""
        
        # Simplified commission structure
        trade_value = quantity * price
        
        if trade_value <= 1000:
            return 1.00  # Minimum $1
        elif trade_value <= 10000:
            return 0.005 * trade_value  # 0.5%
        else:
            return 0.001 * trade_value  # 0.1%

# ==================== ORDER MANAGER ====================

class OrderManager:
    """
    Advanced order management system
    
    Features:
    - Order creation and validation
    - Execution tracking and reporting
    - Advanced order types
    - Execution algorithms
    - Risk checks
    """
    
    def __init__(self, config: Dict = None):
        """Initialize order manager"""
        
        self.config = config or self._default_config()
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self.execution_reports: List[ExecutionReport] = []
        
        # Market simulator
        self.market_simulator = MarketSimulator(self.config.get('market_sim', {}))
        
        # State management
        self.order_lock = threading.RLock()
        self.next_order_id = 1
        
        # Risk limits
        self.max_order_value = self.config.get('max_order_value', 50000)
        self.max_daily_orders = self.config.get('max_daily_orders', 100)
        self.daily_order_count = 0
        self.last_reset_date = datetime.now().date()
        
        # Execution callbacks
        self.fill_callbacks: List[Callable[[Fill], None]] = []
        self.order_update_callbacks: List[Callable[[Order, ExecutionReport], None]] = []
        
        # Database for persistence
        self.db_path = self.config.get('db_path', 'data/orders.db')
        self._init_database()
        
        logger.info("📋 Order Manager initialized")
    
    def _default_config(self) -> Dict:
        """Default order manager configuration"""
        return {
            'max_order_value': 50000,
            'max_daily_orders': 100,
            'default_time_in_force': 'day',
            'commission_rate': 0.001,
            'enable_risk_checks': True,
            'market_sim': {
                'base_spread_bps': 5,
                'market_impact_bps': 1
            },
            'db_path': 'data/orders.db'
        }
    
    def _init_database(self):
        """Initialize SQLite database for order persistence"""
        
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Orders table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id TEXT PRIMARY KEY,
                    symbol TEXT,
                    asset_type TEXT,
                    side TEXT,
                    order_type TEXT,
                    quantity REAL,
                    price REAL,
                    stop_price REAL,
                    time_in_force TEXT,
                    status TEXT,
                    filled_quantity REAL,
                    avg_fill_price REAL,
                    total_commission REAL,
                    created_at TEXT,
                    submitted_at TEXT,
                    filled_at TEXT,
                    strategy_name TEXT,
                    notes TEXT
                )
            """)
            
            # Fills table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fills (
                    id TEXT PRIMARY KEY,
                    order_id TEXT,
                    symbol TEXT,
                    side TEXT,
                    quantity REAL,
                    price REAL,
                    commission REAL,
                    timestamp TEXT,
                    venue TEXT,
                    liquidity TEXT
                )
            """)
            
            conn.commit()
    
    def create_order_from_signal(self, signal: TradingSignal, quantity: float,
                                order_type: OrderType = OrderType.MARKET,
                                time_in_force: TimeInForce = TimeInForce.DAY) -> Order:
        """
        Create order from trading signal
        
        Args:
            signal: Trading signal
            quantity: Order quantity
            order_type: Type of order
            time_in_force: Time in force
            
        Returns:
            Created order
        """
        
        # Determine order side
        side = OrderSide.BUY if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else OrderSide.SELL
        
        # Set prices based on signal
        price = None
        stop_price = None
        
        if order_type == OrderType.LIMIT:
            price = signal.entry_price
        elif order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]:
            stop_price = signal.stop_loss if order_type == OrderType.STOP_LOSS else signal.take_profit
        
        # Create order
        order = Order(
            id=self._generate_order_id(),
            symbol=signal.symbol,
            asset_type=signal.asset_type,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            strategy_name=signal.strategy_name,
            parent_signal_id=getattr(signal, 'id', None),
            notes=f"Generated from signal: {signal.reasoning[:100]}"
        )
        
        return order
    
    def submit_order(self, order: Order) -> bool:
        """
        Submit order for execution
        
        Args:
            order: Order to submit
            
        Returns:
            True if order was successfully submitted
        """
        
        with self.order_lock:
            # Reset daily counter if needed
            self._reset_daily_counters()
            
            # Risk checks
            if self.config.get('enable_risk_checks', True):
                risk_check_result = self._perform_risk_checks(order)
                if not risk_check_result[0]:
                    logger.warning(f"Order {order.id} failed risk check: {risk_check_result[1]}")
                    order.status = OrderStatus.REJECTED
                    self._create_execution_report(order, f"Risk check failed: {risk_check_result[1]}")
                    return False
            
            # Store order
            self.orders[order.id] = order
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now()
            
            # Update daily counter
            self.daily_order_count += 1
            
            # Save to database
            self._save_order(order)
            
            # Execute order (in real system, this would be sent to broker)
            self._execute_order(order)
            
            logger.info(f"📤 Order submitted: {order.id} - {order.side.value} {order.quantity} {order.symbol}")
            return True
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        
        with self.order_lock:
            if order_id not in self.orders:
                logger.warning(f"Order {order_id} not found for cancellation")
                return False
            
            order = self.orders[order_id]
            
            if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                logger.warning(f"Order {order_id} cannot be cancelled (status: {order.status})")
                return False
            
            order.status = OrderStatus.CANCELLED
            self._create_execution_report(order, "Order cancelled by user")
            
            logger.info(f"❌ Order cancelled: {order_id}")
            return True
    
    def update_market_data(self, symbol: str, market_data: MarketData):
        """Update market data for order execution simulation"""
        self.market_simulator.update_market_data(symbol, market_data)
        
        # Check for pending orders that might now be executable
        self._check_pending_orders(symbol)
    
    def _execute_order(self, order: Order):
        """Execute order through market simulator"""
        
        fill, execution_report = self.market_simulator.simulate_execution(order)
        
        # Update order status
        order.status = execution_report.status
        
        # Store execution report
        self.execution_reports.append(execution_report)
        
        # Handle fill
        if fill:
            self.fills.append(fill)
            self._save_fill(fill)
            
            # Notify callbacks
            for callback in self.fill_callbacks:
                try:
                    callback(fill)
                except Exception as e:
                    logger.error(f"Error in fill callback: {e}")
        
        # Update order in database
        self._save_order(order)
        
        # Notify order update callbacks
        for callback in self.order_update_callbacks:
            try:
                callback(order, execution_report)
            except Exception as e:
                logger.error(f"Error in order update callback: {e}")
        
        logger.info(f"📊 Order execution: {order.id} - {execution_report.message}")
    
    def _check_pending_orders(self, symbol: str):
        """Check if any pending orders for symbol can now be executed"""
        
        pending_orders = [
            order for order in self.orders.values()
            if order.symbol == symbol and order.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
        ]
        
        for order in pending_orders:
            if order.remaining_quantity > 0:
                self._execute_order(order)
    
    def _perform_risk_checks(self, order: Order) -> Tuple[bool, str]:
        """Perform pre-trade risk checks"""
        
        # Check daily order limit
        if self.daily_order_count >= self.max_daily_orders:
            return False, f"Daily order limit ({self.max_daily_orders}) exceeded"
        
        # Check order value limit
        if order.price:
            order_value = order.quantity * order.price
        else:
            # Estimate value for market orders
            order_value = order.quantity * 100  # Simplified estimate
        
        if order_value > self.max_order_value:
            return False, f"Order value ({format_currency(order_value)}) exceeds limit ({format_currency(self.max_order_value)})"
        
        # Check for minimum quantity
        if order.quantity <= 0:
            return False, "Order quantity must be positive"
        
        # Check for reasonable price (if limit order)
        if order.order_type == OrderType.LIMIT and order.price and order.price <= 0:
            return False, "Limit price must be positive"
        
        return True, "Risk checks passed"
    
    def _create_execution_report(self, order: Order, message: str):
        """Create execution report"""
        
        report = ExecutionReport(
            order_id=order.id,
            symbol=order.symbol,
            status=order.status,
            filled_quantity=order.filled_quantity,
            remaining_quantity=order.remaining_quantity,
            avg_price=order.avg_fill_price,
            message=message
        )
        
        self.execution_reports.append(report)
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        
        order_id = f"ORD{self.next_order_id:06d}"
        self.next_order_id += 1
        return order_id
    
    def _reset_daily_counters(self):
        """Reset daily counters if new day"""
        
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_order_count = 0
            self.last_reset_date = current_date
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get order status and details"""
        
        if order_id not in self.orders:
            return None
        
        order = self.orders[order_id]
        return {
            'id': order.id,
            'symbol': order.symbol,
            'side': order.side.value,
            'order_type': order.order_type.value,
            'quantity': order.quantity,
            'filled_quantity': order.filled_quantity,
            'remaining_quantity': order.remaining_quantity,
            'avg_fill_price': order.avg_fill_price,
            'status': order.status.value,
            'created_at': order.created_at.isoformat(),
            'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None,
            'filled_at': order.filled_at.isoformat() if order.filled_at else None
        }
    
    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get all open orders, optionally filtered by symbol"""
        
        open_orders = [
            order for order in self.orders.values()
            if order.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
        ]
        
        if symbol:
            open_orders = [order for order in open_orders if order.symbol == symbol]
        
        return [self.get_order_status(order.id) for order in open_orders]
    
    def get_execution_summary(self) -> Dict:
        """Get execution summary statistics"""
        
        total_orders = len(self.orders)
        filled_orders = len([o for o in self.orders.values() if o.status == OrderStatus.FILLED])
        partially_filled = len([o for o in self.orders.values() if o.status == OrderStatus.PARTIALLY_FILLED])
        cancelled_orders = len([o for o in self.orders.values() if o.status == OrderStatus.CANCELLED])
        
        total_fills = len(self.fills)
        total_volume = sum(fill.quantity * fill.price for fill in self.fills)
        total_commission = sum(fill.commission for fill in self.fills)
        
        # Calculate average execution metrics
        fill_rates = []
        for order in self.orders.values():
            if order.quantity > 0:
                fill_rates.append(order.filled_quantity / order.quantity)
        
        avg_fill_rate = np.mean(fill_rates) if fill_rates else 0
        
        return {
            'total_orders': total_orders,
            'filled_orders': filled_orders,
            'partially_filled_orders': partially_filled,
            'cancelled_orders': cancelled_orders,
            'fill_rate': f"{avg_fill_rate:.1%}",
            'total_fills': total_fills,
            'total_volume': format_currency(total_volume),
            'total_commission': format_currency(total_commission),
            'daily_orders_remaining': self.max_daily_orders - self.daily_order_count
        }
    
    def add_fill_callback(self, callback: Callable[[Fill], None]):
        """Add callback for order fills"""
        self.fill_callbacks.append(callback)
    
    def add_order_update_callback(self, callback: Callable[[Order, ExecutionReport], None]):
        """Add callback for order updates"""
        self.order_update_callbacks.append(callback)
    
    def _save_order(self, order: Order):
        """Save order to database"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO orders 
                (id, symbol, asset_type, side, order_type, quantity, price, stop_price,
                 time_in_force, status, filled_quantity, avg_fill_price, total_commission,
                 created_at, submitted_at, filled_at, strategy_name, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order.id, order.symbol, order.asset_type.value, order.side.value,
                order.order_type.value, order.quantity, order.price, order.stop_price,
                order.time_in_force.value, order.status.value, order.filled_quantity,
                order.avg_fill_price, order.total_commission,
                order.created_at.isoformat(),
                order.submitted_at.isoformat() if order.submitted_at else None,
                order.filled_at.isoformat() if order.filled_at else None,
                order.strategy_name, order.notes
            ))
    
    def _save_fill(self, fill: Fill):
        """Save fill to database"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO fills 
                (id, order_id, symbol, side, quantity, price, commission, timestamp, venue, liquidity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fill.id, fill.order_id, fill.symbol, fill.side.value,
                fill.quantity, fill.price, fill.commission,
                fill.timestamp.isoformat(), fill.venue, fill.liquidity
            ))

# ==================== TESTING ====================

def test_order_manager():
    """Test order manager functionality"""
    
    print("📋 Testing Order Manager")
    print("=" * 40)
    
    # Create order manager
    order_manager = OrderManager()
    
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
    
    # Update market data
    market_data = MarketData(
        symbol='AAPL',
        asset_type=AssetType.STOCK,
        price=150.0,
        timestamp=datetime.now()
    )
    order_manager.update_market_data('AAPL', market_data)
    
    # Create and submit order
    order = order_manager.create_order_from_signal(signal, 100, OrderType.MARKET)
    success = order_manager.submit_order(order)
    
    print(f"✅ Order submitted: {success}")
    print(f"   Order ID: {order.id}")
    print(f"   Symbol: {order.symbol}")
    print(f"   Quantity: {order.quantity}")
    print(f"   Status: {order.status.value}")
    
    # Check order status
    status = order_manager.get_order_status(order.id)
    if status:
        print(f"\n✅ Order Status:")
        print(f"   Filled: {status['filled_quantity']}/{status['quantity']}")
        print(f"   Avg Price: ${status['avg_fill_price']:.2f}")
        print(f"   Status: {status['status']}")
    
    # Get execution summary
    summary = order_manager.get_execution_summary()
    print(f"\n✅ Execution Summary:")
    print(f"   Total orders: {summary['total_orders']}")
    print(f"   Fill rate: {summary['fill_rate']}")
    print(f"   Total volume: {summary['total_volume']}")
    print(f"   Total commission: {summary['total_commission']}")
    
    print("\n🎉 Order manager tests completed!")

if __name__ == "__main__":
    test_order_manager()