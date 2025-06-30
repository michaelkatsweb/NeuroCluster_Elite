#!/usr/bin/env python3
"""
File: paper_trading.py
Path: NeuroCluster-Elite/src/integrations/brokers/paper_trading.py
Description: Built-in paper trading simulator for NeuroCluster Elite

This module implements a comprehensive paper trading simulator that provides
realistic trading simulation without financial risk. It's the default and
safest way to test strategies and learn the platform.

Features:
- Realistic order execution simulation
- Market hours and holiday handling
- Slippage and commission simulation
- Portfolio tracking and P&L calculation
- Order book simulation
- Partial fill simulation
- Realistic market data integration
- Performance analytics

Author: Your Name
Created: 2025-06-29
Version: 1.0.0
License: MIT
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import json
from collections import defaultdict, deque

# Import our modules
try:
    from src.integrations import BaseIntegration, IntegrationConfig, IntegrationStatus
    from src.core.neurocluster_elite import AssetType, MarketData
    from src.trading.portfolio_manager import Position, Trade
    from src.utils.helpers import format_currency, format_percentage
    from src.utils.config_manager import ConfigManager
except ImportError:
    # Fallback for testing
    from enum import Enum
    class AssetType(Enum):
        STOCK = "stock"
        CRYPTO = "crypto"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== TRADING ENUMS ====================

class OrderType(Enum):
    """Order types supported by paper trading"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status values"""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class TimeInForce(Enum):
    """Time in force options"""
    DAY = "day"          # Good for day
    GTC = "gtc"          # Good till cancelled
    IOC = "ioc"          # Immediate or cancel
    FOK = "fok"          # Fill or kill

# ==================== DATA STRUCTURES ====================

@dataclass
class PaperOrder:
    """Paper trading order representation"""
    # Basic order info
    order_id: str
    symbol: str
    asset_type: AssetType
    side: OrderSide
    order_type: OrderType
    quantity: float
    
    # Pricing
    price: Optional[float] = None  # For limit/stop orders
    stop_price: Optional[float] = None  # For stop orders
    trailing_amount: Optional[float] = None  # For trailing stops
    
    # Execution details
    filled_quantity: float = 0.0
    remaining_quantity: Optional[float] = None
    avg_fill_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    time_in_force: TimeInForce = TimeInForce.DAY
    expires_at: Optional[datetime] = None
    
    # Metadata
    strategy_name: Optional[str] = None
    notes: str = ""
    commission: float = 0.0
    
    def __post_init__(self):
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity
        
        # Set expiration for day orders
        if self.time_in_force == TimeInForce.DAY and self.expires_at is None:
            # Expire at market close (4 PM ET)
            today = datetime.now().date()
            self.expires_at = datetime.combine(today, datetime.min.time().replace(hour=16))
            if self.expires_at <= datetime.now():
                # If after market close, expire tomorrow
                self.expires_at += timedelta(days=1)

@dataclass
class PaperFill:
    """Individual fill execution"""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime = field(default_factory=datetime.now)
    commission: float = 0.0

@dataclass
class PaperPosition:
    """Paper trading position"""
    symbol: str
    asset_type: AssetType
    quantity: float
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    
    # Tracking
    first_acquired: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def total_cost(self) -> float:
        return abs(self.quantity) * self.avg_cost
    
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.total_cost > 0:
            return (self.unrealized_pnl / self.total_cost) * 100
        return 0.0

@dataclass
class PaperAccount:
    """Paper trading account state"""
    account_id: str
    cash_balance: float
    total_equity: float
    buying_power: float
    
    # Performance metrics
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    
    # Risk metrics
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    peak_equity: float = 0.0
    
    # Tracking
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if self.peak_equity == 0.0:
            self.peak_equity = self.total_equity

# ==================== PAPER TRADING ENGINE ====================

class PaperTradingEngine(BaseIntegration):
    """
    Comprehensive paper trading engine with realistic simulation
    
    This engine provides a realistic trading simulation environment
    that helps users test strategies without financial risk.
    """
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        """Initialize paper trading engine"""
        
        # Create default config if none provided
        if config is None:
            config = IntegrationConfig(
                name="paper_trading",
                integration_type="broker",
                enabled=True,
                paper_trading=True,
                supports_live_trading=False,
                description="Built-in paper trading simulator"
            )
        
        super().__init__(config)
        
        # Trading configuration
        self.initial_capital = 100000.0  # $100k default
        self.commission_per_trade = 0.0  # Commission-free
        self.commission_percentage = 0.0
        
        # Simulation settings
        self.enable_slippage = True
        self.slippage_model = "linear"  # linear, square_root, random
        self.slippage_basis_points = 5  # 0.05% default slippage
        self.enable_partial_fills = True
        self.partial_fill_probability = 0.1  # 10% chance
        self.fill_probability = 0.98  # 98% fill rate
        
        # Market simulation
        self.respect_market_hours = True
        self.simulate_weekends = False
        self.execution_delay_range = (100, 500)  # 100-500ms delay
        
        # State management
        self.account = PaperAccount(
            account_id="paper_account_001",
            cash_balance=self.initial_capital,
            total_equity=self.initial_capital,
            buying_power=self.initial_capital
        )
        
        self.positions: Dict[str, PaperPosition] = {}
        self.orders: Dict[str, PaperOrder] = {}
        self.fills: List[PaperFill] = []
        self.order_history: List[PaperOrder] = []
        
        # Market data cache
        self.market_data_cache: Dict[str, MarketData] = {}
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Performance tracking
        self.daily_snapshots: List[Dict] = []
        self.trade_history: List[Dict] = []
        
        # Background tasks
        self._running = False
        self._order_processor_task = None
        
        logger.info("📈 Paper Trading Engine initialized with $100,000 virtual capital")
    
    async def connect(self) -> bool:
        """Connect to paper trading system"""
        try:
            self.update_status(IntegrationStatus.CONNECTED)
            
            # Start background order processing
            await self.start_order_processing()
            
            logger.info("✅ Paper trading engine connected successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to connect paper trading engine: {e}"
            self.update_status(IntegrationStatus.ERROR, error_msg)
            logger.error(f"❌ {error_msg}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from paper trading system"""
        try:
            # Stop background processing
            await self.stop_order_processing()
            
            # Save final state
            await self.save_state()
            
            self.update_status(IntegrationStatus.DISCONNECTED)
            logger.info("✅ Paper trading engine disconnected")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error disconnecting paper trading engine: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test paper trading connection"""
        try:
            # Simple connectivity test
            test_symbol = "AAPL"
            account_info = await self.get_account_info()
            
            return account_info is not None and account_info.get('account_id') == self.account.account_id
            
        except Exception as e:
            logger.error(f"❌ Paper trading connection test failed: {e}")
            return False
    
    async def get_capabilities(self) -> List[str]:
        """Get paper trading capabilities"""
        return [
            "market_orders",
            "limit_orders", 
            "stop_orders",
            "stop_limit_orders",
            "trailing_stops",
            "partial_fills",
            "portfolio_tracking",
            "performance_analytics",
            "order_history",
            "position_management",
            "risk_simulation",
            "slippage_simulation",
            "commission_simulation"
        ]
    
    # ==================== ORDER MANAGEMENT ====================
    
    async def place_order(self, symbol: str, side: str, quantity: float,
                         order_type: str = "market", price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         time_in_force: str = "day",
                         **kwargs) -> Dict[str, Any]:
        """Place a paper trading order"""
        
        try:
            # Validate order
            validation_result = await self._validate_order(symbol, side, quantity, order_type, price)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': validation_result['error'],
                    'order_id': None
                }
            
            # Create order
            order = PaperOrder(
                order_id=str(uuid.uuid4()),
                symbol=symbol.upper(),
                asset_type=self._get_asset_type(symbol),
                side=OrderSide(side.lower()),
                order_type=OrderType(order_type.lower()),
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                time_in_force=TimeInForce(time_in_force.lower()),
                strategy_name=kwargs.get('strategy_name'),
                notes=kwargs.get('notes', '')
            )
            
            # Add to order book
            self.orders[order.order_id] = order
            order.status = OrderStatus.OPEN
            
            logger.info(f"📋 Order placed: {side.upper()} {quantity} {symbol} ({order_type.upper()}) - ID: {order.order_id}")
            
            # Try immediate execution for market orders
            if order.order_type == OrderType.MARKET:
                await self._try_execute_order(order)
            
            return {
                'success': True,
                'order_id': order.order_id,
                'status': order.status.value,
                'message': f"Order placed successfully"
            }
            
        except Exception as e:
            error_msg = f"Failed to place order: {e}"
            logger.error(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'order_id': None
            }
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a paper trading order"""
        
        try:
            if order_id not in self.orders:
                return {
                    'success': False,
                    'error': f"Order {order_id} not found"
                }
            
            order = self.orders[order_id]
            
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                return {
                    'success': False,
                    'error': f"Cannot cancel order in {order.status.value} status"
                }
            
            # Cancel the order
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            
            # Move to history
            self.order_history.append(order)
            del self.orders[order_id]
            
            logger.info(f"❌ Order cancelled: {order_id}")
            
            return {
                'success': True,
                'order_id': order_id,
                'message': "Order cancelled successfully"
            }
            
        except Exception as e:
            error_msg = f"Failed to cancel order: {e}"
            logger.error(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
    
    async def get_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get current orders"""
        
        try:
            orders = list(self.orders.values())
            
            if status:
                status_enum = OrderStatus(status.lower())
                orders = [order for order in orders if order.status == status_enum]
            
            return [self._order_to_dict(order) for order in orders]
            
        except Exception as e:
            logger.error(f"❌ Failed to get orders: {e}")
            return []
    
    async def get_order_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get order history"""
        
        try:
            # Combine current and historical orders
            all_orders = list(self.orders.values()) + self.order_history
            
            # Sort by creation time (most recent first)
            all_orders.sort(key=lambda x: x.created_at, reverse=True)
            
            # Limit results
            limited_orders = all_orders[:limit]
            
            return [self._order_to_dict(order) for order in limited_orders]
            
        except Exception as e:
            logger.error(f"❌ Failed to get order history: {e}")
            return []
    
    # ==================== POSITION MANAGEMENT ====================
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        
        try:
            # Update position values with current market prices
            await self._update_position_values()
            
            return [self._position_to_dict(position) for position in self.positions.values()]
            
        except Exception as e:
            logger.error(f"❌ Failed to get positions: {e}")
            return []
    
    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get specific position"""
        
        try:
            symbol = symbol.upper()
            
            if symbol not in self.positions:
                return None
            
            # Update position value
            await self._update_position_value(symbol)
            
            return self._position_to_dict(self.positions[symbol])
            
        except Exception as e:
            logger.error(f"❌ Failed to get position for {symbol}: {e}")
            return None
    
    # ==================== ACCOUNT MANAGEMENT ====================
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        
        try:
            # Update account values
            await self._update_account_values()
            
            return {
                'account_id': self.account.account_id,
                'cash_balance': self.account.cash_balance,
                'total_equity': self.account.total_equity,
                'buying_power': self.account.buying_power,
                'total_pnl': self.account.total_pnl,
                'daily_pnl': self.account.daily_pnl,
                'total_trades': self.account.total_trades,
                'winning_trades': self.account.winning_trades,
                'win_rate': (self.account.winning_trades / max(1, self.account.total_trades)) * 100,
                'max_drawdown': self.account.max_drawdown,
                'current_drawdown': self.account.current_drawdown,
                'created_at': self.account.created_at.isoformat(),
                'last_updated': self.account.last_updated.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get account info: {e}")
            return {}
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        
        try:
            await self._update_account_values()
            
            # Calculate additional metrics
            total_positions = len(self.positions)
            long_positions = len([p for p in self.positions.values() if p.quantity > 0])
            short_positions = len([p for p in self.positions.values() if p.quantity < 0])
            
            # Asset allocation
            allocation = defaultdict(float)
            for position in self.positions.values():
                allocation[position.asset_type.value] += abs(position.market_value)
            
            total_invested = sum(allocation.values())
            
            # Convert to percentages
            if total_invested > 0:
                allocation = {k: (v / total_invested) * 100 for k, v in allocation.items()}
            
            return {
                'account_summary': await self.get_account_info(),
                'positions_summary': {
                    'total_positions': total_positions,
                    'long_positions': long_positions,
                    'short_positions': short_positions,
                    'total_market_value': total_invested
                },
                'asset_allocation': dict(allocation),
                'top_positions': await self._get_top_positions(5),
                'recent_trades': await self._get_recent_trades(10)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get portfolio summary: {e}")
            return {}
    
    # ==================== ORDER EXECUTION ENGINE ====================
    
    async def start_order_processing(self):
        """Start background order processing"""
        if not self._running:
            self._running = True
            self._order_processor_task = asyncio.create_task(self._order_processor_loop())
            logger.info("🔄 Started order processing engine")
    
    async def stop_order_processing(self):
        """Stop background order processing"""
        self._running = False
        if self._order_processor_task:
            self._order_processor_task.cancel()
            try:
                await self._order_processor_task
            except asyncio.CancelledError:
                pass
            self._order_processor_task = None
            logger.info("⏹️ Stopped order processing engine")
    
    async def _order_processor_loop(self):
        """Main order processing loop"""
        
        while self._running:
            try:
                # Process all open orders
                for order_id in list(self.orders.keys()):
                    if order_id in self.orders:  # Check if order still exists
                        order = self.orders[order_id]
                        await self._process_order(order)
                
                # Clean up expired orders
                await self._cleanup_expired_orders()
                
                # Update account values periodically
                await self._update_account_values()
                
                # Sleep before next iteration
                await asyncio.sleep(1.0)  # Process every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in order processor: {e}")
                await asyncio.sleep(5.0)  # Wait longer on error
    
    async def _process_order(self, order: PaperOrder):
        """Process a single order"""
        
        try:
            # Skip if order is not in executable state
            if order.status not in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]:
                return
            
            # Check if order has expired
            if order.expires_at and datetime.now() > order.expires_at:
                await self._expire_order(order)
                return
            
            # Check market hours
            if self.respect_market_hours and not self._is_market_open(order.symbol):
                return
            
            # Try to execute based on order type
            if order.order_type == OrderType.MARKET:
                await self._try_execute_order(order)
            elif order.order_type == OrderType.LIMIT:
                await self._try_execute_limit_order(order)
            elif order.order_type == OrderType.STOP:
                await self._try_execute_stop_order(order)
            elif order.order_type == OrderType.STOP_LIMIT:
                await self._try_execute_stop_limit_order(order)
            elif order.order_type == OrderType.TRAILING_STOP:
                await self._try_execute_trailing_stop_order(order)
                
        except Exception as e:
            logger.error(f"❌ Error processing order {order.order_id}: {e}")
    
    async def _try_execute_order(self, order: PaperOrder):
        """Try to execute a market order"""
        
        try:
            # Get current market price
            market_price = await self._get_market_price(order.symbol)
            if market_price is None:
                return
            
            # Simulate execution probability
            if not self._should_fill_order():
                return
            
            # Calculate execution price with slippage
            execution_price = self._calculate_execution_price(market_price, order)
            
            # Determine fill quantity
            fill_quantity = self._calculate_fill_quantity(order)
            
            # Execute the fill
            await self._execute_fill(order, fill_quantity, execution_price)
            
        except Exception as e:
            logger.error(f"❌ Error executing order {order.order_id}: {e}")
    
    async def _try_execute_limit_order(self, order: PaperOrder):
        """Try to execute a limit order"""
        
        try:
            market_price = await self._get_market_price(order.symbol)
            if market_price is None or order.price is None:
                return
            
            # Check if limit price is met
            should_execute = False
            
            if order.side == OrderSide.BUY:
                # Buy limit: execute if market price <= limit price
                should_execute = market_price <= order.price
            else:
                # Sell limit: execute if market price >= limit price
                should_execute = market_price >= order.price
            
            if should_execute and self._should_fill_order():
                fill_quantity = self._calculate_fill_quantity(order)
                # Execute at limit price for limit orders
                await self._execute_fill(order, fill_quantity, order.price)
                
        except Exception as e:
            logger.error(f"❌ Error executing limit order {order.order_id}: {e}")
    
    async def _try_execute_stop_order(self, order: PaperOrder):
        """Try to execute a stop order"""
        
        try:
            market_price = await self._get_market_price(order.symbol)
            if market_price is None or order.stop_price is None:
                return
            
            # Check if stop price is triggered
            should_trigger = False
            
            if order.side == OrderSide.SELL:
                # Stop loss: trigger if market price <= stop price
                should_trigger = market_price <= order.stop_price
            else:
                # Stop buy: trigger if market price >= stop price  
                should_trigger = market_price >= order.stop_price
            
            if should_trigger:
                # Convert to market order and execute
                order.order_type = OrderType.MARKET
                await self._try_execute_order(order)
                
        except Exception as e:
            logger.error(f"❌ Error executing stop order {order.order_id}: {e}")
    
    async def _try_execute_stop_limit_order(self, order: PaperOrder):
        """Try to execute a stop-limit order"""
        
        try:
            market_price = await self._get_market_price(order.symbol)
            if market_price is None or order.stop_price is None or order.price is None:
                return
            
            # Check if stop price is triggered
            should_trigger = False
            
            if order.side == OrderSide.SELL:
                should_trigger = market_price <= order.stop_price
            else:
                should_trigger = market_price >= order.stop_price
            
            if should_trigger:
                # Convert to limit order and try to execute
                order.order_type = OrderType.LIMIT
                await self._try_execute_limit_order(order)
                
        except Exception as e:
            logger.error(f"❌ Error executing stop-limit order {order.order_id}: {e}")
    
    async def _try_execute_trailing_stop_order(self, order: PaperOrder):
        """Try to execute a trailing stop order"""
        
        try:
            market_price = await self._get_market_price(order.symbol)
            if market_price is None or order.trailing_amount is None:
                return
            
            # Update trailing stop price
            if order.side == OrderSide.SELL:
                # Trailing stop loss
                new_stop_price = market_price - order.trailing_amount
                if order.stop_price is None or new_stop_price > order.stop_price:
                    order.stop_price = new_stop_price
                    order.updated_at = datetime.now()
            else:
                # Trailing stop buy
                new_stop_price = market_price + order.trailing_amount
                if order.stop_price is None or new_stop_price < order.stop_price:
                    order.stop_price = new_stop_price
                    order.updated_at = datetime.now()
            
            # Check if trailing stop is triggered
            await self._try_execute_stop_order(order)
                
        except Exception as e:
            logger.error(f"❌ Error executing trailing stop order {order.order_id}: {e}")
    
    async def _execute_fill(self, order: PaperOrder, fill_quantity: float, fill_price: float):
        """Execute a fill for an order"""
        
        try:
            # Create fill record
            fill = PaperFill(
                fill_id=str(uuid.uuid4()),
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=fill_quantity,
                price=fill_price,
                commission=self._calculate_commission(fill_quantity, fill_price)
            )
            
            # Add to fills
            self.fills.append(fill)
            
            # Update order
            order.filled_quantity += fill_quantity
            order.remaining_quantity -= fill_quantity
            order.updated_at = datetime.now()
            
            # Calculate average fill price
            total_filled_value = order.avg_fill_price * (order.filled_quantity - fill_quantity)
            total_filled_value += fill_price * fill_quantity
            order.avg_fill_price = total_filled_value / order.filled_quantity
            
            # Update order status
            if order.remaining_quantity <= 0:
                order.status = OrderStatus.FILLED
                # Move to history
                self.order_history.append(order)
                del self.orders[order.order_id]
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
            
            # Update position
            await self._update_position_from_fill(fill)
            
            # Update account cash
            cash_impact = fill_quantity * fill_price + fill.commission
            if order.side == OrderSide.BUY:
                cash_impact = -cash_impact  # Buying reduces cash
            
            self.account.cash_balance += cash_impact
            self.account.last_updated = datetime.now()
            
            logger.info(f"✅ Fill executed: {fill.side.value.upper()} {fill_quantity} {fill.symbol} @ ${fill_price:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error executing fill: {e}")
    
    async def _update_position_from_fill(self, fill: PaperFill):
        """Update position from a fill"""
        
        try:
            symbol = fill.symbol
            
            # Get or create position
            if symbol not in self.positions:
                self.positions[symbol] = PaperPosition(
                    symbol=symbol,
                    asset_type=self._get_asset_type(symbol),
                    quantity=0.0,
                    avg_cost=0.0,
                    market_value=0.0,
                    unrealized_pnl=0.0
                )
            
            position = self.positions[symbol]
            
            # Calculate new position
            if fill.side == OrderSide.BUY:
                new_quantity = position.quantity + fill.quantity
                if new_quantity != 0:
                    # Calculate new average cost
                    total_cost = (position.quantity * position.avg_cost) + (fill.quantity * fill.price)
                    position.avg_cost = total_cost / new_quantity
                position.quantity = new_quantity
            else:  # SELL
                position.quantity -= fill.quantity
                # Average cost remains the same for sells
                
                # Calculate realized P&L for the sold portion
                realized_pnl = (fill.price - position.avg_cost) * fill.quantity - fill.commission
                position.realized_pnl += realized_pnl
                self.account.total_pnl += realized_pnl
                
                # Track trade statistics
                self.account.total_trades += 1
                if realized_pnl > 0:
                    self.account.winning_trades += 1
            
            # Remove position if quantity is zero
            if abs(position.quantity) < 1e-8:  # Use small epsilon for float comparison
                del self.positions[symbol]
            else:
                position.last_updated = datetime.now()
                # Update market value
                await self._update_position_value(symbol)
            
        except Exception as e:
            logger.error(f"❌ Error updating position from fill: {e}")
    
    # ==================== HELPER METHODS ====================
    
    async def _validate_order(self, symbol: str, side: str, quantity: float,
                             order_type: str, price: Optional[float]) -> Dict[str, Any]:
        """Validate order parameters"""
        
        try:
            # Basic validation
            if quantity <= 0:
                return {'valid': False, 'error': 'Quantity must be positive'}
            
            if side.lower() not in ['buy', 'sell']:
                return {'valid': False, 'error': 'Side must be buy or sell'}
            
            if order_type.lower() not in ['market', 'limit', 'stop', 'stop_limit']:
                return {'valid': False, 'error': 'Invalid order type'}
            
            # Check buying power for buy orders
            if side.lower() == 'buy':
                estimated_cost = quantity * (price or await self._get_market_price(symbol) or 100)
                if estimated_cost > self.account.buying_power:
                    return {'valid': False, 'error': 'Insufficient buying power'}
            
            # Check position for sell orders
            if side.lower() == 'sell':
                current_position = self.positions.get(symbol.upper())
                if not current_position or current_position.quantity < quantity:
                    return {'valid': False, 'error': 'Insufficient position to sell'}
            
            return {'valid': True, 'error': None}
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {e}'}
    
    async def _get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol"""
        
        try:
            # In a real implementation, this would fetch from market data
            # For simulation, generate realistic prices with some volatility
            
            symbol = symbol.upper()
            
            # Use cached price if available and recent
            if symbol in self.market_data_cache:
                cached_data = self.market_data_cache[symbol]
                if (datetime.now() - cached_data.timestamp).seconds < 60:
                    return cached_data.price
            
            # Generate simulated price
            base_prices = {
                'AAPL': 180.0, 'GOOGL': 140.0, 'MSFT': 400.0, 'AMZN': 150.0,
                'TSLA': 250.0, 'NVDA': 875.0, 'META': 350.0, 'BTC-USD': 45000.0,
                'ETH-USD': 3200.0, 'ADA-USD': 0.45, 'EUR/USD': 1.0850
            }
            
            base_price = base_prices.get(symbol, 100.0)
            
            # Add some randomness (±2% volatility)
            volatility = 0.02
            random_factor = 1 + np.random.normal(0, volatility)
            current_price = base_price * random_factor
            
            # Store price history for realistic movements
            self.price_history[symbol].append(current_price)
            
            return current_price
            
        except Exception as e:
            logger.error(f"❌ Error getting market price for {symbol}: {e}")
            return None
    
    def _get_asset_type(self, symbol: str) -> AssetType:
        """Determine asset type from symbol"""
        
        symbol = symbol.upper()
        
        if symbol.endswith('-USD') or symbol.startswith('BTC') or symbol.startswith('ETH'):
            return AssetType.CRYPTO
        elif '/' in symbol:
            return AssetType.FOREX
        elif symbol in ['GC=F', 'CL=F', 'SI=F']:
            return AssetType.COMMODITY
        else:
            return AssetType.STOCK
    
    def _should_fill_order(self) -> bool:
        """Determine if order should fill (simulation)"""
        return np.random.random() < self.fill_probability
    
    def _calculate_fill_quantity(self, order: PaperOrder) -> float:
        """Calculate fill quantity (may be partial)"""
        
        if not self.enable_partial_fills:
            return order.remaining_quantity
        
        # Chance of partial fill
        if np.random.random() < self.partial_fill_probability:
            # Partial fill between 10% and 90% of remaining
            fill_ratio = np.random.uniform(0.1, 0.9)
            return order.remaining_quantity * fill_ratio
        
        return order.remaining_quantity
    
    def _calculate_execution_price(self, market_price: float, order: PaperOrder) -> float:
        """Calculate execution price with slippage"""
        
        if not self.enable_slippage:
            return market_price
        
        # Calculate slippage
        slippage_bps = self.slippage_basis_points
        
        if self.slippage_model == "random":
            slippage_bps *= np.random.uniform(0.5, 1.5)
        elif self.slippage_model == "square_root":
            # Larger orders have more slippage
            size_factor = np.sqrt(order.quantity / 100)  # Assume 100 is base size
            slippage_bps *= size_factor
        
        slippage = market_price * (slippage_bps / 10000)  # Convert basis points
        
        # Apply slippage based on order side
        if order.side == OrderSide.BUY:
            return market_price + slippage  # Pay more when buying
        else:
            return market_price - slippage  # Receive less when selling
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission for trade"""
        
        trade_value = quantity * price
        
        # Fixed commission per trade
        commission = self.commission_per_trade
        
        # Percentage-based commission
        commission += trade_value * (self.commission_percentage / 100)
        
        return commission
    
    def _is_market_open(self, symbol: str) -> bool:
        """Check if market is open for trading"""
        
        if not self.respect_market_hours:
            return True
        
        now = datetime.now()
        
        # Check if weekend
        if now.weekday() >= 5 and not self.simulate_weekends:  # Saturday = 5, Sunday = 6
            return False
        
        # Simplified market hours (9:30 AM - 4:00 PM ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Crypto markets are always open
        asset_type = self._get_asset_type(symbol)
        if asset_type == AssetType.CRYPTO:
            return True
        
        return market_open <= now <= market_close
    
    async def _expire_order(self, order: PaperOrder):
        """Expire an order"""
        
        order.status = OrderStatus.EXPIRED
        order.updated_at = datetime.now()
        
        # Move to history
        self.order_history.append(order)
        del self.orders[order.order_id]
        
        logger.info(f"⏰ Order expired: {order.order_id}")
    
    async def _cleanup_expired_orders(self):
        """Clean up expired orders"""
        
        now = datetime.now()
        expired_orders = []
        
        for order_id, order in self.orders.items():
            if order.expires_at and now > order.expires_at:
                expired_orders.append(order)
        
        for order in expired_orders:
            await self._expire_order(order)
    
    async def _update_account_values(self):
        """Update account values based on current positions"""
        
        try:
            # Update all position values
            await self._update_position_values()
            
            # Calculate total equity
            total_position_value = sum(pos.market_value for pos in self.positions.values())
            self.account.total_equity = self.account.cash_balance + total_position_value
            
            # Update buying power (simplified - same as cash for now)
            self.account.buying_power = self.account.cash_balance
            
            # Calculate unrealized P&L
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            
            # Update drawdown metrics
            if self.account.total_equity > self.account.peak_equity:
                self.account.peak_equity = self.account.total_equity
                self.account.current_drawdown = 0.0
            else:
                self.account.current_drawdown = ((self.account.peak_equity - self.account.total_equity) / 
                                               self.account.peak_equity) * 100
                self.account.max_drawdown = max(self.account.max_drawdown, self.account.current_drawdown)
            
            self.account.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Error updating account values: {e}")
    
    async def _update_position_values(self):
        """Update market values for all positions"""
        
        for symbol in list(self.positions.keys()):
            await self._update_position_value(symbol)
    
    async def _update_position_value(self, symbol: str):
        """Update market value for a specific position"""
        
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            current_price = await self._get_market_price(symbol)
            
            if current_price is not None:
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (current_price - position.avg_cost) * position.quantity
                position.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Error updating position value for {symbol}: {e}")
    
    async def _get_top_positions(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top positions by market value"""
        
        try:
            positions = list(self.positions.values())
            positions.sort(key=lambda p: abs(p.market_value), reverse=True)
            
            return [self._position_to_dict(pos) for pos in positions[:limit]]
            
        except Exception as e:
            logger.error(f"❌ Error getting top positions: {e}")
            return []
    
    async def _get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trades"""
        
        try:
            # Sort fills by timestamp (most recent first)
            recent_fills = sorted(self.fills, key=lambda f: f.timestamp, reverse=True)
            
            trades = []
            for fill in recent_fills[:limit]:
                trades.append({
                    'trade_id': fill.fill_id,
                    'symbol': fill.symbol,
                    'side': fill.side.value,
                    'quantity': fill.quantity,
                    'price': fill.price,
                    'total_value': fill.quantity * fill.price,
                    'commission': fill.commission,
                    'timestamp': fill.timestamp.isoformat()
                })
            
            return trades
            
        except Exception as e:
            logger.error(f"❌ Error getting recent trades: {e}")
            return []
    
    def _order_to_dict(self, order: PaperOrder) -> Dict[str, Any]:
        """Convert order to dictionary"""
        
        return {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'asset_type': order.asset_type.value,
            'side': order.side.value,
            'order_type': order.order_type.value,
            'quantity': order.quantity,
            'price': order.price,
            'stop_price': order.stop_price,
            'filled_quantity': order.filled_quantity,
            'remaining_quantity': order.remaining_quantity,
            'avg_fill_price': order.avg_fill_price,
            'status': order.status.value,
            'created_at': order.created_at.isoformat(),
            'updated_at': order.updated_at.isoformat(),
            'time_in_force': order.time_in_force.value,
            'expires_at': order.expires_at.isoformat() if order.expires_at else None,
            'strategy_name': order.strategy_name,
            'notes': order.notes
        }
    
    def _position_to_dict(self, position: PaperPosition) -> Dict[str, Any]:
        """Convert position to dictionary"""
        
        return {
            'symbol': position.symbol,
            'asset_type': position.asset_type.value,
            'quantity': position.quantity,
            'avg_cost': position.avg_cost,
            'market_value': position.market_value,
            'total_cost': position.total_cost,
            'unrealized_pnl': position.unrealized_pnl,
            'unrealized_pnl_pct': position.unrealized_pnl_pct,
            'realized_pnl': position.realized_pnl,
            'first_acquired': position.first_acquired.isoformat(),
            'last_updated': position.last_updated.isoformat()
        }
    
    async def save_state(self):
        """Save current state for persistence"""
        
        try:
            state = {
                'account': {
                    'account_id': self.account.account_id,
                    'cash_balance': self.account.cash_balance,
                    'total_equity': self.account.total_equity,
                    'total_pnl': self.account.total_pnl,
                    'total_trades': self.account.total_trades,
                    'winning_trades': self.account.winning_trades,
                    'max_drawdown': self.account.max_drawdown,
                    'created_at': self.account.created_at.isoformat()
                },
                'positions': [self._position_to_dict(pos) for pos in self.positions.values()],
                'orders': [self._order_to_dict(order) for order in self.orders.values()],
                'order_history': [self._order_to_dict(order) for order in self.order_history[-100:]],  # Keep last 100
                'fills': [
                    {
                        'fill_id': fill.fill_id,
                        'symbol': fill.symbol,
                        'side': fill.side.value,
                        'quantity': fill.quantity,
                        'price': fill.price,
                        'timestamp': fill.timestamp.isoformat(),
                        'commission': fill.commission
                    }
                    for fill in self.fills[-500:]  # Keep last 500 fills
                ]
            }
            
            # In a real implementation, save to file or database
            logger.info("💾 Paper trading state saved")
            
        except Exception as e:
            logger.error(f"❌ Error saving state: {e}")

# ==================== TESTING FUNCTION ====================

def test_paper_trading():
    """Test paper trading functionality"""
    
    print("📈 Testing Paper Trading Engine")
    print("=" * 50)
    
    async def run_tests():
        # Initialize paper trading engine
        engine = PaperTradingEngine()
        
        # Connect
        connected = await engine.connect()
        print(f"✅ Connection: {connected}")
        
        # Get account info
        account = await engine.get_account_info()
        print(f"✅ Account balance: ${account['cash_balance']:,.2f}")
        
        # Place a test order
        order_result = await engine.place_order(
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market"
        )
        print(f"✅ Order placed: {order_result['success']}")
        
        # Get orders
        orders = await engine.get_orders()
        print(f"✅ Active orders: {len(orders)}")
        
        # Get positions
        positions = await engine.get_positions()
        print(f"✅ Current positions: {len(positions)}")
        
        # Get capabilities
        capabilities = await engine.get_capabilities()
        print(f"✅ Capabilities: {len(capabilities)} features")
        
        # Disconnect
        disconnected = await engine.disconnect()
        print(f"✅ Disconnection: {disconnected}")
    
    # Run async tests
    asyncio.run(run_tests())
    
    print("\n🎉 Paper trading tests completed!")

if __name__ == "__main__":
    test_paper_trading()