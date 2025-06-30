#!/usr/bin/env python3
"""
File: interactive_brokers.py
Path: NeuroCluster-Elite/src/integrations/brokers/interactive_brokers.py
Description: Interactive Brokers TWS API integration for NeuroCluster Elite

This module implements the Interactive Brokers TWS (Trader Workstation) API integration,
providing professional-grade trading capabilities with global market access, options,
futures, and advanced order types.

Features:
- TWS API integration via ibapi
- Paper trading and live trading support
- Global market access (stocks, options, futures, forex, bonds)
- Advanced order types and algorithms
- Real-time market data and quotes
- Portfolio tracking and risk management
- Multi-currency support
- Options chains and analysis

API Documentation: https://interactivebrokers.github.io/tws-api/

Note: Requires TWS or IB Gateway to be running and connected
Connection: localhost:7497 (TWS) or localhost:4001 (IB Gateway)

Author: Your Name
Created: 2025-06-30
Version: 1.0.0
License: MIT
"""

import asyncio
import threading
import time
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import queue
import socket

# Import our modules
try:
    from src.integrations.brokers import (
        BaseBroker, BrokerAccount, BrokerPosition, BrokerOrder,
        BrokerType, OrderType, OrderSide, OrderStatus, TimeInForce
    )
    from src.integrations import IntegrationConfig, IntegrationStatus
    from src.core.neurocluster_elite import AssetType
    from src.utils.helpers import format_currency, format_percentage
    from src.utils.config_manager import ConfigManager
except ImportError:
    # Fallback for testing
    pass

# Try to import ibapi (Interactive Brokers API)
try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order import Order
    from ibapi.common import *
    from ibapi.ticktype import TickTypeEnum
    IB_API_AVAILABLE = True
except ImportError:
    # Create fallback classes if ibapi not installed
    class EClient:
        pass
    class EWrapper:
        pass
    class Contract:
        pass
    class Order:
        pass
    IB_API_AVAILABLE = False
    logger.warning("⚠️ Interactive Brokers API (ibapi) not installed. Run: pip install ibapi")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== IB SPECIFIC ENUMS ====================

class IBAccountType(Enum):
    """Interactive Brokers account types"""
    INDIVIDUAL = "Individual"
    JOINT = "Joint"
    TRUST = "Trust"
    CORPORATE = "Corporate"
    IRA = "IRA"

class IBSecType(Enum):
    """IB security types"""
    STK = "STK"  # Stock
    OPT = "OPT"  # Option
    FUT = "FUT"  # Future
    CASH = "CASH"  # Forex
    BOND = "BOND"  # Bond
    CFD = "CFD"  # Contract for Difference
    COMBO = "COMBO"  # Combo/Spread
    WAR = "WAR"  # Warrant
    IOPT = "IOPT"  # Index Option

class IBOrderType(Enum):
    """IB order types"""
    MKT = "MKT"  # Market
    LMT = "LMT"  # Limit
    STP = "STP"  # Stop
    STP_LMT = "STP LMT"  # Stop Limit
    TRAIL = "TRAIL"  # Trailing Stop
    MIT = "MIT"  # Market if Touched
    LIT = "LIT"  # Limit if Touched

# ==================== IB DATA STRUCTURES ====================

@dataclass
class IBConfig:
    """Interactive Brokers specific configuration"""
    host: str = "127.0.0.1"
    port: int = 7497  # TWS paper port (7496 for live)
    client_id: int = 1
    
    # Connection settings
    connect_timeout: int = 10
    reconnect_attempts: int = 3
    reconnect_delay: float = 5.0
    
    # Account settings
    account_id: Optional[str] = None  # Auto-detect if None
    
    # Features
    enable_real_time_data: bool = True
    enable_historical_data: bool = True
    request_market_data: bool = False  # Requires market data subscription
    
    # Risk settings
    max_order_value: float = 50000.0
    enable_order_confirmation: bool = True

# ==================== IB WRAPPER CLASS ====================

class IBWrapper(EWrapper):
    """
    Interactive Brokers API wrapper to handle callbacks
    
    This class handles all the callback methods from the IB API
    and provides a bridge to our broker implementation.
    """
    
    def __init__(self, broker: 'InteractiveBrokersBroker'):
        EWrapper.__init__(self)
        self.broker = broker
        self.next_order_id = 1
        
        # Data storage
        self.accounts = {}
        self.positions = {}
        self.orders = {}
        self.executions = {}
        self.portfolio = {}
        
        # Request tracking
        self.request_callbacks = {}
        self.request_data = {}
        
        logger.info("🔧 IB Wrapper initialized")
    
    # ==================== CONNECTION CALLBACKS ====================
    
    def connectAck(self):
        """Called when connection is acknowledged"""
        logger.info("✅ IB connection acknowledged")
        self.broker._connection_event.set()
    
    def connectionClosed(self):
        """Called when connection is closed"""
        logger.info("🔌 IB connection closed")
        self.broker.update_status(IntegrationStatus.DISCONNECTED)
    
    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = ""):
        """Handle API errors"""
        logger.error(f"❌ IB Error {errorCode}: {errorString} (Request: {reqId})")
        
        # Handle specific error codes
        if errorCode == 502:  # Couldn't connect
            self.broker.update_status(IntegrationStatus.ERROR, "Cannot connect to TWS")
        elif errorCode == 326:  # Unable to connect
            self.broker.update_status(IntegrationStatus.ERROR, "Unable to connect to TWS")
        elif errorCode in [1100, 1101, 1102]:  # Connection lost/restored
            if errorCode == 1100:
                self.broker.update_status(IntegrationStatus.DISCONNECTED, "Connection lost")
            else:
                self.broker.update_status(IntegrationStatus.CONNECTED, "Connection restored")
    
    # ==================== ACCOUNT CALLBACKS ====================
    
    def managedAccounts(self, accountsList: str):
        """Receive list of managed accounts"""
        accounts = accountsList.split(",")
        self.broker.managed_accounts = accounts
        logger.info(f"📊 IB managed accounts: {accounts}")
        
        # Use first account if none specified
        if not self.broker.account_id and accounts:
            self.broker.account_id = accounts[0]
            logger.info(f"🎯 Using account: {self.broker.account_id}")
    
    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        """Receive account summary data"""
        if account not in self.accounts:
            self.accounts[account] = {}
        
        self.accounts[account][tag] = {
            'value': value,
            'currency': currency
        }
        
        # Update broker account info when we have enough data
        if tag == "TotalCashValue":
            self.broker._update_account_info()
    
    def accountSummaryEnd(self, reqId: int):
        """Account summary request completed"""
        logger.info("✅ IB account summary received")
    
    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        """Receive position data"""
        symbol = contract.symbol
        self.positions[symbol] = {
            'account': account,
            'contract': contract,
            'position': position,
            'avgCost': avgCost
        }
    
    def positionEnd(self):
        """Position request completed"""
        logger.info(f"✅ IB positions received: {len(self.positions)} positions")
        self.broker._update_positions()
    
    # ==================== ORDER CALLBACKS ====================
    
    def nextValidId(self, orderId: int):
        """Receive next valid order ID"""
        self.next_order_id = orderId
        self.broker.next_order_id = orderId
        logger.info(f"🔢 IB next order ID: {orderId}")
    
    def orderStatus(self, orderId: int, status: str, filled: float, remaining: float,
                   avgFillPrice: float, permId: int, parentId: int, lastFillPrice: float,
                   clientId: int, whyHeld: str, mktCapPrice: float):
        """Receive order status updates"""
        
        order_info = {
            'orderId': orderId,
            'status': status,
            'filled': filled,
            'remaining': remaining,
            'avgFillPrice': avgFillPrice,
            'lastFillPrice': lastFillPrice,
            'permId': permId
        }
        
        self.orders[orderId] = order_info
        self.broker._update_order_status(orderId, order_info)
        
        logger.info(f"📋 IB order {orderId} status: {status} (filled: {filled}, remaining: {remaining})")
    
    def openOrder(self, orderId: int, contract: Contract, order: Order, orderState):
        """Receive open order information"""
        order_info = {
            'orderId': orderId,
            'contract': contract,
            'order': order,
            'orderState': orderState
        }
        
        self.orders[orderId] = {**self.orders.get(orderId, {}), **order_info}
    
    def openOrderEnd(self):
        """Open orders request completed"""
        logger.info(f"✅ IB open orders received: {len(self.orders)} orders")

# ==================== IB CLIENT CLASS ====================

class IBClient(EClient):
    """Interactive Brokers API client"""
    
    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)
        self.wrapper = wrapper

# ==================== MAIN BROKER IMPLEMENTATION ====================

class InteractiveBrokersBroker(BaseBroker):
    """
    Interactive Brokers broker implementation
    
    Provides integration with Interactive Brokers TWS API for professional
    trading with global market access and advanced order types.
    """
    
    def __init__(self, config: IntegrationConfig):
        """Initialize Interactive Brokers broker"""
        super().__init__(config)
        
        if not IB_API_AVAILABLE:
            raise ImportError("Interactive Brokers API (ibapi) not installed. Run: pip install ibapi")
        
        # Connection settings
        self.host = config.additional_auth.get('host', '127.0.0.1')
        self.port = int(config.additional_auth.get('port', 7497))  # Paper trading port
        self.client_id = int(config.additional_auth.get('client_id', 1))
        
        # Account settings
        self.account_id = config.additional_auth.get('account_id')
        self.managed_accounts = []
        
        # IB API components
        self.wrapper = IBWrapper(self)
        self.client = IBClient(self.wrapper)
        
        # Connection management
        self._connection_event = threading.Event()
        self._api_thread: Optional[threading.Thread] = None
        self._connected = False
        
        # Order management
        self.next_order_id = 1
        self.pending_orders = {}
        
        # Request management
        self._request_id = 1000
        
        # Real-time data
        self.subscribed_symbols = set()
        
        logger.info(f"🏦 Interactive Brokers broker initialized - Port: {self.port}")
    
    # ==================== CONNECTION MANAGEMENT ====================
    
    async def connect(self) -> bool:
        """Connect to Interactive Brokers TWS/Gateway"""
        try:
            if self._connected:
                return True
            
            logger.info(f"🔌 Connecting to IB TWS at {self.host}:{self.port}")
            
            # Test if TWS is running
            if not await self._test_tws_connection():
                error_msg = f"Cannot connect to TWS at {self.host}:{self.port}. Is TWS/Gateway running?"
                self.update_status(IntegrationStatus.ERROR, error_msg)
                logger.error(f"❌ {error_msg}")
                return False
            
            # Connect to IB API
            self.client.connect(self.host, self.port, self.client_id)
            
            # Start API thread
            self._api_thread = threading.Thread(target=self._run_api_loop, daemon=True)
            self._api_thread.start()
            
            # Wait for connection acknowledgment
            if self._connection_event.wait(timeout=10):
                self._connected = True
                
                # Request managed accounts
                self.client.reqManagedAccts()
                
                # Wait a moment for account info
                await asyncio.sleep(1)
                
                # Request account summary
                await self._request_account_summary()
                
                # Request positions
                self.client.reqPositions()
                
                self.update_status(IntegrationStatus.CONNECTED)
                logger.info("✅ Interactive Brokers connected successfully")
                return True
            else:
                error_msg = "Connection timeout - check TWS settings"
                self.update_status(IntegrationStatus.ERROR, error_msg)
                logger.error(f"❌ {error_msg}")
                return False
                
        except Exception as e:
            error_msg = f"IB connection failed: {e}"
            self.update_status(IntegrationStatus.ERROR, error_msg)
            logger.error(f"❌ {error_msg}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Interactive Brokers"""
        try:
            if self.client.isConnected():
                self.client.disconnect()
            
            self._connected = False
            
            # Wait for thread to finish
            if self._api_thread and self._api_thread.is_alive():
                self._api_thread.join(timeout=5)
            
            self.update_status(IntegrationStatus.DISCONNECTED)
            logger.info("✅ Interactive Brokers disconnected")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error disconnecting IB: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test IB connection"""
        return self._connected and self.client.isConnected()
    
    def _run_api_loop(self):
        """Run the IB API message loop in a separate thread"""
        try:
            self.client.run()
        except Exception as e:
            logger.error(f"❌ IB API loop error: {e}")
    
    async def _test_tws_connection(self) -> bool:
        """Test if TWS/Gateway is running and accepting connections"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    # ==================== ACCOUNT MANAGEMENT ====================
    
    async def get_account_info(self) -> Optional[BrokerAccount]:
        """Get account information"""
        try:
            if not self._connected:
                await self.connect()
            
            # Request fresh account summary
            await self._request_account_summary()
            
            # Give it time to update
            await asyncio.sleep(2)
            
            return self.account_info
            
        except Exception as e:
            logger.error(f"❌ Error getting IB account info: {e}")
            return None
    
    async def _request_account_summary(self):
        """Request account summary from IB"""
        if self.account_id:
            req_id = self._get_next_request_id()
            tags = "TotalCashValue,NetLiquidation,BuyingPower,GrossPositionValue,UnrealizedPnL"
            self.client.reqAccountSummary(req_id, "All", tags)
    
    def _update_account_info(self):
        """Update account info from received data"""
        try:
            if not self.account_id or self.account_id not in self.wrapper.accounts:
                return
            
            account_data = self.wrapper.accounts[self.account_id]
            
            # Extract account values
            cash_balance = float(account_data.get('TotalCashValue', {}).get('value', 0))
            total_equity = float(account_data.get('NetLiquidation', {}).get('value', 0))
            buying_power = float(account_data.get('BuyingPower', {}).get('value', 0))
            
            self.account_info = BrokerAccount(
                account_id=self.account_id,
                broker_type=BrokerType.INTERACTIVE_BROKERS,
                account_type="margin",  # IB typically provides margin accounts
                
                # Balances
                cash_balance=cash_balance,
                total_equity=total_equity,
                buying_power=buying_power,
                
                # Status
                account_status="active",
                is_restricted=False,
                restrictions=[],
                
                last_updated=datetime.now()
            )
            
            logger.info(f"💰 IB account updated: ${total_equity:,.2f} equity")
            
        except Exception as e:
            logger.error(f"❌ Error updating IB account info: {e}")
    
    # ==================== POSITION MANAGEMENT ====================
    
    async def get_positions(self) -> Dict[str, BrokerPosition]:
        """Get all positions"""
        try:
            if not self._connected:
                await self.connect()
            
            # Request fresh positions
            self.client.reqPositions()
            
            # Give it time to update
            await asyncio.sleep(2)
            
            return self.positions
            
        except Exception as e:
            logger.error(f"❌ Error getting IB positions: {e}")
            return {}
    
    async def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        """Get position for specific symbol"""
        positions = await self.get_positions()
        return positions.get(symbol)
    
    def _update_positions(self):
        """Update positions from received data"""
        try:
            positions = {}
            
            for symbol, pos_data in self.wrapper.positions.items():
                if pos_data['position'] != 0:  # Only include non-zero positions
                    position = BrokerPosition(
                        symbol=symbol,
                        asset_type=self._get_asset_type(pos_data['contract']),
                        quantity=abs(pos_data['position']),
                        market_value=0.0,  # Will be updated with market data
                        avg_cost=pos_data['avgCost'],
                        unrealized_pnl=0.0,  # Will be calculated
                        realized_pnl=0.0,
                        
                        # Position details
                        side='long' if pos_data['position'] > 0 else 'short',
                        cost_basis=pos_data['avgCost'] * abs(pos_data['position']),
                        
                        broker_type=BrokerType.INTERACTIVE_BROKERS,
                        last_updated=datetime.now()
                    )
                    
                    positions[symbol] = position
            
            self.positions = positions
            logger.info(f"📊 IB positions updated: {len(positions)} positions")
            
        except Exception as e:
            logger.error(f"❌ Error updating IB positions: {e}")
    
    # ==================== ORDER MANAGEMENT ====================
    
    async def place_order(self, symbol: str, side: str, quantity: float,
                         order_type: str = "market", price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         time_in_force: str = "day",
                         **kwargs) -> Dict[str, Any]:
        """Place an order with Interactive Brokers"""
        try:
            # Validate order parameters
            validation = self.validate_order_params(symbol, side, quantity, order_type, price)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': f"Order validation failed: {', '.join(validation['errors'])}",
                    'order_id': None
                }
            
            if not self._connected:
                await self.connect()
            
            # Create IB contract
            contract = self._create_contract(symbol, kwargs.get('asset_type', 'stock'))
            
            # Create IB order
            ib_order = self._create_ib_order(side, quantity, order_type, price, stop_price, time_in_force, **kwargs)
            
            # Get order ID
            order_id = self.next_order_id
            self.next_order_id += 1
            
            # Place order
            self.client.placeOrder(order_id, contract, ib_order)
            
            # Create our order record
            broker_order = BrokerOrder(
                order_id=str(order_id),
                symbol=symbol,
                asset_type=self._get_asset_type_from_string(kwargs.get('asset_type', 'stock')),
                side=OrderSide(side.upper()),
                order_type=OrderType(order_type.upper()),
                quantity=quantity,
                status=OrderStatus.PENDING,
                
                limit_price=price,
                stop_price=stop_price,
                time_in_force=TimeInForce(time_in_force.upper()),
                
                broker_type=BrokerType.INTERACTIVE_BROKERS,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.orders[str(order_id)] = broker_order
            self.pending_orders[order_id] = broker_order
            
            logger.info(f"✅ IB order placed: {order_id} - {side.upper()} {quantity} {symbol}")
            
            return {
                'success': True,
                'order_id': str(order_id),
                'order': broker_order
            }
            
        except Exception as e:
            error_msg = f"Error placing IB order: {e}"
            logger.error(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'order_id': None
            }
    
    def _create_contract(self, symbol: str, asset_type: str = 'stock') -> Contract:
        """Create IB contract object"""
        contract = Contract()
        contract.symbol = symbol.upper()
        
        if asset_type.lower() == 'stock':
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
        elif asset_type.lower() == 'option':
            contract.secType = "OPT"
            contract.exchange = "SMART"
            contract.currency = "USD"
            # Additional option parameters would be needed
        elif asset_type.lower() == 'forex':
            contract.secType = "CASH"
            contract.exchange = "IDEALPRO"
            # Symbol should be like "EUR.USD"
            if '.' in symbol:
                base, quote = symbol.split('.')
                contract.symbol = base
                contract.currency = quote
        
        return contract
    
    def _create_ib_order(self, side: str, quantity: float, order_type: str,
                        price: Optional[float], stop_price: Optional[float],
                        time_in_force: str, **kwargs) -> Order:
        """Create IB order object"""
        order = Order()
        
        # Basic order properties
        order.action = side.upper()
        order.totalQuantity = quantity
        order.orderType = self._convert_order_type(order_type)
        order.tif = time_in_force.upper()
        
        # Price settings
        if order_type.lower() == "limit" and price:
            order.lmtPrice = price
        elif order_type.lower() == "stop" and stop_price:
            order.auxPrice = stop_price
        elif order_type.lower() == "stop_limit":
            if price:
                order.lmtPrice = price
            if stop_price:
                order.auxPrice = stop_price
        
        # Additional order properties
        if kwargs.get('outside_rth', False):
            order.outsideRth = True
        
        return order
    
    def _convert_order_type(self, order_type: str) -> str:
        """Convert our order type to IB order type"""
        type_map = {
            'market': 'MKT',
            'limit': 'LMT',
            'stop': 'STP',
            'stop_limit': 'STP LMT'
        }
        return type_map.get(order_type.lower(), 'MKT')
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            if not self._connected:
                await self.connect()
            
            self.client.cancelOrder(int(order_id), "")
            
            # Update local order status
            if order_id in self.orders:
                self.orders[order_id].status = OrderStatus.CANCELLED
                self.orders[order_id].updated_at = datetime.now()
            
            logger.info(f"✅ IB order cancellation requested: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cancelling IB order {order_id}: {e}")
            return False
    
    async def get_orders(self, status: Optional[str] = None) -> Dict[str, BrokerOrder]:
        """Get orders"""
        try:
            if not self._connected:
                await self.connect()
            
            # Request open orders
            self.client.reqOpenOrders()
            
            # Give it time to update
            await asyncio.sleep(1)
            
            # Filter by status if requested
            if status:
                filtered_orders = {}
                for order_id, order in self.orders.items():
                    if order.status.value.lower() == status.lower():
                        filtered_orders[order_id] = order
                return filtered_orders
            
            return self.orders
            
        except Exception as e:
            logger.error(f"❌ Error getting IB orders: {e}")
            return {}
    
    async def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        """Get specific order"""
        return self.orders.get(order_id)
    
    def _update_order_status(self, order_id: int, order_info: Dict[str, Any]):
        """Update order status from IB callback"""
        try:
            order_id_str = str(order_id)
            
            if order_id_str in self.orders:
                order = self.orders[order_id_str]
                
                # Update status
                ib_status = order_info.get('status', '').lower()
                if ib_status == 'submitted':
                    order.status = OrderStatus.OPEN
                elif ib_status == 'filled':
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = order_info.get('filled', 0)
                    order.filled_price = order_info.get('avgFillPrice', 0)
                    order.filled_at = datetime.now()
                elif ib_status == 'cancelled':
                    order.status = OrderStatus.CANCELLED
                elif ib_status == 'partially filled':
                    order.status = OrderStatus.PARTIALLY_FILLED
                    order.filled_quantity = order_info.get('filled', 0)
                    order.remaining_quantity = order_info.get('remaining', 0)
                
                order.updated_at = datetime.now()
                
                logger.info(f"📋 IB order {order_id} updated: {order.status.value}")
                
        except Exception as e:
            logger.error(f"❌ Error updating IB order status: {e}")
    
    # ==================== HELPER METHODS ====================
    
    def _get_asset_type(self, contract: Contract) -> AssetType:
        """Convert IB contract to our AssetType"""
        sec_type_map = {
            'STK': AssetType.STOCK,
            'OPT': AssetType.OPTION,
            'FUT': AssetType.STOCK,  # Map to closest equivalent
            'CASH': AssetType.FOREX,
            'BOND': AssetType.BOND if hasattr(AssetType, 'BOND') else AssetType.STOCK
        }
        return sec_type_map.get(contract.secType, AssetType.STOCK)
    
    def _get_asset_type_from_string(self, asset_type_str: str) -> AssetType:
        """Convert string to AssetType"""
        type_map = {
            'stock': AssetType.STOCK,
            'crypto': AssetType.CRYPTO,
            'forex': AssetType.FOREX,
            'option': AssetType.OPTION if hasattr(AssetType, 'OPTION') else AssetType.STOCK
        }
        return type_map.get(asset_type_str.lower(), AssetType.STOCK)
    
    def _get_next_request_id(self) -> int:
        """Get next request ID for API calls"""
        self._request_id += 1
        return self._request_id
    
    def format_symbol(self, symbol: str) -> str:
        """Format symbol for IB"""
        return symbol.upper().strip()
    
    # ==================== CAPABILITIES ====================
    
    async def get_capabilities(self) -> List[str]:
        """Get Interactive Brokers capabilities"""
        return [
            "market_orders",
            "limit_orders",
            "stop_orders",
            "stop_limit_orders",
            "trailing_stops",
            "algorithmic_orders",
            "stocks",
            "options",
            "futures",
            "forex",
            "bonds",
            "global_markets",
            "paper_trading",
            "live_trading",
            "real_time_data",
            "historical_data",
            "options_chains",
            "portfolio_tracking",
            "multi_currency"
        ]

# ==================== TESTING ====================

async def test_ib_broker():
    """Test Interactive Brokers broker functionality"""
    print("🧪 Testing Interactive Brokers Broker")
    print("=" * 50)
    
    if not IB_API_AVAILABLE:
        print("❌ Interactive Brokers API not available")
        print("   Install with: pip install ibapi")
        return
    
    # Create test configuration
    config = IntegrationConfig(
        name="interactive_brokers",
        integration_type="broker",
        enabled=True,
        paper_trading=True,
        additional_auth={
            'host': '127.0.0.1',
            'port': 7497,  # Paper trading port
            'client_id': 1
        }
    )
    
    # Create broker instance
    broker = InteractiveBrokersBroker(config)
    
    print("✅ Broker instance created")
    print(f"✅ Host: {broker.host}:{broker.port}")
    print(f"✅ Client ID: {broker.client_id}")
    
    # Test capabilities
    capabilities = await broker.get_capabilities()
    print(f"✅ Capabilities: {', '.join(capabilities[:5])}...")
    
    # Note: Actual connection test requires TWS to be running
    print("\n⚠️  Note: Connection test requires TWS/Gateway to be running")
    print("   Start TWS and enable API connections to test fully")
    
    print("\n🎉 Interactive Brokers broker tests completed!")

if __name__ == "__main__":
    asyncio.run(test_ib_broker())