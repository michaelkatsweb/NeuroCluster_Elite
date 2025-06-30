#!/usr/bin/env python3
"""
File: kraken.py
Path: NeuroCluster-Elite/src/integrations/exchanges/kraken.py
Description: Kraken exchange integration for NeuroCluster Elite

This module implements the Kraken API integration, providing access to one of the
most established and secure cryptocurrency exchanges with advanced trading features.

Features:
- Kraken REST API v2 integration
- Kraken WebSocket API support
- Spot and margin trading
- Advanced order types
- Fiat and crypto deposits/withdrawals
- Institutional-grade security
- European regulatory compliance
- Staking and earning features

API Documentation: https://docs.kraken.com/rest/

Author: Your Name
Created: 2025-06-30
Version: 1.0.0
License: MIT
"""

import asyncio
import aiohttp
import websockets
import json
import hmac
import hashlib
import time
import base64
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from urllib.parse import urlencode
import uuid

# Import our modules
try:
    from src.integrations.exchanges import (
        BaseExchange, ExchangeBalance, ExchangePosition, ExchangeOrder,
        ExchangeTicker, ExchangeOrderBook, ExchangeType, OrderType, 
        OrderSide, OrderStatus, TradingMode
    )
    from src.integrations import IntegrationConfig, IntegrationStatus
    from src.core.neurocluster_elite import AssetType
    from src.utils.helpers import format_currency, format_percentage
except ImportError:
    # Fallback for testing
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== KRAKEN SPECIFIC ENUMS ====================

class KrakenOrderType(Enum):
    """Kraken order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop-loss"
    TAKE_PROFIT = "take-profit"
    STOP_LOSS_LIMIT = "stop-loss-limit"
    TAKE_PROFIT_LIMIT = "take-profit-limit"
    SETTLE_POSITION = "settle-position"

class KrakenOrderSide(Enum):
    """Kraken order sides"""
    BUY = "buy"
    SELL = "sell"

class KrakenTimeInForce(Enum):
    """Kraken time in force options"""
    GTC = "GTC"  # Good Till Canceled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill

class KrakenOrderStatus(Enum):
    """Kraken order status"""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    EXPIRED = "expired"

# ==================== KRAKEN CONFIG ====================

@dataclass
class KrakenConfig:
    """Kraken-specific configuration"""
    api_key: str
    private_key: str
    
    # URLs
    base_url: str = "https://api.kraken.com"
    websocket_url: str = "wss://ws.kraken.com"
    auth_websocket_url: str = "wss://ws-auth.kraken.com"
    
    # API version
    api_version: str = "0"
    
    # Trading settings
    use_sandbox: bool = False  # Kraken doesn't have sandbox
    
    # Features
    enable_margin: bool = True
    enable_spot: bool = True
    enable_futures: bool = False  # Separate Kraken Futures API
    
    # Risk settings
    max_order_value: float = 50000.0
    
    # Rate limiting
    api_counter_decay: int = 2  # seconds
    api_counter_max: int = 15

# ==================== KRAKEN EXCHANGE IMPLEMENTATION ====================

class KrakenExchange(BaseExchange):
    """
    Kraken exchange implementation
    
    Provides integration with Kraken's cryptocurrency exchange platform
    with emphasis on security, compliance, and advanced trading features.
    """
    
    def __init__(self, config: IntegrationConfig):
        """Initialize Kraken exchange"""
        super().__init__(config)
        
        # Extract Kraken-specific config
        self.api_key = config.api_key or config.additional_auth.get('api_key', '')
        self.private_key = config.secret_key or config.additional_auth.get('private_key', '')
        
        # URLs
        self.base_url = "https://api.kraken.com"
        self.websocket_url = "wss://ws.kraken.com"
        self.auth_websocket_url = "wss://ws-auth.kraken.com"
        
        # API version
        self.api_version = "0"
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # WebSocket connections
        self.public_ws: Optional[websockets.WebSocketServerProtocol] = None
        self.private_ws: Optional[websockets.WebSocketServerProtocol] = None
        
        # Trading features
        self.supports_spot = True
        self.supports_margin = True
        self.supports_futures = False  # Would require separate Kraken Futures integration
        self.supports_websocket = True
        
        # Rate limiting
        self.api_counter = 0
        self.last_api_call = datetime.now()
        
        # Market data cache
        self.asset_pairs: Dict[str, Any] = {}
        self.assets: Dict[str, Any] = {}
        
        # Symbol mapping (Kraken uses unique symbol names)
        self.symbol_map: Dict[str, str] = {}  # Our format -> Kraken format
        self.reverse_symbol_map: Dict[str, str] = {}  # Kraken format -> Our format
        
        logger.info("🐙 Kraken exchange initialized")
    
    # ==================== CONNECTION MANAGEMENT ====================
    
    async def connect(self) -> bool:
        """Connect to Kraken API"""
        try:
            # Validate credentials
            if not self.api_key or not self.private_key:
                error_msg = "Kraken API credentials not provided"
                self.update_status(IntegrationStatus.ERROR, error_msg)
                logger.error(f"❌ {error_msg}")
                return False
            
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Test connection and load markets
            if await self._test_connectivity():
                # Load market data
                await self.load_markets()
                
                # Start WebSocket connections
                await self._start_websockets()
                
                self.update_status(IntegrationStatus.CONNECTED)
                logger.info("✅ Kraken exchange connected successfully")
                return True
            else:
                error_msg = "Failed to connect to Kraken API"
                self.update_status(IntegrationStatus.ERROR, error_msg)
                logger.error(f"❌ {error_msg}")
                return False
                
        except Exception as e:
            error_msg = f"Kraken connection failed: {e}"
            self.update_status(IntegrationStatus.ERROR, error_msg)
            logger.error(f"❌ {error_msg}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Kraken API"""
        try:
            # Close WebSocket connections
            if self.public_ws and not self.public_ws.closed:
                await self.public_ws.close()
                self.public_ws = None
            
            if self.private_ws and not self.private_ws.closed:
                await self.private_ws.close()
                self.private_ws = None
            
            # Close HTTP session
            if self.session:
                await self.session.close()
                self.session = None
            
            self.update_status(IntegrationStatus.DISCONNECTED)
            logger.info("✅ Kraken exchange disconnected")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error disconnecting Kraken: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test Kraken connection"""
        return await self._test_connectivity()
    
    async def _test_connectivity(self) -> bool:
        """Test API connectivity"""
        try:
            if not self.session:
                return False
            
            # Test with server time endpoint
            async with self.session.get(f"{self.base_url}/{self.api_version}/public/Time") as response:
                if response.status == 200:
                    data = await response.json()
                    return 'error' not in data or len(data['error']) == 0
                return False
                
        except Exception as e:
            logger.error(f"❌ Kraken connectivity test failed: {e}")
            return False
    
    # ==================== AUTHENTICATION ====================
    
    def _create_signature(self, uri_path: str, data: Dict[str, Any], nonce: str) -> str:
        """Create authentication signature for Kraken API"""
        # Create the message to sign
        postdata = urlencode(data)
        encoded = (nonce + postdata).encode('utf-8')
        message = uri_path.encode('utf-8') + hashlib.sha256(encoded).digest()
        
        # Sign with private key
        signature = hmac.new(
            base64.b64decode(self.private_key),
            message,
            hashlib.sha512
        )
        
        return base64.b64encode(signature.digest()).decode('utf-8')
    
    async def _make_private_request(self, endpoint: str, data: Optional[Dict] = None) -> Optional[Dict]:
        """Make authenticated API request"""
        try:
            if not self.session:
                await self.connect()
            
            # Check rate limiting
            if not self._check_rate_limit():
                await asyncio.sleep(1)  # Wait before retrying
            
            # Prepare request data
            if data is None:
                data = {}
            
            # Add nonce
            nonce = str(int(time.time() * 1000000))
            data['nonce'] = nonce
            
            # Create signature
            uri_path = f"/{self.api_version}/private/{endpoint}"
            signature = self._create_signature(uri_path, data, nonce)
            
            # Set headers
            headers = {
                'API-Key': self.api_key,
                'API-Sign': signature,
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            # Make request
            url = f"{self.base_url}{uri_path}"
            async with self.session.post(url, data=data, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Check for API errors
                    if 'error' in result and result['error']:
                        logger.error(f"❌ Kraken API error: {result['error']}")
                        return None
                    
                    return result.get('result')
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Kraken API error {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Kraken private API request failed: {e}")
            return None
    
    async def _make_public_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make public API request"""
        try:
            if not self.session:
                await self.connect()
            
            url = f"{self.base_url}/{self.api_version}/public/{endpoint}"
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Check for API errors
                    if 'error' in result and result['error']:
                        logger.error(f"❌ Kraken API error: {result['error']}")
                        return None
                    
                    return result.get('result')
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Kraken API error {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Kraken public API request failed: {e}")
            return None
    
    def _check_rate_limit(self) -> bool:
        """Check API rate limiting"""
        now = datetime.now()
        time_diff = (now - self.last_api_call).total_seconds()
        
        # Decay counter based on time
        if time_diff >= 2:  # Kraken's API counter decays every 2 seconds
            decay_amount = int(time_diff / 2)
            self.api_counter = max(0, self.api_counter - decay_amount)
        
        # Check if we can make another request
        if self.api_counter >= 15:  # Kraken's default tier limit
            return False
        
        # Increment counter and update timestamp
        self.api_counter += 1
        self.last_api_call = now
        return True
    
    # ==================== MARKET DATA ====================
    
    async def load_markets(self) -> Dict[str, Any]:
        """Load available markets and trading pairs"""
        try:
            # Get asset pairs
            pairs_data = await self._make_public_request('AssetPairs')
            
            if pairs_data:
                self.asset_pairs = pairs_data
                
                # Build symbol mappings
                for kraken_symbol, pair_info in pairs_data.items():
                    # Skip dark pool pairs
                    if '.d' in kraken_symbol:
                        continue
                    
                    # Get base and quote from pair info
                    base = pair_info.get('base', '')
                    quote = pair_info.get('quote', '')
                    
                    # Create standard symbol format
                    standard_symbol = f"{base}/{quote}"
                    
                    # Map both ways
                    self.symbol_map[standard_symbol] = kraken_symbol
                    self.reverse_symbol_map[kraken_symbol] = standard_symbol
                
                # Get assets
                assets_data = await self._make_public_request('Assets')
                if assets_data:
                    self.assets = assets_data
                
                self.symbols = list(self.symbol_map.keys())
                self.markets = self.asset_pairs
                
                logger.info(f"📊 Loaded {len(self.symbols)} Kraken trading pairs")
                return self.markets
            else:
                logger.error("❌ Failed to load Kraken markets")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Error loading Kraken markets: {e}")
            return {}
    
    async def fetch_ticker(self, symbol: str) -> ExchangeTicker:
        """Fetch ticker data for a symbol"""
        try:
            kraken_symbol = self._to_kraken_symbol(symbol)
            if not kraken_symbol:
                raise ValueError(f"Symbol {symbol} not found on Kraken")
            
            # Get ticker data
            response = await self._make_public_request('Ticker', {'pair': kraken_symbol})
            
            if response and kraken_symbol in response:
                ticker_data = response[kraken_symbol]
                
                # Parse ticker data
                last_price = float(ticker_data['c'][0])
                bid_price = float(ticker_data['b'][0])
                ask_price = float(ticker_data['a'][0])
                volume = float(ticker_data['v'][1])  # 24h volume
                high = float(ticker_data['h'][1])    # 24h high
                low = float(ticker_data['l'][1])     # 24h low
                open_price = float(ticker_data['o'])
                
                # Calculate change
                change = last_price - open_price
                change_percent = (change / open_price) * 100 if open_price > 0 else 0
                
                ticker = ExchangeTicker(
                    symbol=symbol,
                    last=last_price,
                    bid=bid_price,
                    ask=ask_price,
                    volume=volume,
                    change=change,
                    change_percent=change_percent,
                    
                    high=high,
                    low=low,
                    open=open_price,
                    close=last_price,
                    
                    base_volume=volume,
                    quote_volume=0.0,  # Not provided separately
                    
                    exchange="kraken",
                    timestamp=datetime.now()
                )
                
                self.tickers[symbol] = ticker
                return ticker
            else:
                raise Exception("No ticker data received")
                
        except Exception as e:
            logger.error(f"❌ Error fetching Kraken ticker for {symbol}: {e}")
            raise
    
    async def fetch_orderbook(self, symbol: str, limit: int = 100) -> ExchangeOrderBook:
        """Fetch orderbook data for a symbol"""
        try:
            kraken_symbol = self._to_kraken_symbol(symbol)
            if not kraken_symbol:
                raise ValueError(f"Symbol {symbol} not found on Kraken")
            
            # Get order book (Kraken supports up to 500 depth)
            count = min(limit, 500)
            response = await self._make_public_request('Depth', {
                'pair': kraken_symbol,
                'count': count
            })
            
            if response and kraken_symbol in response:
                orderbook_data = response[kraken_symbol]
                
                # Convert to our format
                bids = [(float(price), float(amount)) for price, amount, _ in orderbook_data['bids']]
                asks = [(float(price), float(amount)) for price, amount, _ in orderbook_data['asks']]
                
                orderbook = ExchangeOrderBook(
                    symbol=symbol,
                    bids=bids,
                    asks=asks,
                    timestamp=datetime.now()
                )
                
                self.orderbooks[symbol] = orderbook
                return orderbook
            else:
                raise Exception("No orderbook data received")
                
        except Exception as e:
            logger.error(f"❌ Error fetching Kraken orderbook for {symbol}: {e}")
            raise
    
    # ==================== ACCOUNT MANAGEMENT ====================
    
    async def fetch_balance(self) -> Dict[str, ExchangeBalance]:
        """Fetch account balance"""
        try:
            response = await self._make_private_request('Balance')
            
            if response:
                balances = {}
                
                for asset, balance_str in response.items():
                    balance = float(balance_str)
                    
                    if balance > 0:  # Only include non-zero balances
                        # Clean up asset name (remove X prefix for most assets)
                        clean_asset = asset
                        if asset.startswith('X') and len(asset) == 4:
                            clean_asset = asset[1:]
                        elif asset.startswith('Z') and len(asset) == 4:
                            clean_asset = asset[1:]
                        
                        balances[clean_asset] = ExchangeBalance(
                            currency=clean_asset,
                            total=balance,
                            used=0.0,  # Would need to fetch open orders to calculate
                            free=balance,
                            last_updated=datetime.now()
                        )
                
                # Get open orders to calculate used balance
                await self._update_used_balances(balances)
                
                self.balances = balances
                return balances
            else:
                logger.error("❌ Failed to fetch Kraken balance")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Error fetching Kraken balance: {e}")
            return {}
    
    async def _update_used_balances(self, balances: Dict[str, ExchangeBalance]):
        """Update used balances based on open orders"""
        try:
            # Get open orders
            orders_response = await self._make_private_request('OpenOrders')
            
            if orders_response and 'open' in orders_response:
                for order_id, order_data in orders_response['open'].items():
                    # Calculate reserved amount for this order
                    desc = order_data['descr']
                    pair = desc['pair']
                    order_type = desc['type']
                    volume = float(order_data['vol'])
                    price = float(desc.get('price', 0))
                    
                    # Determine which currency is being used
                    if pair in self.reverse_symbol_map:
                        symbol = self.reverse_symbol_map[pair]
                        base, quote = symbol.split('/')
                        
                        if order_type == 'buy':
                            # Buying base with quote currency
                            used_currency = quote
                            used_amount = volume * price if price > 0 else 0
                        else:
                            # Selling base currency
                            used_currency = base
                            used_amount = volume
                        
                        # Update balance
                        if used_currency in balances:
                            balances[used_currency].used += used_amount
                            balances[used_currency].free = max(0, balances[used_currency].total - balances[used_currency].used)
                            
        except Exception as e:
            logger.error(f"❌ Error updating used balances: {e}")
    
    async def fetch_positions(self) -> Dict[str, ExchangePosition]:
        """Fetch account positions (margin positions)"""
        try:
            # Get open positions (margin trading)
            response = await self._make_private_request('OpenPositions')
            
            positions = {}
            if response:
                for position_id, position_data in response.items():
                    # Parse position data
                    pair = position_data['pair']
                    
                    if pair in self.reverse_symbol_map:
                        symbol = self.reverse_symbol_map[pair]
                        
                        position = ExchangePosition(
                            symbol=symbol,
                            side='long' if position_data['type'] == 'buy' else 'short',
                            size=float(position_data['vol']),
                            notional=float(position_data['cost']),
                            entry_price=float(position_data['cost']) / float(position_data['vol']),
                            mark_price=0.0,  # Would need current market price
                            unrealized_pnl=float(position_data.get('net', 0)),
                            realized_pnl=0.0,  # Not provided
                            
                            leverage=float(position_data.get('margin', 1)),
                            
                            exchange="kraken",
                            last_updated=datetime.now()
                        )
                        
                        positions[symbol] = position
                
                self.positions = positions
                return positions
            else:
                return {}
                
        except Exception as e:
            logger.error(f"❌ Error fetching Kraken positions: {e}")
            return {}
    
    # ==================== ORDER MANAGEMENT ====================
    
    async def create_order(self, symbol: str, side: str, amount: float,
                          order_type: str = "market", price: Optional[float] = None,
                          **kwargs) -> ExchangeOrder:
        """Create an order on Kraken"""
        try:
            kraken_symbol = self._to_kraken_symbol(symbol)
            if not kraken_symbol:
                raise ValueError(f"Symbol {symbol} not found on Kraken")
            
            # Prepare order data
            order_data = {
                'pair': kraken_symbol,
                'type': side.lower(),
                'ordertype': self._convert_order_type(order_type),
                'volume': str(amount)
            }
            
            # Add price for limit orders
            if order_type.lower() in ['limit', 'stop-loss-limit', 'take-profit-limit']:
                if not price:
                    raise ValueError(f"Price required for {order_type} orders")
                order_data['price'] = str(price)
            
            # Add stop price for stop orders
            if order_type.lower() in ['stop-loss', 'stop-loss-limit', 'take-profit', 'take-profit-limit']:
                stop_price = kwargs.get('stop_price') or price
                if stop_price:
                    order_data['price2'] = str(stop_price)
            
            # Add additional parameters
            if kwargs.get('leverage'):
                order_data['leverage'] = str(kwargs['leverage'])
            
            if kwargs.get('validate_only'):
                order_data['validate'] = 'true'
            
            # Submit order
            response = await self._make_private_request('AddOrder', order_data)
            
            if response and 'txid' in response:
                # Get order ID(s)
                order_ids = response['txid']
                order_id = order_ids[0] if order_ids else str(uuid.uuid4())
                
                # Create order object
                order = ExchangeOrder(
                    id=order_id,
                    symbol=symbol,
                    side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
                    type=self._parse_order_type(order_data['ordertype']),
                    amount=amount,
                    price=price,
                    status=OrderStatus.PENDING,
                    
                    exchange="kraken",
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    info=response
                )
                
                self.orders[order_id] = order
                
                logger.info(f"✅ Kraken order created: {order_id} - {side.upper()} {amount} {symbol}")
                return order
            else:
                error_msg = "Failed to create order - no transaction ID received"
                logger.error(f"❌ Kraken order failed: {error_msg}")
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"❌ Error creating Kraken order: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        try:
            response = await self._make_private_request('CancelOrder', {'txid': order_id})
            
            if response and 'count' in response:
                count = response['count']
                if count > 0:
                    # Update local order if exists
                    if order_id in self.orders:
                        self.orders[order_id].status = OrderStatus.CANCELED
                        self.orders[order_id].updated_at = datetime.now()
                    
                    logger.info(f"✅ Kraken order cancelled: {order_id}")
                    return True
                else:
                    logger.error(f"❌ Kraken cancel order failed: No orders cancelled")
                    return False
            else:
                logger.error("❌ Kraken cancel order failed: Invalid response")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error cancelling Kraken order {order_id}: {e}")
            return False
    
    async def fetch_order(self, order_id: str, symbol: str) -> Optional[ExchangeOrder]:
        """Fetch order status"""
        try:
            response = await self._make_private_request('QueryOrders', {'txid': order_id})
            
            if response and order_id in response:
                order_data = response[order_id]
                order = self._parse_order(order_id, order_data)
                self.orders[order_id] = order
                return order
            else:
                logger.error(f"❌ Failed to fetch Kraken order {order_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error fetching Kraken order {order_id}: {e}")
            return None
    
    async def fetch_orders(self, symbol: Optional[str] = None,
                          status: Optional[str] = None) -> List[ExchangeOrder]:
        """Fetch orders"""
        try:
            orders = []
            
            # Get open orders
            open_response = await self._make_private_request('OpenOrders')
            if open_response and 'open' in open_response:
                for order_id, order_data in open_response['open'].items():
                    order = self._parse_order(order_id, order_data)
                    
                    # Filter by symbol if specified
                    if symbol and order.symbol != symbol:
                        continue
                    
                    # Filter by status if specified
                    if status and order.status.value.lower() != status.lower():
                        continue
                    
                    orders.append(order)
                    self.orders[order_id] = order
            
            # Get closed orders (recent)
            closed_response = await self._make_private_request('ClosedOrders')
            if closed_response and 'closed' in closed_response:
                for order_id, order_data in closed_response['closed'].items():
                    order = self._parse_order(order_id, order_data)
                    
                    # Filter by symbol if specified
                    if symbol and order.symbol != symbol:
                        continue
                    
                    # Filter by status if specified
                    if status and order.status.value.lower() != status.lower():
                        continue
                    
                    orders.append(order)
                    self.orders[order_id] = order
            
            return orders
            
        except Exception as e:
            logger.error(f"❌ Error fetching Kraken orders: {e}")
            return []
    
    # ==================== HELPER METHODS ====================
    
    def _to_kraken_symbol(self, symbol: str) -> Optional[str]:
        """Convert standard symbol to Kraken symbol"""
        return self.symbol_map.get(symbol)
    
    def _from_kraken_symbol(self, kraken_symbol: str) -> str:
        """Convert Kraken symbol to standard symbol"""
        return self.reverse_symbol_map.get(kraken_symbol, kraken_symbol)
    
    def _convert_order_type(self, order_type: str) -> str:
        """Convert our order type to Kraken order type"""
        type_map = {
            'market': 'market',
            'limit': 'limit',
            'stop_loss': 'stop-loss',
            'stop_loss_limit': 'stop-loss-limit',
            'take_profit': 'take-profit',
            'take_profit_limit': 'take-profit-limit'
        }
        return type_map.get(order_type.lower(), 'market')
    
    def _parse_order_type(self, kraken_type: str) -> OrderType:
        """Convert Kraken order type to our OrderType"""
        type_map = {
            'market': OrderType.MARKET,
            'limit': OrderType.LIMIT,
            'stop-loss': OrderType.STOP_LOSS,
            'stop-loss-limit': OrderType.STOP_LOSS_LIMIT,
            'take-profit': OrderType.TAKE_PROFIT,
            'take-profit-limit': OrderType.TAKE_PROFIT_LIMIT
        }
        return type_map.get(kraken_type, OrderType.MARKET)
    
    def _parse_order_status(self, kraken_status: str) -> OrderStatus:
        """Convert Kraken order status to our OrderStatus"""
        status_map = {
            'pending': OrderStatus.PENDING,
            'open': OrderStatus.OPEN,
            'closed': OrderStatus.CLOSED,
            'canceled': OrderStatus.CANCELED,
            'expired': OrderStatus.EXPIRED
        }
        return status_map.get(kraken_status, OrderStatus.PENDING)
    
    def _parse_order(self, order_id: str, order_data: Dict[str, Any]) -> ExchangeOrder:
        """Parse Kraken order data into our format"""
        
        # Extract order description
        desc = order_data.get('descr', {})
        pair = desc.get('pair', '')
        symbol = self._from_kraken_symbol(pair)
        
        return ExchangeOrder(
            id=order_id,
            symbol=symbol,
            side=OrderSide.BUY if desc.get('type') == 'buy' else OrderSide.SELL,
            type=self._parse_order_type(desc.get('ordertype', 'market')),
            amount=float(order_data.get('vol', 0)),
            price=float(desc.get('price', 0)) if desc.get('price') else None,
            status=self._parse_order_status(order_data.get('status', 'pending')),
            
            filled=float(order_data.get('vol_exec', 0)),
            remaining=float(order_data.get('vol', 0)) - float(order_data.get('vol_exec', 0)),
            cost=float(order_data.get('cost', 0)),
            average=float(order_data.get('price', 0)) if order_data.get('price') else None,
            
            timestamp=datetime.fromtimestamp(float(order_data.get('opentm', time.time()))),
            created_at=datetime.fromtimestamp(float(order_data.get('opentm', time.time()))),
            updated_at=datetime.now(),
            
            exchange="kraken",
            info=order_data
        )
    
    def format_symbol(self, symbol: str) -> str:
        """Format symbol for Kraken (return as-is, mapping handled internally)"""
        # Kraken symbols are handled via internal mapping
        return symbol.upper().replace('-', '/').replace('_', '/')
    
    # ==================== WEBSOCKET IMPLEMENTATION ====================
    
    async def _start_websockets(self):
        """Start WebSocket connections"""
        try:
            # Start public WebSocket for market data
            asyncio.create_task(self._handle_public_websocket())
            
            # Start private WebSocket for account updates
            asyncio.create_task(self._handle_private_websocket())
            
            logger.info("📡 Kraken WebSocket connections started")
            
        except Exception as e:
            logger.error(f"❌ Error starting Kraken WebSockets: {e}")
    
    async def _handle_public_websocket(self):
        """Handle public WebSocket for market data"""
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                self.public_ws = websocket
                
                # Subscribe to ticker for a few symbols
                top_symbols = list(self.symbol_map.keys())[:5]  # Limit to top 5
                kraken_symbols = [self.symbol_map[s] for s in top_symbols if s in self.symbol_map]
                
                if kraken_symbols:
                    subscribe_msg = {
                        "event": "subscribe",
                        "pair": kraken_symbols,
                        "subscription": {"name": "ticker"}
                    }
                    
                    await websocket.send(json.dumps(subscribe_msg))
                    logger.info(f"📡 Subscribed to Kraken ticker for {len(kraken_symbols)} symbols")
                
                # Listen for messages
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self._process_public_message(data)
                    except Exception as e:
                        logger.error(f"❌ Error processing public WebSocket message: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Kraken public WebSocket error: {e}")
    
    async def _handle_private_websocket(self):
        """Handle private WebSocket for account updates"""
        try:
            # Get WebSocket token
            token_response = await self._make_private_request('GetWebSocketsToken')
            if not token_response or 'token' not in token_response:
                logger.warning("⚠️ Could not get WebSocket token for private feeds")
                return
            
            token = token_response['token']
            
            async with websockets.connect(self.auth_websocket_url) as websocket:
                self.private_ws = websocket
                
                # Subscribe to own trades
                subscribe_msg = {
                    "event": "subscribe",
                    "subscription": {
                        "name": "ownTrades",
                        "token": token
                    }
                }
                
                await websocket.send(json.dumps(subscribe_msg))
                logger.info("📡 Subscribed to Kraken private feeds")
                
                # Listen for messages
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self._process_private_message(data)
                    except Exception as e:
                        logger.error(f"❌ Error processing private WebSocket message: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Kraken private WebSocket error: {e}")
    
    async def _process_public_message(self, data: Any):
        """Process public WebSocket messages"""
        try:
            if isinstance(data, list) and len(data) >= 4:
                # Ticker update format: [channelID, data, channelName, pair]
                channel_name = data[2] if len(data) > 2 else ""
                
                if channel_name == "ticker":
                    pair = data[3] if len(data) > 3 else ""
                    ticker_data = data[1] if len(data) > 1 else {}
                    
                    if pair in self.reverse_symbol_map:
                        symbol = self.reverse_symbol_map[pair]
                        
                        # Update ticker
                        if isinstance(ticker_data, dict):
                            ticker = ExchangeTicker(
                                symbol=symbol,
                                last=float(ticker_data.get('c', [0, 0])[0]),
                                bid=float(ticker_data.get('b', [0, 0])[0]),
                                ask=float(ticker_data.get('a', [0, 0])[0]),
                                volume=float(ticker_data.get('v', [0, 0])[1]),  # 24h volume
                                change=0.0,  # Calculate from previous
                                change_percent=0.0,
                                
                                high=float(ticker_data.get('h', [0, 0])[1]),
                                low=float(ticker_data.get('l', [0, 0])[1]),
                                open=float(ticker_data.get('o', [0, 0])[0]),
                                
                                exchange="kraken",
                                timestamp=datetime.now()
                            )
                            
                            self.tickers[symbol] = ticker
                            
        except Exception as e:
            logger.error(f"❌ Error processing public message: {e}")
    
    async def _process_private_message(self, data: Any):
        """Process private WebSocket messages"""
        try:
            if isinstance(data, list) and len(data) >= 3:
                channel_name = data[2] if len(data) > 2 else ""
                
                if channel_name == "ownTrades":
                    trades_data = data[0] if len(data) > 0 else []
                    
                    for trade_id, trade_info in trades_data.items():
                        # Process trade execution
                        logger.info(f"📋 Trade execution: {trade_id}")
                        
        except Exception as e:
            logger.error(f"❌ Error processing private message: {e}")
    
    # ==================== CAPABILITIES ====================
    
    async def get_capabilities(self) -> List[str]:
        """Get Kraken exchange capabilities"""
        return [
            "spot_trading",
            "margin_trading",
            "fiat_onramp",
            "fiat_offramp",
            "institutional_custody",
            "regulatory_compliance",
            "market_orders",
            "limit_orders",
            "stop_orders",
            "take_profit_orders",
            "advanced_order_types",
            "real_time_data",
            "websocket_streaming",
            "high_security",
            "staking",
            "lending",
            "futures_trading"  # Via separate Kraken Futures
        ]

# ==================== TESTING ====================

async def test_kraken_exchange():
    """Test Kraken exchange functionality"""
    print("🧪 Testing Kraken Exchange")
    print("=" * 40)
    
    # Create test configuration
    config = IntegrationConfig(
        name="kraken",
        integration_type="exchange",
        enabled=True,
        sandbox_mode=False,  # Kraken doesn't have sandbox
        api_key="test_key",
        secret_key="test_private_key"
    )
    
    # Create exchange instance
    exchange = KrakenExchange(config)
    
    print("✅ Exchange instance created")
    print(f"✅ Base URL: {exchange.base_url}")
    print(f"✅ WebSocket URL: {exchange.websocket_url}")
    
    # Test capabilities
    capabilities = await exchange.get_capabilities()
    print(f"✅ Capabilities: {', '.join(capabilities[:5])}...")
    
    # Test symbol formatting
    test_symbols = ['BTC/USD', 'ETH-EUR', 'ADA_USD']
    for symbol in test_symbols:
        formatted = exchange.format_symbol(symbol)
        print(f"✅ Symbol format: {symbol} -> {formatted}")
    
    print("\n⚠️  Note: Full testing requires valid API credentials")
    print("   Kraken doesn't provide sandbox - use small amounts for testing")
    
    print("\n🎉 Kraken exchange tests completed!")

if __name__ == "__main__":
    asyncio.run(test_kraken_exchange())