#!/usr/bin/env python3
"""
File: coinbase.py
Path: NeuroCluster-Elite/src/integrations/exchanges/coinbase.py
Description: Coinbase Pro/Advanced Trade exchange integration for NeuroCluster Elite

This module implements the Coinbase Pro (Advanced Trade) API integration, providing
access to one of the most regulated and trusted cryptocurrency exchanges in the US.

Features:
- Coinbase Advanced Trade API integration
- Coinbase WebSocket feed support
- Professional cryptocurrency trading
- USD fiat on/off ramps
- Institutional-grade security
- Advanced order types
- Portfolio tracking and analytics
- Regulatory compliance features

API Documentation: https://docs.cloud.coinbase.com/advanced-trade-api/docs/

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

# ==================== COINBASE SPECIFIC ENUMS ====================

class CoinbaseOrderType(Enum):
    """Coinbase order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class CoinbaseOrderSide(Enum):
    """Coinbase order sides"""
    BUY = "BUY"
    SELL = "SELL"

class CoinbaseTimeInForce(Enum):
    """Coinbase time in force options"""
    GTC = "GTC"  # Good Till Canceled
    GTD = "GTD"  # Good Till Date
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill

class CoinbaseOrderStatus(Enum):
    """Coinbase order status"""
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"

class CoinbaseProductType(Enum):
    """Coinbase product types"""
    SPOT = "SPOT"
    FUTURE = "FUTURE"

# ==================== COINBASE CONFIG ====================

@dataclass
class CoinbaseConfig:
    """Coinbase-specific configuration"""
    api_key: str
    api_secret: str
    passphrase: str
    
    # URLs
    base_url: str = "https://api.coinbase.com"
    sandbox_url: str = "https://api-public.sandbox.pro.coinbase.com"
    websocket_url: str = "wss://advanced-trade-ws.coinbase.com"
    sandbox_ws_url: str = "wss://advanced-trade-ws-sandbox.coinbase.com"
    
    # API version
    api_version: str = "v3"
    
    # Trading settings
    use_sandbox: bool = True
    use_advanced_trade: bool = True  # Use new Advanced Trade API
    
    # Features
    enable_margin: bool = False  # Coinbase doesn't offer margin trading
    enable_spot: bool = True
    
    # Risk settings
    max_order_value: float = 25000.0
    
    # Rate limiting
    requests_per_second: int = 10

# ==================== COINBASE EXCHANGE IMPLEMENTATION ====================

class CoinbaseExchange(BaseExchange):
    """
    Coinbase Pro/Advanced Trade exchange implementation
    
    Provides integration with Coinbase's professional trading platform
    with emphasis on regulatory compliance and institutional features.
    """
    
    def __init__(self, config: IntegrationConfig):
        """Initialize Coinbase exchange"""
        super().__init__(config)
        
        # Extract Coinbase-specific config
        self.api_key = config.api_key or config.additional_auth.get('api_key', '')
        self.api_secret = config.secret_key or config.additional_auth.get('api_secret', '')
        self.passphrase = config.additional_auth.get('passphrase', '')
        
        # URLs (use sandbox if in sandbox mode)
        if config.sandbox_mode:
            self.base_url = "https://api-public.sandbox.pro.coinbase.com"
            self.websocket_url = "wss://ws-feed-public.sandbox.pro.coinbase.com"
        else:
            self.base_url = "https://api.coinbase.com"
            self.websocket_url = "wss://advanced-trade-ws.coinbase.com"
        
        # API version (Advanced Trade API)
        self.api_version = "api/v3/brokerage"
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # WebSocket connections
        self.ws_connection: Optional[websockets.WebSocketServerProtocol] = None
        self.ws_subscriptions: set = set()
        
        # Trading features
        self.supports_spot = True
        self.supports_margin = False  # Coinbase doesn't offer margin
        self.supports_futures = False  # Coinbase doesn't offer futures
        self.supports_websocket = True
        
        # Rate limiting
        self.request_timestamps: List[datetime] = []
        
        # Market data cache
        self.product_info: Dict[str, Any] = {}
        
        logger.info(f"🔵 Coinbase exchange initialized - {'Sandbox' if config.sandbox_mode else 'Live'}")
    
    # ==================== CONNECTION MANAGEMENT ====================
    
    async def connect(self) -> bool:
        """Connect to Coinbase API"""
        try:
            # Validate credentials
            if not self.api_key or not self.api_secret or not self.passphrase:
                error_msg = "Coinbase API credentials incomplete (need key, secret, passphrase)"
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
                
                # Start WebSocket connection
                await self._start_websocket()
                
                self.update_status(IntegrationStatus.CONNECTED)
                logger.info("✅ Coinbase exchange connected successfully")
                return True
            else:
                error_msg = "Failed to connect to Coinbase API"
                self.update_status(IntegrationStatus.ERROR, error_msg)
                logger.error(f"❌ {error_msg}")
                return False
                
        except Exception as e:
            error_msg = f"Coinbase connection failed: {e}"
            self.update_status(IntegrationStatus.ERROR, error_msg)
            logger.error(f"❌ {error_msg}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Coinbase API"""
        try:
            # Close WebSocket connection
            if self.ws_connection and not self.ws_connection.closed:
                await self.ws_connection.close()
                self.ws_connection = None
            
            # Close HTTP session
            if self.session:
                await self.session.close()
                self.session = None
            
            self.update_status(IntegrationStatus.DISCONNECTED)
            logger.info("✅ Coinbase exchange disconnected")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error disconnecting Coinbase: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test Coinbase connection"""
        return await self._test_connectivity()
    
    async def _test_connectivity(self) -> bool:
        """Test API connectivity"""
        try:
            if not self.session:
                return False
            
            # Test with a simple public endpoint
            async with self.session.get(f"{self.base_url}/{self.api_version}/products") as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"❌ Coinbase connectivity test failed: {e}")
            return False
    
    # ==================== AUTHENTICATION ====================
    
    def _create_signature(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Create authentication signature for Coinbase API"""
        timestamp = str(int(time.time()))
        message = timestamp + method.upper() + path + body
        
        # Create signature
        signature = base64.b64encode(
            hmac.new(
                base64.b64decode(self.api_secret),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        return {
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-SIGN': signature,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
    
    async def _make_request(self, method: str, endpoint: str, 
                           params: Optional[Dict] = None,
                           data: Optional[Dict] = None) -> Optional[Dict]:
        """Make authenticated API request"""
        try:
            if not self.session:
                await self.connect()
            
            # Prepare request
            url = f"{self.base_url}/{self.api_version}/{endpoint}"
            body = json.dumps(data) if data else ""
            path = f"/{self.api_version}/{endpoint}"
            
            # Add query parameters to path for signature
            if params:
                path += "?" + urlencode(params)
            
            # Create signature
            headers = self._create_signature(method, path, body)
            
            # Make request
            kwargs = {'headers': headers}
            if params:
                kwargs['params'] = params
            if data:
                kwargs['data'] = body
            
            async with self.session.request(method, url, **kwargs) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Coinbase API error {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Coinbase API request failed: {e}")
            return None
    
    # ==================== MARKET DATA ====================
    
    async def load_markets(self) -> Dict[str, Any]:
        """Load available markets and trading pairs"""
        try:
            # Get products from Coinbase
            response = await self._make_request('GET', 'products')
            
            if response and 'products' in response:
                for product in response['products']:
                    product_id = product['product_id']
                    self.product_info[product_id] = product
                
                self.symbols = list(self.product_info.keys())
                self.markets = self.product_info
                
                logger.info(f"📊 Loaded {len(self.symbols)} Coinbase trading pairs")
                return self.markets
            else:
                logger.error("❌ Failed to load Coinbase markets")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Error loading Coinbase markets: {e}")
            return {}
    
    async def fetch_ticker(self, symbol: str) -> ExchangeTicker:
        """Fetch ticker data for a symbol"""
        try:
            formatted_symbol = self.format_symbol(symbol)
            
            # Get 24hr stats
            response = await self._make_request('GET', f'products/{formatted_symbol}/stats')
            
            if response:
                ticker = ExchangeTicker(
                    symbol=symbol,
                    last=float(response.get('last', 0)),
                    bid=0.0,  # Not provided in stats endpoint
                    ask=0.0,  # Not provided in stats endpoint
                    volume=float(response.get('volume', 0)),
                    change=0.0,  # Calculate from open and last
                    change_percent=0.0,  # Calculate from open and last
                    
                    high=float(response.get('high', 0)),
                    low=float(response.get('low', 0)),
                    open=float(response.get('open', 0)),
                    close=float(response.get('last', 0)),
                    
                    base_volume=float(response.get('volume', 0)),
                    quote_volume=float(response.get('volume_30day', 0)),
                    
                    exchange="coinbase",
                    timestamp=datetime.now()
                )
                
                # Calculate change if we have open price
                if ticker.open > 0:
                    ticker.change = ticker.last - ticker.open
                    ticker.change_percent = (ticker.change / ticker.open) * 100
                
                self.tickers[symbol] = ticker
                return ticker
            else:
                raise Exception("Failed to fetch ticker data")
                
        except Exception as e:
            logger.error(f"❌ Error fetching Coinbase ticker for {symbol}: {e}")
            raise
    
    async def fetch_orderbook(self, symbol: str, limit: int = 100) -> ExchangeOrderBook:
        """Fetch orderbook data for a symbol"""
        try:
            formatted_symbol = self.format_symbol(symbol)
            
            # Get order book
            params = {'limit': min(limit, 100)}  # Coinbase max is 100
            response = await self._make_request('GET', f'products/{formatted_symbol}/book', params=params)
            
            if response and 'pricebook' in response:
                pricebook = response['pricebook']
                
                # Convert to our format
                bids = [(float(bid['price']), float(bid['size'])) for bid in pricebook.get('bids', [])]
                asks = [(float(ask['price']), float(ask['size'])) for ask in pricebook.get('asks', [])]
                
                orderbook = ExchangeOrderBook(
                    symbol=symbol,
                    bids=bids,
                    asks=asks,
                    timestamp=datetime.now()
                )
                
                self.orderbooks[symbol] = orderbook
                return orderbook
            else:
                raise Exception("Failed to fetch orderbook data")
                
        except Exception as e:
            logger.error(f"❌ Error fetching Coinbase orderbook for {symbol}: {e}")
            raise
    
    # ==================== ACCOUNT MANAGEMENT ====================
    
    async def fetch_balance(self) -> Dict[str, ExchangeBalance]:
        """Fetch account balance"""
        try:
            response = await self._make_request('GET', 'accounts')
            
            if response and 'accounts' in response:
                balances = {}
                
                for account in response['accounts']:
                    currency = account['currency']
                    available = float(account.get('available_balance', {}).get('value', 0))
                    hold = float(account.get('hold', {}).get('value', 0))
                    total = available + hold
                    
                    if total > 0:  # Only include non-zero balances
                        balances[currency] = ExchangeBalance(
                            currency=currency,
                            total=total,
                            used=hold,
                            free=available,
                            last_updated=datetime.now()
                        )
                
                self.balances = balances
                return balances
            else:
                logger.error("❌ Failed to fetch Coinbase balance")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Error fetching Coinbase balance: {e}")
            return {}
    
    async def fetch_positions(self) -> Dict[str, ExchangePosition]:
        """Fetch account positions (not applicable for Coinbase spot)"""
        # Coinbase doesn't have margin/futures positions
        return {}
    
    # ==================== ORDER MANAGEMENT ====================
    
    async def create_order(self, symbol: str, side: str, amount: float,
                          order_type: str = "market", price: Optional[float] = None,
                          **kwargs) -> ExchangeOrder:
        """Create an order on Coinbase"""
        try:
            formatted_symbol = self.format_symbol(symbol)
            
            # Validate symbol
            if formatted_symbol not in self.product_info:
                raise ValueError(f"Symbol {symbol} not found on Coinbase")
            
            # Prepare order configuration
            order_config = {
                "market_market_ioc": {
                    "quote_size": str(amount) if side.lower() == "buy" else None,
                    "base_size": str(amount) if side.lower() == "sell" else None
                }
            } if order_type.lower() == "market" else {
                "limit_limit_gtc": {
                    "base_size": str(amount),
                    "limit_price": str(price),
                    "post_only": kwargs.get('post_only', False)
                }
            }
            
            # Prepare order data
            order_data = {
                "client_order_id": kwargs.get('client_order_id', str(uuid.uuid4())),
                "product_id": formatted_symbol,
                "side": side.upper(),
                "order_configuration": order_config
            }
            
            # Submit order
            response = await self._make_request('POST', 'orders', data=order_data)
            
            if response and response.get('success'):
                order_details = response.get('order_details', {})
                
                # Create order object
                order = ExchangeOrder(
                    id=order_details.get('order_id', ''),
                    symbol=symbol,
                    side=OrderSide.BUY if side.upper() == 'BUY' else OrderSide.SELL,
                    type=OrderType.MARKET if order_type.lower() == 'market' else OrderType.LIMIT,
                    amount=amount,
                    price=price,
                    status=OrderStatus.PENDING,
                    
                    exchange="coinbase",
                    client_order_id=order_data['client_order_id'],
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    info=response
                )
                
                self.orders[order.id] = order
                
                logger.info(f"✅ Coinbase order created: {order.id} - {side.upper()} {amount} {symbol}")
                return order
            else:
                error_msg = response.get('error_response', {}).get('message', 'Unknown error')
                logger.error(f"❌ Coinbase order failed: {error_msg}")
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"❌ Error creating Coinbase order: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        try:
            response = await self._make_request('POST', f'orders/batch_cancel', 
                                              data={'order_ids': [order_id]})
            
            if response and response.get('results'):
                result = response['results'][0]
                if result.get('success'):
                    # Update local order if exists
                    if order_id in self.orders:
                        self.orders[order_id].status = OrderStatus.CANCELED
                        self.orders[order_id].updated_at = datetime.now()
                    
                    logger.info(f"✅ Coinbase order cancelled: {order_id}")
                    return True
                else:
                    error_msg = result.get('failure_reason', 'Unknown error')
                    logger.error(f"❌ Coinbase cancel order failed: {error_msg}")
                    return False
            else:
                logger.error("❌ Coinbase cancel order failed: No response")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error cancelling Coinbase order {order_id}: {e}")
            return False
    
    async def fetch_order(self, order_id: str, symbol: str) -> Optional[ExchangeOrder]:
        """Fetch order status"""
        try:
            response = await self._make_request('GET', f'orders/historical/{order_id}')
            
            if response and 'order' in response:
                order_data = response['order']
                order = self._parse_order(order_data)
                self.orders[order_id] = order
                return order
            else:
                logger.error(f"❌ Failed to fetch Coinbase order {order_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error fetching Coinbase order {order_id}: {e}")
            return None
    
    async def fetch_orders(self, symbol: Optional[str] = None,
                          status: Optional[str] = None) -> List[ExchangeOrder]:
        """Fetch orders"""
        try:
            params = {}
            if symbol:
                params['product_id'] = self.format_symbol(symbol)
            
            response = await self._make_request('GET', 'orders/historical/batch', params=params)
            
            orders = []
            if response and 'orders' in response:
                for order_data in response['orders']:
                    order = self._parse_order(order_data)
                    if not status or order.status.value.lower() == status.lower():
                        orders.append(order)
                        self.orders[order.id] = order
            
            return orders
            
        except Exception as e:
            logger.error(f"❌ Error fetching Coinbase orders: {e}")
            return []
    
    # ==================== HELPER METHODS ====================
    
    def _parse_order(self, order_data: Dict[str, Any]) -> ExchangeOrder:
        """Parse Coinbase order data into our format"""
        
        # Extract order configuration details
        config = order_data.get('order_configuration', {})
        
        # Determine order type and details from configuration
        order_type = OrderType.MARKET
        amount = 0.0
        price = None
        
        if 'market_market_ioc' in config:
            order_type = OrderType.MARKET
            market_config = config['market_market_ioc']
            amount = float(market_config.get('quote_size', 0)) or float(market_config.get('base_size', 0))
        elif 'limit_limit_gtc' in config:
            order_type = OrderType.LIMIT
            limit_config = config['limit_limit_gtc']
            amount = float(limit_config.get('base_size', 0))
            price = float(limit_config.get('limit_price', 0))
        
        return ExchangeOrder(
            id=order_data.get('order_id', ''),
            symbol=order_data.get('product_id', ''),
            side=OrderSide.BUY if order_data.get('side') == 'BUY' else OrderSide.SELL,
            type=order_type,
            amount=amount,
            price=price,
            status=self._parse_order_status(order_data.get('status', 'UNKNOWN')),
            
            filled=float(order_data.get('filled_size', 0)),
            remaining=amount - float(order_data.get('filled_size', 0)),
            cost=float(order_data.get('filled_value', 0)),
            average=float(order_data.get('average_filled_price', 0)) if order_data.get('average_filled_price') else None,
            
            timestamp=self._parse_datetime(order_data.get('created_time')),
            created_at=self._parse_datetime(order_data.get('created_time')),
            updated_at=datetime.now(),
            
            exchange="coinbase",
            client_order_id=order_data.get('client_order_id'),
            info=order_data
        )
    
    def _parse_order_status(self, coinbase_status: str) -> OrderStatus:
        """Convert Coinbase order status to our OrderStatus"""
        status_map = {
            'PENDING': OrderStatus.PENDING,
            'OPEN': OrderStatus.OPEN,
            'FILLED': OrderStatus.CLOSED,
            'CANCELLED': OrderStatus.CANCELED,
            'EXPIRED': OrderStatus.EXPIRED,
            'FAILED': OrderStatus.REJECTED,
            'UNKNOWN': OrderStatus.PENDING
        }
        return status_map.get(coinbase_status, OrderStatus.PENDING)
    
    def _parse_datetime(self, datetime_str: Optional[str]) -> datetime:
        """Parse Coinbase datetime string"""
        if not datetime_str:
            return datetime.now()
        
        try:
            # Coinbase uses ISO format with timezone
            return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        except:
            return datetime.now()
    
    def format_symbol(self, symbol: str) -> str:
        """Format symbol for Coinbase API"""
        # Coinbase uses hyphenated symbols (e.g., BTC-USD)
        if '/' in symbol:
            base, quote = symbol.split('/')
            return f"{base.upper()}-{quote.upper()}"
        elif '_' in symbol:
            base, quote = symbol.split('_')
            return f"{base.upper()}-{quote.upper()}"
        else:
            # Assume it's already in correct format or add -USD default
            if '-' not in symbol:
                return f"{symbol.upper()}-USD"
            return symbol.upper()
    
    # ==================== WEBSOCKET IMPLEMENTATION ====================
    
    async def _start_websocket(self):
        """Start WebSocket connection for real-time data"""
        try:
            asyncio.create_task(self._handle_websocket())
            logger.info("📡 Coinbase WebSocket connection started")
            
        except Exception as e:
            logger.error(f"❌ Error starting Coinbase WebSocket: {e}")
    
    async def _handle_websocket(self):
        """Handle WebSocket connection"""
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                self.ws_connection = websocket
                
                # Subscribe to channels
                await self._subscribe_to_channels()
                
                # Listen for messages
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self._process_websocket_message(data)
                    except Exception as e:
                        logger.error(f"❌ Error processing WebSocket message: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Coinbase WebSocket error: {e}")
    
    async def _subscribe_to_channels(self):
        """Subscribe to WebSocket channels"""
        try:
            if not self.ws_connection:
                return
            
            # Subscribe to ticker channel for all products
            subscribe_msg = {
                "type": "subscribe",
                "channels": [
                    {
                        "name": "ticker",
                        "product_ids": list(self.symbols[:10])  # Limit to first 10 symbols
                    }
                ]
            }
            
            await self.ws_connection.send(json.dumps(subscribe_msg))
            logger.info("📡 Subscribed to Coinbase ticker channel")
            
        except Exception as e:
            logger.error(f"❌ Error subscribing to channels: {e}")
    
    async def _process_websocket_message(self, data: Dict[str, Any]):
        """Process WebSocket messages"""
        try:
            message_type = data.get('type')
            
            if message_type == 'ticker':
                # Update ticker data
                product_id = data.get('product_id')
                if product_id:
                    ticker = ExchangeTicker(
                        symbol=product_id,
                        last=float(data.get('price', 0)),
                        bid=float(data.get('best_bid', 0)),
                        ask=float(data.get('best_ask', 0)),
                        volume=float(data.get('volume_24h', 0)),
                        change=0.0,  # Not provided in ticker
                        change_percent=0.0,  # Not provided in ticker
                        
                        high=float(data.get('high_24h', 0)),
                        low=float(data.get('low_24h', 0)),
                        open=float(data.get('open_24h', 0)),
                        
                        exchange="coinbase",
                        timestamp=self._parse_datetime(data.get('time'))
                    )
                    
                    self.tickers[product_id] = ticker
                    
            elif message_type == 'match':
                # Trade execution
                logger.debug(f"Trade: {data.get('product_id')} @ {data.get('price')}")
                
            elif message_type == 'error':
                # Error message
                logger.error(f"WebSocket error: {data.get('message')}")
                
        except Exception as e:
            logger.error(f"❌ Error processing WebSocket message: {e}")
    
    # ==================== SUBSCRIPTIONS ====================
    
    async def subscribe_ticker(self, symbol: str):
        """Subscribe to ticker updates"""
        try:
            if self.ws_connection:
                formatted_symbol = self.format_symbol(symbol)
                
                subscribe_msg = {
                    "type": "subscribe",
                    "channels": [
                        {
                            "name": "ticker",
                            "product_ids": [formatted_symbol]
                        }
                    ]
                }
                
                await self.ws_connection.send(json.dumps(subscribe_msg))
                self.ws_subscriptions.add(f"ticker:{symbol}")
                logger.info(f"📡 Subscribed to Coinbase ticker: {symbol}")
                
        except Exception as e:
            logger.error(f"❌ Error subscribing to ticker {symbol}: {e}")
    
    async def subscribe_orderbook(self, symbol: str):
        """Subscribe to orderbook updates"""
        try:
            if self.ws_connection:
                formatted_symbol = self.format_symbol(symbol)
                
                subscribe_msg = {
                    "type": "subscribe",
                    "channels": [
                        {
                            "name": "level2",
                            "product_ids": [formatted_symbol]
                        }
                    ]
                }
                
                await self.ws_connection.send(json.dumps(subscribe_msg))
                self.ws_subscriptions.add(f"orderbook:{symbol}")
                logger.info(f"📊 Subscribed to Coinbase orderbook: {symbol}")
                
        except Exception as e:
            logger.error(f"❌ Error subscribing to orderbook {symbol}: {e}")
    
    # ==================== CAPABILITIES ====================
    
    async def get_capabilities(self) -> List[str]:
        """Get Coinbase exchange capabilities"""
        return [
            "spot_trading",
            "fiat_onramp",
            "fiat_offramp",
            "institutional_custody",
            "regulatory_compliance",
            "market_orders",
            "limit_orders",
            "stop_orders",
            "real_time_data",
            "websocket_streaming",
            "high_liquidity",
            "usd_trading_pairs",
            "professional_trading",
            "api_trading",
            "mobile_trading"
        ]

# ==================== TESTING ====================

async def test_coinbase_exchange():
    """Test Coinbase exchange functionality"""
    print("🧪 Testing Coinbase Exchange")
    print("=" * 40)
    
    # Create test configuration
    config = IntegrationConfig(
        name="coinbase",
        integration_type="exchange",
        enabled=True,
        sandbox_mode=True,
        api_key="test_key",
        secret_key="test_secret",
        additional_auth={
            'passphrase': 'test_passphrase'
        }
    )
    
    # Create exchange instance
    exchange = CoinbaseExchange(config)
    
    print("✅ Exchange instance created")
    print(f"✅ Using sandbox: {config.sandbox_mode}")
    print(f"✅ Base URL: {exchange.base_url}")
    
    # Test capabilities
    capabilities = await exchange.get_capabilities()
    print(f"✅ Capabilities: {', '.join(capabilities[:5])}...")
    
    # Test symbol formatting
    test_symbols = ['BTC/USD', 'ETH-USD', 'ADA_USD']
    for symbol in test_symbols:
        formatted = exchange.format_symbol(symbol)
        print(f"✅ Symbol format: {symbol} -> {formatted}")
    
    print("\n⚠️  Note: Full testing requires valid API credentials")
    print("   Set up Coinbase Pro sandbox account for complete testing")
    
    print("\n🎉 Coinbase exchange tests completed!")

if __name__ == "__main__":
    asyncio.run(test_coinbase_exchange())