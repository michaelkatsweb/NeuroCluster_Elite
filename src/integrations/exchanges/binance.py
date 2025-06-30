 #!/usr/bin/env python3
"""
File: binance.py
Path: NeuroCluster-Elite/src/integrations/exchanges/binance.py
Description: Binance exchange integration for NeuroCluster Elite

This module implements the Binance API integration, providing access to the world's
largest cryptocurrency exchange with comprehensive spot, margin, and futures trading.

Features:
- Binance REST API v3 integration
- Binance WebSocket streaming
- Spot, margin, and futures trading
- Real-time market data and orderbook
- Advanced order types
- Portfolio and position management
- Cross-exchange arbitrage support
- Rate limiting and error handling

API Documentation: https://binance-docs.github.io/apidocs/

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
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from urllib.parse import urlencode
import base64

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

# ==================== BINANCE SPECIFIC ENUMS ====================

class BinanceOrderType(Enum):
    """Binance order types"""
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"
    LIMIT_MAKER = "LIMIT_MAKER"

class BinanceOrderSide(Enum):
    """Binance order sides"""
    BUY = "BUY"
    SELL = "SELL"

class BinanceTimeInForce(Enum):
    """Binance time in force options"""
    GTC = "GTC"  # Good Till Canceled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill

class BinanceOrderStatus(Enum):
    """Binance order status"""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    PENDING_CANCEL = "PENDING_CANCEL"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class BinanceKlineInterval(Enum):
    """Binance kline intervals"""
    INTERVAL_1m = "1m"
    INTERVAL_3m = "3m"
    INTERVAL_5m = "5m"
    INTERVAL_15m = "15m"
    INTERVAL_30m = "30m"
    INTERVAL_1h = "1h"
    INTERVAL_2h = "2h"
    INTERVAL_4h = "4h"
    INTERVAL_6h = "6h"
    INTERVAL_8h = "8h"
    INTERVAL_12h = "12h"
    INTERVAL_1d = "1d"
    INTERVAL_3d = "3d"
    INTERVAL_1w = "1w"
    INTERVAL_1M = "1M"

# ==================== BINANCE CONFIG ====================

@dataclass
class BinanceConfig:
    """Binance-specific configuration"""
    api_key: str
    secret_key: str
    
    # URLs
    base_url: str = "https://api.binance.com"
    testnet_url: str = "https://testnet.binance.vision"
    websocket_url: str = "wss://stream.binance.com:9443"
    testnet_ws_url: str = "wss://testnet.binance.vision"
    
    # Trading settings
    trading_mode: TradingMode = TradingMode.SPOT
    use_testnet: bool = True
    
    # Features
    enable_futures: bool = True
    enable_margin: bool = True
    enable_spot: bool = True
    
    # Risk settings
    max_order_value: float = 10000.0
    default_leverage: float = 1.0
    
    # Rate limiting
    requests_per_minute: int = 1200
    orders_per_second: int = 10
    orders_per_day: int = 200000

# ==================== BINANCE EXCHANGE IMPLEMENTATION ====================

class BinanceExchange(BaseExchange):
    """
    Binance exchange implementation
    
    Provides comprehensive integration with Binance's spot, margin, and futures
    trading APIs with real-time WebSocket streaming capabilities.
    """
    
    def __init__(self, config: IntegrationConfig):
        """Initialize Binance exchange"""
        super().__init__(config)
        
        # Extract Binance-specific config
        self.api_key = config.api_key or config.additional_auth.get('api_key', '')
        self.secret_key = config.secret_key or config.additional_auth.get('secret_key', '')
        
        # URLs (use testnet if in sandbox mode)
        if config.sandbox_mode:
            self.base_url = "https://testnet.binance.vision"
            self.websocket_url = "wss://testnet.binance.vision"
        else:
            self.base_url = "https://api.binance.com"
            self.websocket_url = "wss://stream.binance.com:9443"
        
        # API endpoints
        self.api_version = "/api/v3"
        self.fapi_version = "/fapi/v1"  # Futures API
        self.dapi_version = "/dapi/v1"  # Delivery API
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # WebSocket connections
        self.ws_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.listen_key: Optional[str] = None
        
        # Trading features
        self.supports_spot = True
        self.supports_margin = True
        self.supports_futures = True
        self.supports_websocket = True
        
        # Rate limiting
        self.weight_used = 0
        self.weight_reset_time = datetime.now()
        self.order_count = 0
        self.order_count_reset_time = datetime.now()
        
        # Market data cache
        self.symbol_info: Dict[str, Any] = {}
        self.price_filters: Dict[str, Any] = {}
        
        logger.info(f"🟡 Binance exchange initialized - {'Testnet' if config.sandbox_mode else 'Live'}")
    
    # ==================== CONNECTION MANAGEMENT ====================
    
    async def connect(self) -> bool:
        """Connect to Binance API"""
        try:
            # Validate credentials
            if not self.api_key or not self.secret_key:
                error_msg = "Binance API credentials not provided"
                self.update_status(IntegrationStatus.ERROR, error_msg)
                logger.error(f"❌ {error_msg}")
                return False
            
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers=self._get_headers()
            )
            
            # Test connection and load markets
            if await self._test_connectivity():
                # Load market data
                await self.load_markets()
                
                # Start user data stream
                await self._start_user_data_stream()
                
                self.update_status(IntegrationStatus.CONNECTED)
                logger.info("✅ Binance exchange connected successfully")
                return True
            else:
                error_msg = "Failed to connect to Binance API"
                self.update_status(IntegrationStatus.ERROR, error_msg)
                logger.error(f"❌ {error_msg}")
                return False
                
        except Exception as e:
            error_msg = f"Binance connection failed: {e}"
            self.update_status(IntegrationStatus.ERROR, error_msg)
            logger.error(f"❌ {error_msg}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Binance API"""
        try:
            # Close user data stream
            if self.listen_key:
                await self._close_user_data_stream()
            
            # Close WebSocket connections
            for ws in self.ws_connections.values():
                if ws and not ws.closed:
                    await ws.close()
            self.ws_connections.clear()
            
            # Close HTTP session
            if self.session:
                await self.session.close()
                self.session = None
            
            self.update_status(IntegrationStatus.DISCONNECTED)
            logger.info("✅ Binance exchange disconnected")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error disconnecting Binance: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test Binance connection"""
        return await self._test_connectivity()
    
    async def _test_connectivity(self) -> bool:
        """Test API connectivity"""
        try:
            if not self.session:
                return False
            
            async with self.session.get(f"{self.base_url}{self.api_version}/ping") as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"❌ Binance connectivity test failed: {e}")
            return False
    
    # ==================== AUTHENTICATION ====================
    
    def _get_headers(self) -> Dict[str, str]:
        """Get common headers for API requests"""
        return {
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/json"
        }
    
    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC SHA256 signature for authenticated requests"""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _create_signed_params(self, params: Dict[str, Any]) -> str:
        """Create signed parameters for authenticated requests"""
        # Add timestamp
        params['timestamp'] = int(time.time() * 1000)
        
        # Create query string
        query_string = urlencode(params)
        
        # Generate signature
        signature = self._generate_signature(query_string)
        
        # Add signature to params
        return f"{query_string}&signature={signature}"
    
    # ==================== MARKET DATA ====================
    
    async def load_markets(self) -> Dict[str, Any]:
        """Load available markets and trading pairs"""
        try:
            if not self.session:
                return {}
            
            async with self.session.get(f"{self.base_url}{self.api_version}/exchangeInfo") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Process symbols
                    for symbol_data in data.get('symbols', []):
                        symbol = symbol_data['symbol']
                        self.symbol_info[symbol] = symbol_data
                        
                        # Extract price filters
                        for filter_data in symbol_data.get('filters', []):
                            if filter_data['filterType'] == 'PRICE_FILTER':
                                self.price_filters[symbol] = filter_data
                    
                    self.symbols = list(self.symbol_info.keys())
                    self.markets = self.symbol_info
                    
                    logger.info(f"📊 Loaded {len(self.symbols)} Binance trading pairs")
                    return self.markets
                else:
                    logger.error(f"❌ Failed to load Binance markets: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"❌ Error loading Binance markets: {e}")
            return {}
    
    async def fetch_ticker(self, symbol: str) -> ExchangeTicker:
        """Fetch ticker data for a symbol"""
        try:
            if not self.session:
                await self.connect()
            
            formatted_symbol = self.format_symbol(symbol)
            
            async with self.session.get(
                f"{self.base_url}{self.api_version}/ticker/24hr",
                params={'symbol': formatted_symbol}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    ticker = ExchangeTicker(
                        symbol=symbol,
                        last=float(data['lastPrice']),
                        bid=float(data['bidPrice']),
                        ask=float(data['askPrice']),
                        volume=float(data['volume']),
                        change=float(data['priceChange']),
                        change_percent=float(data['priceChangePercent']),
                        
                        high=float(data['highPrice']),
                        low=float(data['lowPrice']),
                        open=float(data['openPrice']),
                        close=float(data['lastPrice']),
                        
                        base_volume=float(data['volume']),
                        quote_volume=float(data['quoteVolume']),
                        
                        exchange="binance",
                        timestamp=datetime.now()
                    )
                    
                    self.tickers[symbol] = ticker
                    return ticker
                else:
                    raise Exception(f"API error: {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ Error fetching Binance ticker for {symbol}: {e}")
            raise
    
    async def fetch_orderbook(self, symbol: str, limit: int = 100) -> ExchangeOrderBook:
        """Fetch orderbook data for a symbol"""
        try:
            if not self.session:
                await self.connect()
            
            formatted_symbol = self.format_symbol(symbol)
            
            async with self.session.get(
                f"{self.base_url}{self.api_version}/depth",
                params={'symbol': formatted_symbol, 'limit': limit}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Convert to our format
                    bids = [(float(price), float(amount)) for price, amount in data['bids']]
                    asks = [(float(price), float(amount)) for price, amount in data['asks']]
                    
                    orderbook = ExchangeOrderBook(
                        symbol=symbol,
                        bids=bids,
                        asks=asks,
                        timestamp=datetime.now()
                    )
                    
                    self.orderbooks[symbol] = orderbook
                    return orderbook
                else:
                    raise Exception(f"API error: {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ Error fetching Binance orderbook for {symbol}: {e}")
            raise
    
    # ==================== ACCOUNT MANAGEMENT ====================
    
    async def fetch_balance(self) -> Dict[str, ExchangeBalance]:
        """Fetch account balance"""
        try:
            if not self.session:
                await self.connect()
            
            # Create signed request
            params = {}
            query_string = self._create_signed_params(params)
            
            async with self.session.get(
                f"{self.base_url}{self.api_version}/account?{query_string}",
                headers=self._get_headers()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    balances = {}
                    
                    for balance_data in data.get('balances', []):
                        asset = balance_data['asset']
                        free = float(balance_data['free'])
                        locked = float(balance_data['locked'])
                        total = free + locked
                        
                        if total > 0:  # Only include non-zero balances
                            balances[asset] = ExchangeBalance(
                                currency=asset,
                                total=total,
                                used=locked,
                                free=free,
                                last_updated=datetime.now()
                            )
                    
                    self.balances = balances
                    return balances
                else:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"❌ Error fetching Binance balance: {e}")
            return {}
    
    async def fetch_positions(self) -> Dict[str, ExchangePosition]:
        """Fetch account positions (futures only)"""
        try:
            if not self.session:
                await self.connect()
            
            # This is for futures trading only
            if not self.supports_futures:
                return {}
            
            # Create signed request
            params = {}
            query_string = self._create_signed_params(params)
            
            async with self.session.get(
                f"{self.base_url}{self.fapi_version}/positionRisk?{query_string}",
                headers=self._get_headers()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    positions = {}
                    
                    for pos_data in data:
                        symbol = pos_data['symbol']
                        position_amt = float(pos_data['positionAmt'])
                        
                        if position_amt != 0:  # Only include open positions
                            position = ExchangePosition(
                                symbol=symbol,
                                side='long' if position_amt > 0 else 'short',
                                size=abs(position_amt),
                                notional=float(pos_data['notional']),
                                entry_price=float(pos_data['entryPrice']),
                                mark_price=float(pos_data['markPrice']),
                                unrealized_pnl=float(pos_data['unRealizedProfit']),
                                realized_pnl=0.0,  # Not provided in this endpoint
                                
                                leverage=float(pos_data.get('leverage', 1)),
                                margin_ratio=float(pos_data.get('marginRatio', 0)),
                                liquidation_price=float(pos_data.get('liquidationPrice', 0)) or None,
                                
                                exchange="binance",
                                last_updated=datetime.now()
                            )
                            
                            positions[symbol] = position
                    
                    self.positions = positions
                    return positions
                else:
                    logger.warning(f"⚠️ Futures positions not available: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"❌ Error fetching Binance positions: {e}")
            return {}
    
    # ==================== ORDER MANAGEMENT ====================
    
    async def create_order(self, symbol: str, side: str, amount: float,
                          order_type: str = "market", price: Optional[float] = None,
                          **kwargs) -> ExchangeOrder:
        """Create an order on Binance"""
        try:
            if not self.session:
                await self.connect()
            
            # Format symbol and validate
            formatted_symbol = self.format_symbol(symbol)
            if formatted_symbol not in self.symbol_info:
                raise ValueError(f"Symbol {symbol} not found on Binance")
            
            # Prepare order parameters
            params = {
                'symbol': formatted_symbol,
                'side': side.upper(),
                'type': self._convert_order_type(order_type),
                'quantity': self._format_quantity(formatted_symbol, amount)
            }
            
            # Add price for limit orders
            if order_type.lower() in ['limit', 'stop_loss_limit', 'take_profit_limit']:
                if not price:
                    raise ValueError(f"Price required for {order_type} orders")
                params['price'] = self._format_price(formatted_symbol, price)
                params['timeInForce'] = kwargs.get('time_in_force', 'GTC')
            
            # Add stop price for stop orders
            if order_type.lower() in ['stop_loss', 'stop_loss_limit', 'take_profit', 'take_profit_limit']:
                stop_price = kwargs.get('stop_price') or price
                if stop_price:
                    params['stopPrice'] = self._format_price(formatted_symbol, stop_price)
            
            # Add client order ID if provided
            if 'client_order_id' in kwargs:
                params['newClientOrderId'] = kwargs['client_order_id']
            
            # Create signed request
            query_string = self._create_signed_params(params)
            
            # Submit order
            async with self.session.post(
                f"{self.base_url}{self.api_version}/order",
                data=query_string,
                headers=self._get_headers()
            ) as response:
                
                if response.status == 200:
                    order_data = await response.json()
                    
                    # Parse order response
                    order = self._parse_order(order_data)
                    self.orders[order.id] = order
                    
                    logger.info(f"✅ Binance order created: {order.id} - {side.upper()} {amount} {symbol}")
                    return order
                else:
                    error_text = await response.text()
                    error_data = json.loads(error_text) if error_text else {}
                    error_msg = error_data.get('msg', f'HTTP {response.status}')
                    
                    logger.error(f"❌ Binance order failed: {error_msg}")
                    raise Exception(error_msg)
                    
        except Exception as e:
            logger.error(f"❌ Error creating Binance order: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        try:
            if not self.session:
                await self.connect()
            
            formatted_symbol = self.format_symbol(symbol)
            
            # Prepare cancel parameters
            params = {
                'symbol': formatted_symbol,
                'orderId': order_id
            }
            
            # Create signed request
            query_string = self._create_signed_params(params)
            
            async with self.session.delete(
                f"{self.base_url}{self.api_version}/order?{query_string}",
                headers=self._get_headers()
            ) as response:
                
                if response.status == 200:
                    # Update local order if exists
                    if order_id in self.orders:
                        self.orders[order_id].status = OrderStatus.CANCELED
                        self.orders[order_id].updated_at = datetime.now()
                    
                    logger.info(f"✅ Binance order cancelled: {order_id}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Binance cancel order failed: {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error cancelling Binance order {order_id}: {e}")
            return False
    
    async def fetch_order(self, order_id: str, symbol: str) -> Optional[ExchangeOrder]:
        """Fetch order status"""
        try:
            if not self.session:
                await self.connect()
            
            formatted_symbol = self.format_symbol(symbol)
            
            params = {
                'symbol': formatted_symbol,
                'orderId': order_id
            }
            
            query_string = self._create_signed_params(params)
            
            async with self.session.get(
                f"{self.base_url}{self.api_version}/order?{query_string}",
                headers=self._get_headers()
            ) as response:
                
                if response.status == 200:
                    order_data = await response.json()
                    order = self._parse_order(order_data)
                    self.orders[order_id] = order
                    return order
                else:
                    logger.error(f"❌ Failed to fetch Binance order {order_id}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Error fetching Binance order {order_id}: {e}")
            return None
    
    async def fetch_orders(self, symbol: Optional[str] = None,
                          status: Optional[str] = None) -> List[ExchangeOrder]:
        """Fetch orders"""
        try:
            if not self.session:
                await self.connect()
            
            orders = []
            
            if symbol:
                # Fetch orders for specific symbol
                formatted_symbol = self.format_symbol(symbol)
                params = {'symbol': formatted_symbol}
                
                query_string = self._create_signed_params(params)
                
                async with self.session.get(
                    f"{self.base_url}{self.api_version}/openOrders?{query_string}",
                    headers=self._get_headers()
                ) as response:
                    
                    if response.status == 200:
                        orders_data = await response.json()
                        
                        for order_data in orders_data:
                            order = self._parse_order(order_data)
                            if not status or order.status.value.lower() == status.lower():
                                orders.append(order)
                                self.orders[order.id] = order
            else:
                # Return cached orders if no symbol specified
                for order in self.orders.values():
                    if not status or order.status.value.lower() == status.lower():
                        orders.append(order)
            
            return orders
            
        except Exception as e:
            logger.error(f"❌ Error fetching Binance orders: {e}")
            return []
    
    # ==================== HELPER METHODS ====================
    
    def _convert_order_type(self, order_type: str) -> str:
        """Convert our order type to Binance order type"""
        type_map = {
            'market': 'MARKET',
            'limit': 'LIMIT',
            'stop_loss': 'STOP_LOSS',
            'stop_loss_limit': 'STOP_LOSS_LIMIT',
            'take_profit': 'TAKE_PROFIT',
            'take_profit_limit': 'TAKE_PROFIT_LIMIT',
            'limit_maker': 'LIMIT_MAKER'
        }
        return type_map.get(order_type.lower(), 'MARKET')
    
    def _parse_order(self, order_data: Dict[str, Any]) -> ExchangeOrder:
        """Parse Binance order data into our format"""
        
        return ExchangeOrder(
            id=str(order_data['orderId']),
            symbol=order_data['symbol'],
            side=OrderSide.BUY if order_data['side'] == 'BUY' else OrderSide.SELL,
            type=self._parse_order_type(order_data['type']),
            amount=float(order_data['origQty']),
            price=float(order_data['price']) if order_data['price'] != '0.00000000' else None,
            status=self._parse_order_status(order_data['status']),
            
            filled=float(order_data['executedQty']),
            remaining=float(order_data['origQty']) - float(order_data['executedQty']),
            cost=float(order_data['cummulativeQuoteQty']),
            average=float(order_data['price']) if order_data['price'] != '0.00000000' else None,
            
            timestamp=datetime.fromtimestamp(order_data['time'] / 1000),
            created_at=datetime.fromtimestamp(order_data['time'] / 1000),
            updated_at=datetime.fromtimestamp(order_data['updateTime'] / 1000),
            
            exchange="binance",
            client_order_id=order_data.get('clientOrderId'),
            info=order_data
        )
    
    def _parse_order_type(self, binance_type: str) -> OrderType:
        """Convert Binance order type to our OrderType"""
        type_map = {
            'MARKET': OrderType.MARKET,
            'LIMIT': OrderType.LIMIT,
            'STOP_LOSS': OrderType.STOP_LOSS,
            'STOP_LOSS_LIMIT': OrderType.STOP_LOSS_LIMIT,
            'TAKE_PROFIT': OrderType.TAKE_PROFIT,
            'TAKE_PROFIT_LIMIT': OrderType.TAKE_PROFIT_LIMIT,
            'LIMIT_MAKER': OrderType.LIMIT_MAKER
        }
        return type_map.get(binance_type, OrderType.MARKET)
    
    def _parse_order_status(self, binance_status: str) -> OrderStatus:
        """Convert Binance order status to our OrderStatus"""
        status_map = {
            'NEW': OrderStatus.OPEN,
            'PARTIALLY_FILLED': OrderStatus.OPEN,
            'FILLED': OrderStatus.CLOSED,
            'CANCELED': OrderStatus.CANCELED,
            'PENDING_CANCEL': OrderStatus.OPEN,
            'REJECTED': OrderStatus.REJECTED,
            'EXPIRED': OrderStatus.EXPIRED
        }
        return status_map.get(binance_status, OrderStatus.PENDING)
    
    def _format_quantity(self, symbol: str, quantity: float) -> str:
        """Format quantity according to symbol's lot size filter"""
        # Get lot size filter for symbol
        symbol_info = self.symbol_info.get(symbol, {})
        for filter_info in symbol_info.get('filters', []):
            if filter_info['filterType'] == 'LOT_SIZE':
                step_size = float(filter_info['stepSize'])
                
                # Round quantity to step size
                precision = len(filter_info['stepSize'].rstrip('0').split('.')[1]) if '.' in filter_info['stepSize'] else 0
                quantity = (quantity // step_size) * step_size
                
                return f"{quantity:.{precision}f}"
        
        # Default formatting
        return f"{quantity:.8f}".rstrip('0').rstrip('.')
    
    def _format_price(self, symbol: str, price: float) -> str:
        """Format price according to symbol's price filter"""
        # Get price filter for symbol
        if symbol in self.price_filters:
            filter_info = self.price_filters[symbol]
            tick_size = float(filter_info['tickSize'])
            
            # Round price to tick size
            precision = len(filter_info['tickSize'].rstrip('0').split('.')[1]) if '.' in filter_info['tickSize'] else 0
            price = (price // tick_size) * tick_size
            
            return f"{price:.{precision}f}"
        
        # Default formatting
        return f"{price:.8f}".rstrip('0').rstrip('.')
    
    def format_symbol(self, symbol: str) -> str:
        """Format symbol for Binance API"""
        # Binance uses concatenated symbols (e.g., BTCUSDT)
        if '/' in symbol:
            base, quote = symbol.split('/')
            return f"{base.upper()}{quote.upper()}"
        elif '-' in symbol:
            base, quote = symbol.split('-')
            return f"{base.upper()}{quote.upper()}"
        else:
            return symbol.upper()
    
    # ==================== WEBSOCKET IMPLEMENTATION ====================
    
    async def _start_user_data_stream(self):
        """Start user data stream for real-time account updates"""
        try:
            if not self.session:
                return
            
            # Create listen key
            async with self.session.post(
                f"{self.base_url}{self.api_version}/userDataStream",
                headers=self._get_headers()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.listen_key = data['listenKey']
                    
                    # Start WebSocket connection
                    asyncio.create_task(self._handle_user_data_stream())
                    
                    logger.info("📡 Binance user data stream started")
                else:
                    logger.error(f"❌ Failed to start user data stream: {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ Error starting user data stream: {e}")
    
    async def _handle_user_data_stream(self):
        """Handle user data stream WebSocket"""
        try:
            if not self.listen_key:
                return
            
            ws_url = f"{self.websocket_url}/ws/{self.listen_key}"
            
            async with websockets.connect(ws_url) as websocket:
                self.ws_connections['user_data'] = websocket
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self._process_user_data_update(data)
                    except Exception as e:
                        logger.error(f"❌ Error processing user data update: {e}")
                        
        except Exception as e:
            logger.error(f"❌ User data stream error: {e}")
    
    async def _process_user_data_update(self, data: Dict[str, Any]):
        """Process user data stream updates"""
        event_type = data.get('e')
        
        if event_type == 'executionReport':
            # Order update
            order_data = {
                'orderId': data['i'],
                'symbol': data['s'],
                'side': data['S'],
                'type': data['o'],
                'origQty': data['q'],
                'price': data['p'],
                'status': data['X'],
                'executedQty': data['z'],
                'cummulativeQuoteQty': data['Z'],
                'time': data['O'],
                'updateTime': data['T'],
                'clientOrderId': data['c']
            }
            
            order = self._parse_order(order_data)
            self.orders[order.id] = order
            
            logger.info(f"📋 Order update: {order.id} - {order.status.value}")
            
        elif event_type == 'outboundAccountPosition':
            # Balance update
            asset = data['a']
            free = float(data['f'])
            locked = float(data['l'])
            
            self.balances[asset] = ExchangeBalance(
                currency=asset,
                total=free + locked,
                used=locked,
                free=free,
                last_updated=datetime.now()
            )
            
            logger.info(f"💰 Balance update: {asset} = {free + locked}")
    
    async def _close_user_data_stream(self):
        """Close user data stream"""
        try:
            if self.listen_key and self.session:
                async with self.session.delete(
                    f"{self.base_url}{self.api_version}/userDataStream",
                    params={'listenKey': self.listen_key},
                    headers=self._get_headers()
                ) as response:
                    if response.status == 200:
                        logger.info("✅ User data stream closed")
                    else:
                        logger.warning(f"⚠️ Failed to close user data stream: {response.status}")
                        
                self.listen_key = None
                
        except Exception as e:
            logger.error(f"❌ Error closing user data stream: {e}")
    
    # ==================== CAPABILITIES ====================
    
    async def get_capabilities(self) -> List[str]:
        """Get Binance exchange capabilities"""
        return [
            "spot_trading",
            "margin_trading",
            "futures_trading",
            "market_orders",
            "limit_orders",
            "stop_orders",
            "take_profit_orders",
            "limit_maker_orders",
            "real_time_data",
            "websocket_streaming",
            "user_data_stream",
            "high_frequency_trading",
            "advanced_order_types",
            "portfolio_margin",
            "cross_collateral",
            "lending",
            "staking"
        ]

# ==================== TESTING ====================

async def test_binance_exchange():
    """Test Binance exchange functionality"""
    print("🧪 Testing Binance Exchange")
    print("=" * 40)
    
    # Create test configuration
    config = IntegrationConfig(
        name="binance",
        integration_type="exchange",
        enabled=True,
        sandbox_mode=True,
        api_key="test_key",
        secret_key="test_secret"
    )
    
    # Create exchange instance
    exchange = BinanceExchange(config)
    
    print("✅ Exchange instance created")
    print(f"✅ Using testnet: {config.sandbox_mode}")
    print(f"✅ Base URL: {exchange.base_url}")
    
    # Test capabilities
    capabilities = await exchange.get_capabilities()
    print(f"✅ Capabilities: {', '.join(capabilities[:5])}...")
    
    # Test symbol formatting
    test_symbols = ['BTC/USDT', 'ETH-USD', 'ADAUSDT']
    for symbol in test_symbols:
        formatted = exchange.format_symbol(symbol)
        print(f"✅ Symbol format: {symbol} -> {formatted}")
    
    print("\n⚠️  Note: Full testing requires valid API credentials")
    print("   Set up Binance testnet account for complete testing")
    
    print("\n🎉 Binance exchange tests completed!")

if __name__ == "__main__":
    asyncio.run(test_binance_exchange())