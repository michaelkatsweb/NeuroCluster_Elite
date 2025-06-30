#!/usr/bin/env python3
"""
File: __init__.py
Path: NeuroCluster-Elite/src/integrations/exchanges/__init__.py
Description: Cryptocurrency exchange integrations package for NeuroCluster Elite

This module initializes the exchange integrations package and provides a unified interface
for all supported cryptocurrency exchanges including Binance, Coinbase, Kraken, and others.

Features:
- Exchange factory pattern for easy instantiation
- Common exchange interface and base classes
- Multi-exchange arbitrage capabilities
- Real-time orderbook and trade data
- WebSocket streaming for all exchanges
- Unified API across different exchanges
- Portfolio aggregation across exchanges

Author: Your Name
Created: 2025-06-30
Version: 1.0.0
License: MIT
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import ccxt
import ccxt.async_support as ccxt_async

# Import base integration classes
try:
    from src.integrations import BaseIntegration, IntegrationConfig, IntegrationStatus, IntegrationType
    from src.core.neurocluster_elite import AssetType
    from src.utils.helpers import format_currency, format_percentage
except ImportError:
    # Fallback for testing
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== EXCHANGE ENUMS ====================

class ExchangeType(Enum):
    """Supported exchange types"""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    KUCOIN = "kucoin"
    HUOBI = "huobi"
    OKEX = "okex"
    BITFINEX = "bitfinex"
    BITSTAMP = "bitstamp"
    GEMINI = "gemini"
    FTX = "ftx"  # Note: FTX is defunct but keeping for reference

class OrderType(Enum):
    """Exchange order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LOSS_LIMIT = "stop_loss_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"
    LIMIT_MAKER = "limit_maker"
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill

class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status values"""
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"
    PENDING = "pending"

class TradingMode(Enum):
    """Trading modes"""
    SPOT = "spot"
    MARGIN = "margin"
    FUTURES = "futures"
    OPTIONS = "options"

# ==================== DATA STRUCTURES ====================

@dataclass
class ExchangeBalance:
    """Exchange balance information"""
    currency: str
    total: float = 0.0
    used: float = 0.0
    free: float = 0.0
    
    # Additional fields
    usd_value: float = 0.0
    btc_value: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class ExchangePosition:
    """Exchange position information (for margin/futures)"""
    symbol: str
    side: str  # long, short
    size: float
    notional: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    realized_pnl: float
    
    # Risk metrics
    leverage: float = 1.0
    margin_ratio: float = 0.0
    liquidation_price: Optional[float] = None
    
    # Metadata
    exchange: str = ""
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class ExchangeOrder:
    """Exchange order information"""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    amount: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    
    # Execution details
    filled: float = 0.0
    remaining: float = 0.0
    cost: float = 0.0
    average: Optional[float] = None
    
    # Fees
    fee_currency: Optional[str] = None
    fee_cost: float = 0.0
    fee_rate: float = 0.0
    
    # Timestamps
    timestamp: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Metadata
    exchange: str = ""
    client_order_id: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExchangeTicker:
    """Exchange ticker information"""
    symbol: str
    last: float
    bid: float
    ask: float
    volume: float
    change: float
    change_percent: float
    
    # Additional price data
    high: float = 0.0
    low: float = 0.0
    open: float = 0.0
    close: float = 0.0
    
    # Volume data
    base_volume: float = 0.0
    quote_volume: float = 0.0
    
    # Metadata
    exchange: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ExchangeOrderBook:
    """Exchange orderbook data"""
    symbol: str
    bids: List[Tuple[float, float]]  # [(price, amount), ...]
    asks: List[Tuple[float, float]]  # [(price, amount), ...]
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Derived properties
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        if self.bids and self.asks:
            return self.asks[0][0] - self.bids[0][0]
        return 0.0
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price"""
        if self.bids and self.asks:
            return (self.bids[0][0] + self.asks[0][0]) / 2
        return 0.0

# ==================== BASE EXCHANGE CLASS ====================

class BaseExchange(BaseIntegration):
    """
    Base class for all exchange integrations
    
    This abstract base class defines the standard interface that all exchange
    integrations must implement, ensuring consistency across different exchanges.
    """
    
    def __init__(self, config: IntegrationConfig):
        """Initialize base exchange"""
        super().__init__(config)
        self.exchange_type = ExchangeType(config.name)
        
        # Exchange features
        self.supports_spot = True
        self.supports_margin = getattr(config, 'supports_margin', False)
        self.supports_futures = getattr(config, 'supports_futures', False)
        self.supports_options = getattr(config, 'supports_options', False)
        self.supports_websocket = getattr(config, 'supports_websocket', True)
        
        # Trading pairs and markets
        self.markets: Dict[str, Any] = {}
        self.symbols: List[str] = []
        self.currencies: List[str] = []
        
        # Account data
        self.balances: Dict[str, ExchangeBalance] = {}
        self.positions: Dict[str, ExchangePosition] = {}
        self.orders: Dict[str, ExchangeOrder] = {}
        
        # Market data
        self.tickers: Dict[str, ExchangeTicker] = {}
        self.orderbooks: Dict[str, ExchangeOrderBook] = {}
        
        # WebSocket connections
        self.ws_connections: Dict[str, Any] = {}
        self.ws_subscriptions: set = set()
        
        # Rate limiting
        self.rate_limit = config.rate_limit or 1200  # requests per minute
        self.request_timestamps: List[datetime] = []
    
    # ==================== ABSTRACT METHODS ====================
    
    async def load_markets(self) -> Dict[str, Any]:
        """Load available markets and trading pairs"""
        raise NotImplementedError("Subclasses must implement load_markets")
    
    async def fetch_balance(self) -> Dict[str, ExchangeBalance]:
        """Fetch account balance"""
        raise NotImplementedError("Subclasses must implement fetch_balance")
    
    async def fetch_positions(self) -> Dict[str, ExchangePosition]:
        """Fetch account positions (for margin/futures)"""
        raise NotImplementedError("Subclasses must implement fetch_positions")
    
    async def create_order(self, symbol: str, side: str, amount: float,
                          order_type: str = "market", price: Optional[float] = None,
                          **kwargs) -> ExchangeOrder:
        """Create an order"""
        raise NotImplementedError("Subclasses must implement create_order")
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        raise NotImplementedError("Subclasses must implement cancel_order")
    
    async def fetch_order(self, order_id: str, symbol: str) -> Optional[ExchangeOrder]:
        """Fetch order status"""
        raise NotImplementedError("Subclasses must implement fetch_order")
    
    async def fetch_orders(self, symbol: Optional[str] = None,
                          status: Optional[str] = None) -> List[ExchangeOrder]:
        """Fetch orders"""
        raise NotImplementedError("Subclasses must implement fetch_orders")
    
    async def fetch_ticker(self, symbol: str) -> ExchangeTicker:
        """Fetch ticker data"""
        raise NotImplementedError("Subclasses must implement fetch_ticker")
    
    async def fetch_orderbook(self, symbol: str, limit: int = 100) -> ExchangeOrderBook:
        """Fetch orderbook data"""
        raise NotImplementedError("Subclasses must implement fetch_orderbook")
    
    # ==================== COMMON HELPER METHODS ====================
    
    def check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = datetime.now()
        
        # Remove old timestamps (older than 1 minute)
        cutoff = now - timedelta(minutes=1)
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > cutoff]
        
        # Check if we're under the limit
        return len(self.request_timestamps) < self.rate_limit
    
    def record_request(self):
        """Record a new API request timestamp"""
        self.request_timestamps.append(datetime.now())
    
    def format_symbol(self, symbol: str) -> str:
        """Format symbol for exchange-specific requirements"""
        # Default implementation - override in subclasses
        return symbol.upper().replace('-', '/').replace('_', '/')
    
    def parse_symbol(self, symbol: str) -> Tuple[str, str]:
        """Parse symbol into base and quote currencies"""
        # Handle common formats: BTC/USD, BTC-USD, BTCUSD
        for sep in ['/', '-', '_']:
            if sep in symbol:
                base, quote = symbol.split(sep, 1)
                return base.upper(), quote.upper()
        
        # Try to parse concatenated symbols (BTCUSDT -> BTC/USDT)
        common_quotes = ['USDT', 'USD', 'EUR', 'BTC', 'ETH', 'BNB']
        for quote in common_quotes:
            if symbol.endswith(quote) and len(symbol) > len(quote):
                base = symbol[:-len(quote)]
                return base.upper(), quote.upper()
        
        # Default fallback
        return symbol.upper(), 'USD'
    
    def calculate_order_value(self, amount: float, price: float) -> float:
        """Calculate total order value"""
        return amount * price
    
    async def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """Get trading fees for a symbol"""
        # Default fees - override in subclasses with actual fee structure
        return {
            'maker': 0.001,  # 0.1%
            'taker': 0.001   # 0.1%
        }
    
    # ==================== WEBSOCKET HELPERS ====================
    
    async def subscribe_ticker(self, symbol: str):
        """Subscribe to ticker updates"""
        # Default implementation - override in subclasses
        self.ws_subscriptions.add(f"ticker:{symbol}")
        logger.info(f"📡 Subscribed to ticker: {symbol}")
    
    async def subscribe_orderbook(self, symbol: str):
        """Subscribe to orderbook updates"""
        # Default implementation - override in subclasses
        self.ws_subscriptions.add(f"orderbook:{symbol}")
        logger.info(f"📊 Subscribed to orderbook: {symbol}")
    
    async def subscribe_trades(self, symbol: str):
        """Subscribe to trade updates"""
        # Default implementation - override in subclasses
        self.ws_subscriptions.add(f"trades:{symbol}")
        logger.info(f"💱 Subscribed to trades: {symbol}")
    
    async def unsubscribe_all(self):
        """Unsubscribe from all WebSocket feeds"""
        self.ws_subscriptions.clear()
        for connection in self.ws_connections.values():
            if hasattr(connection, 'close'):
                await connection.close()
        self.ws_connections.clear()
        logger.info("🔇 Unsubscribed from all feeds")

# ==================== EXCHANGE FACTORY ====================

class ExchangeFactory:
    """Factory for creating exchange instances"""
    
    # Registry of available exchange classes
    _exchange_classes: Dict[ExchangeType, Type[BaseExchange]] = {}
    
    @classmethod
    def register_exchange(cls, exchange_type: ExchangeType, exchange_class: Type[BaseExchange]):
        """Register an exchange class"""
        cls._exchange_classes[exchange_type] = exchange_class
        logger.info(f"Registered exchange: {exchange_type.value}")
    
    @classmethod
    def create_exchange(cls, config: IntegrationConfig) -> Optional[BaseExchange]:
        """Create an exchange instance"""
        try:
            exchange_type = ExchangeType(config.name)
            exchange_class = cls._exchange_classes.get(exchange_type)
            
            if exchange_class:
                return exchange_class(config)
            else:
                logger.error(f"Exchange class not found: {exchange_type.value}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create exchange {config.name}: {e}")
            return None
    
    @classmethod
    def get_available_exchanges(cls) -> List[ExchangeType]:
        """Get list of available exchange types"""
        return list(cls._exchange_classes.keys())
    
    @classmethod
    def get_exchange_capabilities(cls, exchange_type: ExchangeType) -> Dict[str, Any]:
        """Get capabilities of a specific exchange"""
        exchange_class = cls._exchange_classes.get(exchange_type)
        if exchange_class:
            return {
                'supports_spot': getattr(exchange_class, 'supports_spot', True),
                'supports_margin': getattr(exchange_class, 'supports_margin', False),
                'supports_futures': getattr(exchange_class, 'supports_futures', False),
                'supports_options': getattr(exchange_class, 'supports_options', False),
                'supports_websocket': getattr(exchange_class, 'supports_websocket', True),
                'supports_paper_trading': getattr(exchange_class, 'supports_paper_trading', True)
            }
        return {}

# ==================== EXCHANGE MANAGER ====================

class ExchangeManager:
    """Manages multiple exchange connections"""
    
    def __init__(self):
        self.exchanges: Dict[str, BaseExchange] = {}
        self.active_exchange: Optional[str] = None
        
        # Cross-exchange data
        self.aggregated_balances: Dict[str, float] = {}
        self.arbitrage_opportunities: List[Dict[str, Any]] = []
        
    async def add_exchange(self, name: str, config: IntegrationConfig) -> bool:
        """Add an exchange connection"""
        try:
            exchange = ExchangeFactory.create_exchange(config)
            if exchange:
                # Test connection
                if await exchange.connect():
                    self.exchanges[name] = exchange
                    
                    # Set as active if it's the first exchange
                    if not self.active_exchange:
                        self.active_exchange = name
                        
                    logger.info(f"✅ Added exchange: {name}")
                    return True
                else:
                    logger.error(f"❌ Failed to connect exchange: {name}")
                    return False
            else:
                logger.error(f"❌ Failed to create exchange: {name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error adding exchange {name}: {e}")
            return False
    
    async def remove_exchange(self, name: str) -> bool:
        """Remove an exchange connection"""
        try:
            if name in self.exchanges:
                await self.exchanges[name].disconnect()
                del self.exchanges[name]
                
                # Update active exchange if removed
                if self.active_exchange == name:
                    self.active_exchange = next(iter(self.exchanges), None)
                
                logger.info(f"✅ Removed exchange: {name}")
                return True
            else:
                logger.warning(f"⚠️ Exchange not found: {name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error removing exchange {name}: {e}")
            return False
    
    def get_active_exchange(self) -> Optional[BaseExchange]:
        """Get the currently active exchange"""
        if self.active_exchange and self.active_exchange in self.exchanges:
            return self.exchanges[self.active_exchange]
        return None
    
    def set_active_exchange(self, name: str) -> bool:
        """Set the active exchange"""
        if name in self.exchanges:
            self.active_exchange = name
            logger.info(f"🎯 Active exchange set to: {name}")
            return True
        else:
            logger.error(f"❌ Exchange not found: {name}")
            return False
    
    async def aggregate_balances(self) -> Dict[str, float]:
        """Aggregate balances across all exchanges"""
        try:
            aggregated = {}
            
            for name, exchange in self.exchanges.items():
                if exchange.status == IntegrationStatus.CONNECTED:
                    balances = await exchange.fetch_balance()
                    
                    for currency, balance in balances.items():
                        if currency not in aggregated:
                            aggregated[currency] = 0.0
                        aggregated[currency] += balance.total
            
            self.aggregated_balances = aggregated
            return aggregated
            
        except Exception as e:
            logger.error(f"❌ Error aggregating balances: {e}")
            return {}
    
    async def find_arbitrage_opportunities(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Find arbitrage opportunities across exchanges"""
        try:
            opportunities = []
            
            # Get tickers from all connected exchanges
            exchange_tickers = {}
            for name, exchange in self.exchanges.items():
                if exchange.status == IntegrationStatus.CONNECTED:
                    tickers = {}
                    for symbol in symbols:
                        try:
                            ticker = await exchange.fetch_ticker(symbol)
                            tickers[symbol] = ticker
                        except:
                            continue
                    exchange_tickers[name] = tickers
            
            # Find price differences
            for symbol in symbols:
                prices = {}
                for exchange_name, tickers in exchange_tickers.items():
                    if symbol in tickers:
                        prices[exchange_name] = tickers[symbol].last
                
                if len(prices) >= 2:
                    min_exchange = min(prices, key=prices.get)
                    max_exchange = max(prices, key=prices.get)
                    
                    min_price = prices[min_exchange]
                    max_price = prices[max_exchange]
                    
                    if max_price > min_price:
                        profit_percent = ((max_price - min_price) / min_price) * 100
                        
                        if profit_percent > 0.5:  # Minimum 0.5% profit
                            opportunities.append({
                                'symbol': symbol,
                                'buy_exchange': min_exchange,
                                'sell_exchange': max_exchange,
                                'buy_price': min_price,
                                'sell_price': max_price,
                                'profit_percent': profit_percent,
                                'timestamp': datetime.now()
                            })
            
            # Sort by profit percentage
            opportunities.sort(key=lambda x: x['profit_percent'], reverse=True)
            self.arbitrage_opportunities = opportunities
            
            return opportunities
            
        except Exception as e:
            logger.error(f"❌ Error finding arbitrage opportunities: {e}")
            return []
    
    def get_exchange_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all exchanges"""
        status = {}
        for name, exchange in self.exchanges.items():
            status[name] = {
                'type': exchange.exchange_type.value,
                'status': exchange.status.value,
                'connected': exchange.status == IntegrationStatus.CONNECTED,
                'active': name == self.active_exchange,
                'supports_spot': exchange.supports_spot,
                'supports_margin': exchange.supports_margin,
                'supports_futures': exchange.supports_futures
            }
        return status

# ==================== DEFAULT CONFIGURATIONS ====================

def get_default_exchange_configs() -> Dict[str, IntegrationConfig]:
    """Get default exchange configurations"""
    
    configs = {
        "binance": IntegrationConfig(
            name="binance",
            integration_type=IntegrationType.EXCHANGE,
            enabled=False,  # Disabled until configured
            base_url="https://testnet.binance.vision",
            sandbox_mode=True,
            paper_trading=True,
            supports_websocket=True,
            rate_limit=1200,
            description="Binance - World's largest cryptocurrency exchange"
        ),
        
        "coinbase": IntegrationConfig(
            name="coinbase",
            integration_type=IntegrationType.EXCHANGE,
            enabled=False,  # Disabled until configured
            base_url="https://api-public.sandbox.pro.coinbase.com",
            sandbox_mode=True,
            paper_trading=True,
            supports_websocket=True,
            rate_limit=10,  # per second
            description="Coinbase Pro - Professional cryptocurrency trading"
        ),
        
        "kraken": IntegrationConfig(
            name="kraken",
            integration_type=IntegrationType.EXCHANGE,
            enabled=False,  # Disabled until configured
            base_url="https://api.kraken.com",
            sandbox_mode=False,  # Kraken doesn't have sandbox
            paper_trading=True,
            supports_websocket=True,
            rate_limit=60,
            description="Kraken - Secure and reliable crypto exchange"
        )
    }
    
    return configs

# ==================== INITIALIZATION ====================

def register_exchanges():
    """Register all available exchange implementations"""
    try:
        # Import exchange implementations
        try:
            from .binance import BinanceExchange
            ExchangeFactory.register_exchange(ExchangeType.BINANCE, BinanceExchange)
        except ImportError:
            logger.debug("Binance exchange not available")
        
        try:
            from .coinbase import CoinbaseExchange
            ExchangeFactory.register_exchange(ExchangeType.COINBASE, CoinbaseExchange)
        except ImportError:
            logger.debug("Coinbase exchange not available")
        
        try:
            from .kraken import KrakenExchange
            ExchangeFactory.register_exchange(ExchangeType.KRAKEN, KrakenExchange)
        except ImportError:
            logger.debug("Kraken exchange not available")
            
        logger.info(f"✅ Registered {len(ExchangeFactory.get_available_exchanges())} exchange(s)")
        
    except Exception as e:
        logger.error(f"❌ Error registering exchanges: {e}")

# Register exchanges on import
register_exchanges()

# ==================== EXPORTS ====================

__all__ = [
    # Enums
    'ExchangeType',
    'OrderType',
    'OrderSide', 
    'OrderStatus',
    'TradingMode',
    
    # Data structures
    'ExchangeBalance',
    'ExchangePosition',
    'ExchangeOrder',
    'ExchangeTicker',
    'ExchangeOrderBook',
    
    # Classes
    'BaseExchange',
    'ExchangeFactory',
    'ExchangeManager',
    
    # Functions
    'get_default_exchange_configs',
    'register_exchanges'
]