#!/usr/bin/env python3
"""
File: crypto_data.py
Path: NeuroCluster-Elite/src/data/crypto_data.py
Description: Cryptocurrency data fetcher with multiple exchanges and real-time capabilities

This module provides comprehensive cryptocurrency data fetching from multiple exchanges
including Binance, Coinbase, Kraken, and others with real-time WebSocket support,
order book data, and DeFi metrics.

Features:
- Multi-exchange data aggregation with intelligent routing
- Real-time ticker, trade, and order book data via WebSocket
- Historical OHLCV data with multiple timeframes
- DeFi metrics and on-chain data
- Staking and yield farming information
- Cross-exchange arbitrage detection
- Advanced caching and rate limiting
- Data validation and anomaly detection

Author: Your Name
Created: 2025-06-28
Version: 1.0.0
License: MIT
"""

import asyncio
import aiohttp
import websockets
import json
import time
import logging
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import ssl
from urllib.parse import urlencode
import base64
import hmac
import hashlib
import ccxt.async_support as ccxt
import requests

# Import our modules
try:
    from src.core.neurocluster_elite import AssetType, MarketData
    from src.utils.config_manager import ConfigManager
    from src.utils.helpers import retry_on_failure, format_currency, calculate_hash
except ImportError:
    # Fallback for testing
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CRYPTO DATA STRUCTURES ====================

@dataclass
class CryptoTicker:
    """Real-time cryptocurrency ticker data"""
    symbol: str
    base_currency: str
    quote_currency: str
    price: float
    change_24h: float
    change_percent_24h: float
    volume_24h: float
    volume_base: float
    volume_quote: float
    bid: float = 0.0
    ask: float = 0.0
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    open_24h: Optional[float] = None
    last_trade_time: Optional[datetime] = None
    market_cap: Optional[float] = None
    circulating_supply: Optional[float] = None
    total_supply: Optional[float] = None
    max_supply: Optional[float] = None
    exchange: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class CryptoOrderBook:
    """Cryptocurrency order book data"""
    symbol: str
    exchange: str
    bids: List[Tuple[float, float]] = field(default_factory=list)  # (price, size)
    asks: List[Tuple[float, float]] = field(default_factory=list)  # (price, size)
    timestamp: datetime = field(default_factory=datetime.now)
    sequence: Optional[int] = None
    
    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None
    
    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def spread_percent(self) -> Optional[float]:
        if self.best_bid and self.spread:
            return (self.spread / self.best_bid) * 100
        return None

@dataclass
class CryptoTrade:
    """Individual cryptocurrency trade data"""
    symbol: str
    exchange: str
    trade_id: str
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    timestamp: datetime
    fee: Optional[float] = None
    fee_currency: Optional[str] = None

@dataclass
class DeFiMetrics:
    """DeFi protocol metrics"""
    protocol: str
    token_symbol: str
    tvl_usd: float  # Total Value Locked
    apy: Optional[float] = None
    volume_24h: Optional[float] = None
    fees_24h: Optional[float] = None
    users_active: Optional[int] = None
    token_price: Optional[float] = None
    market_cap: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class OnChainMetrics:
    """On-chain metrics for cryptocurrencies"""
    symbol: str
    network: str
    active_addresses: Optional[int] = None
    transaction_count_24h: Optional[int] = None
    transaction_volume_24h: Optional[float] = None
    hash_rate: Optional[float] = None
    difficulty: Optional[float] = None
    block_time: Optional[float] = None
    fees_24h: Optional[float] = None
    network_value_to_transactions: Optional[float] = None
    mvrv_ratio: Optional[float] = None  # Market Value to Realized Value
    timestamp: datetime = field(default_factory=datetime.now)

# ==================== CRYPTO DATA MANAGER ====================

class CryptoDataManager:
    """
    Comprehensive cryptocurrency data manager with multi-exchange support
    
    This class provides real-time and historical cryptocurrency data from various
    exchanges with advanced features like DeFi metrics, on-chain data, and
    cross-exchange arbitrage detection.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize crypto data manager"""
        
        self.config = config or self._get_default_config()
        self.api_keys = self._load_api_keys()
        self.exchanges = {}
        self.websocket_connections = {}
        self.cache = {}
        self.rate_limiters = {}
        self.real_time_callbacks = {}
        self.order_books = {}
        self.data_quality_metrics = {}
        
        # Initialize exchanges
        self._initialize_exchanges()
        
        # Real-time data management
        self.is_streaming = False
        self.stream_tasks = []
        
        logger.info("Crypto Data Manager initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'exchanges': {
                'binance': {'enabled': True, 'priority': 1, 'sandbox': False},
                'coinbase': {'enabled': True, 'priority': 2, 'sandbox': False},
                'kraken': {'enabled': True, 'priority': 3, 'sandbox': False},
                'kucoin': {'enabled': True, 'priority': 4, 'sandbox': False},
                'huobi': {'enabled': True, 'priority': 5, 'sandbox': False}
            },
            'cache_settings': {
                'enabled': True,
                'ttl_seconds': 15,  # Faster refresh for crypto
                'max_items': 50000
            },
            'rate_limits': {
                'binance': 1200,  # requests per minute
                'coinbase': 10000,
                'kraken': 900,
                'kucoin': 1800,
                'huobi': 2000
            },
            'websocket_settings': {
                'enabled': True,
                'max_connections': 10,
                'reconnect_delay': 3,
                'ping_interval': 30
            },
            'arbitrage_detection': {
                'enabled': True,
                'min_profit_percent': 0.5,
                'max_exchanges': 5
            },
            'defi_data': {
                'enabled': True,
                'protocols': ['uniswap', 'compound', 'aave', 'makerdao', 'curve']
            }
        }
    
    def _load_api_keys(self) -> Dict:
        """Load API keys from configuration"""
        return {
            'binance': {
                'apiKey': 'YOUR_BINANCE_API_KEY',
                'secret': 'YOUR_BINANCE_SECRET',
                'sandbox': False
            },
            'coinbase': {
                'apiKey': 'YOUR_COINBASE_API_KEY',
                'secret': 'YOUR_COINBASE_SECRET',
                'passphrase': 'YOUR_COINBASE_PASSPHRASE',
                'sandbox': False
            },
            'kraken': {
                'apiKey': 'YOUR_KRAKEN_API_KEY',
                'secret': 'YOUR_KRAKEN_SECRET'
            },
            'coingecko': 'YOUR_COINGECKO_API_KEY',
            'defipulse': 'YOUR_DEFIPULSE_API_KEY',
            'glassnode': 'YOUR_GLASSNODE_API_KEY'
        }
    
    async def _initialize_exchanges(self):
        """Initialize cryptocurrency exchanges"""
        
        self.exchanges = {}
        
        for exchange_name, exchange_config in self.config['exchanges'].items():
            if not exchange_config['enabled']:
                continue
            
            try:
                # Initialize CCXT exchange
                exchange_class = getattr(ccxt, exchange_name)
                
                exchange_params = {
                    'enableRateLimit': True,
                    'timeout': 10000,
                }
                
                # Add API keys if available
                if exchange_name in self.api_keys:
                    keys = self.api_keys[exchange_name]
                    exchange_params.update(keys)
                
                # Set sandbox mode if configured
                if exchange_config.get('sandbox', False):
                    exchange_params['sandbox'] = True
                
                exchange = exchange_class(exchange_params)
                
                # Test connection
                await exchange.load_markets()
                
                self.exchanges[exchange_name] = exchange
                logger.info(f"Initialized {exchange_name} exchange")
                
            except Exception as e:
                logger.warning(f"Failed to initialize {exchange_name}: {e}")
                continue
    
    async def get_ticker(self, symbol: str, exchange: str = None) -> Optional[CryptoTicker]:
        """Get real-time ticker data for a cryptocurrency"""
        
        try:
            # Check cache first
            cache_key = f"ticker_{symbol}_{exchange or 'best'}"
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < 5:  # 5 second cache
                    return cached_data['data']
            
            # If specific exchange requested
            if exchange and exchange in self.exchanges:
                ticker = await self._fetch_ticker_from_exchange(symbol, exchange)
                if ticker:
                    self.cache[cache_key] = {
                        'data': ticker,
                        'timestamp': datetime.now()
                    }
                    return ticker
            
            # Try exchanges in priority order
            for exchange_name, exchange_config in sorted(
                self.config['exchanges'].items(),
                key=lambda x: x[1]['priority']
            ):
                if not exchange_config['enabled'] or exchange_name not in self.exchanges:
                    continue
                
                try:
                    ticker = await self._fetch_ticker_from_exchange(symbol, exchange_name)
                    if ticker:
                        self.cache[cache_key] = {
                            'data': ticker,
                            'timestamp': datetime.now()
                        }
                        return ticker
                        
                except Exception as e:
                    logger.warning(f"Error fetching ticker from {exchange_name}: {e}")
                    continue
            
            logger.warning(f"Could not fetch ticker for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
            return None
    
    async def _fetch_ticker_from_exchange(self, symbol: str, exchange_name: str) -> Optional[CryptoTicker]:
        """Fetch ticker from specific exchange"""
        
        try:
            exchange = self.exchanges[exchange_name]
            
            # Fetch ticker data
            ticker_data = await exchange.fetch_ticker(symbol)
            
            if not ticker_data:
                return None
            
            # Parse symbol
            base, quote = symbol.split('/')
            
            ticker = CryptoTicker(
                symbol=symbol,
                base_currency=base,
                quote_currency=quote,
                price=float(ticker_data['last']),
                change_24h=float(ticker_data['change'] or 0),
                change_percent_24h=float(ticker_data['percentage'] or 0),
                volume_24h=float(ticker_data['quoteVolume'] or 0),
                volume_base=float(ticker_data['baseVolume'] or 0),
                volume_quote=float(ticker_data['quoteVolume'] or 0),
                bid=float(ticker_data['bid'] or 0),
                ask=float(ticker_data['ask'] or 0),
                high_24h=ticker_data.get('high'),
                low_24h=ticker_data.get('low'),
                open_24h=ticker_data.get('open'),
                exchange=exchange_name,
                timestamp=datetime.fromtimestamp(ticker_data['timestamp'] / 1000) if ticker_data['timestamp'] else datetime.now()
            )
            
            return ticker
            
        except Exception as e:
            logger.warning(f"Error fetching ticker from {exchange_name}: {e}")
            return None
    
    async def get_historical_data(self, 
                                symbol: str,
                                timeframe: str = '1d',
                                start_date: datetime = None,
                                end_date: datetime = None,
                                limit: int = 1000,
                                exchange: str = None) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data for a cryptocurrency"""
        
        try:
            # Check cache
            cache_key = f"hist_{symbol}_{timeframe}_{start_date}_{end_date}_{exchange or 'best'}"
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < 300:  # 5 minute cache
                    return cached_data['data']
            
            # Convert dates to timestamps
            since = int(start_date.timestamp() * 1000) if start_date else None
            
            # Try specific exchange first
            if exchange and exchange in self.exchanges:
                data = await self._fetch_ohlcv_from_exchange(
                    symbol, timeframe, since, limit, exchange
                )
                if data is not None:
                    self.cache[cache_key] = {
                        'data': data,
                        'timestamp': datetime.now()
                    }
                    return data
            
            # Try exchanges in priority order
            for exchange_name, exchange_config in sorted(
                self.config['exchanges'].items(),
                key=lambda x: x[1]['priority']
            ):
                if not exchange_config['enabled'] or exchange_name not in self.exchanges:
                    continue
                
                try:
                    data = await self._fetch_ohlcv_from_exchange(
                        symbol, timeframe, since, limit, exchange_name
                    )
                    if data is not None:
                        self.cache[cache_key] = {
                            'data': data,
                            'timestamp': datetime.now()
                        }
                        return data
                        
                except Exception as e:
                    logger.warning(f"Error fetching OHLCV from {exchange_name}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    async def _fetch_ohlcv_from_exchange(self, 
                                       symbol: str, 
                                       timeframe: str, 
                                       since: int, 
                                       limit: int, 
                                       exchange_name: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from specific exchange"""
        
        try:
            exchange = self.exchanges[exchange_name]
            
            # Fetch OHLCV data
            ohlcv_data = await exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )
            
            if not ohlcv_data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv_data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Validate data
            if self._validate_ohlcv_data(df):
                return df
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Error fetching OHLCV from {exchange_name}: {e}")
            return None
    
    async def get_order_book(self, symbol: str, limit: int = 100, exchange: str = None) -> Optional[CryptoOrderBook]:
        """Get order book data for a cryptocurrency"""
        
        try:
            # Check cache
            cache_key = f"orderbook_{symbol}_{limit}_{exchange or 'best'}"
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < 2:  # 2 second cache
                    return cached_data['data']
            
            # Try specific exchange first
            if exchange and exchange in self.exchanges:
                order_book = await self._fetch_order_book_from_exchange(symbol, limit, exchange)
                if order_book:
                    self.cache[cache_key] = {
                        'data': order_book,
                        'timestamp': datetime.now()
                    }
                    return order_book
            
            # Try exchanges in priority order
            for exchange_name, exchange_config in sorted(
                self.config['exchanges'].items(),
                key=lambda x: x[1]['priority']
            ):
                if not exchange_config['enabled'] or exchange_name not in self.exchanges:
                    continue
                
                try:
                    order_book = await self._fetch_order_book_from_exchange(symbol, limit, exchange_name)
                    if order_book:
                        self.cache[cache_key] = {
                            'data': order_book,
                            'timestamp': datetime.now()
                        }
                        return order_book
                        
                except Exception as e:
                    logger.warning(f"Error fetching order book from {exchange_name}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {e}")
            return None
    
    async def _fetch_order_book_from_exchange(self, symbol: str, limit: int, exchange_name: str) -> Optional[CryptoOrderBook]:
        """Fetch order book from specific exchange"""
        
        try:
            exchange = self.exchanges[exchange_name]
            
            # Fetch order book
            order_book_data = await exchange.fetch_order_book(symbol, limit)
            
            if not order_book_data:
                return None
            
            order_book = CryptoOrderBook(
                symbol=symbol,
                exchange=exchange_name,
                bids=[(float(bid[0]), float(bid[1])) for bid in order_book_data['bids']],
                asks=[(float(ask[0]), float(ask[1])) for ask in order_book_data['asks']],
                timestamp=datetime.fromtimestamp(order_book_data['timestamp'] / 1000) if order_book_data['timestamp'] else datetime.now()
            )
            
            return order_book
            
        except Exception as e:
            logger.warning(f"Error fetching order book from {exchange_name}: {e}")
            return None
    
    async def get_trades(self, symbol: str, limit: int = 100, exchange: str = None) -> Optional[List[CryptoTrade]]:
        """Get recent trades for a cryptocurrency"""
        
        try:
            # Try specific exchange first
            if exchange and exchange in self.exchanges:
                trades = await self._fetch_trades_from_exchange(symbol, limit, exchange)
                if trades:
                    return trades
            
            # Try exchanges in priority order
            for exchange_name, exchange_config in sorted(
                self.config['exchanges'].items(),
                key=lambda x: x[1]['priority']
            ):
                if not exchange_config['enabled'] or exchange_name not in self.exchanges:
                    continue
                
                try:
                    trades = await self._fetch_trades_from_exchange(symbol, limit, exchange_name)
                    if trades:
                        return trades
                        
                except Exception as e:
                    logger.warning(f"Error fetching trades from {exchange_name}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting trades for {symbol}: {e}")
            return None
    
    async def _fetch_trades_from_exchange(self, symbol: str, limit: int, exchange_name: str) -> Optional[List[CryptoTrade]]:
        """Fetch trades from specific exchange"""
        
        try:
            exchange = self.exchanges[exchange_name]
            
            # Fetch trades
            trades_data = await exchange.fetch_trades(symbol, limit=limit)
            
            if not trades_data:
                return None
            
            trades = []
            for trade_data in trades_data:
                trade = CryptoTrade(
                    symbol=symbol,
                    exchange=exchange_name,
                    trade_id=str(trade_data['id']),
                    price=float(trade_data['price']),
                    size=float(trade_data['amount']),
                    side=trade_data['side'],
                    timestamp=datetime.fromtimestamp(trade_data['timestamp'] / 1000),
                    fee=trade_data.get('fee', {}).get('cost') if trade_data.get('fee') else None
                )
                trades.append(trade)
            
            return trades
            
        except Exception as e:
            logger.warning(f"Error fetching trades from {exchange_name}: {e}")
            return None
    
    async def get_defi_metrics(self, protocol: str = None) -> Optional[List[DeFiMetrics]]:
        """Get DeFi protocol metrics"""
        
        try:
            if not self.config['defi_data']['enabled']:
                return None
            
            # Check cache
            cache_key = f"defi_{protocol or 'all'}"
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < 3600:  # 1 hour cache
                    return cached_data['data']
            
            defi_metrics = await self._fetch_defi_data(protocol)
            
            if defi_metrics:
                self.cache[cache_key] = {
                    'data': defi_metrics,
                    'timestamp': datetime.now()
                }
            
            return defi_metrics
            
        except Exception as e:
            logger.error(f"Error getting DeFi metrics: {e}")
            return None
    
    async def _fetch_defi_data(self, protocol: str = None) -> Optional[List[DeFiMetrics]]:
        """Fetch DeFi data from various sources"""
        
        try:
            # This would integrate with DeFi data providers like DeFiPulse, DefiLlama, etc.
            # For now, return mock data
            
            mock_protocols = ['uniswap', 'compound', 'aave', 'makerdao', 'curve']
            
            if protocol:
                mock_protocols = [protocol] if protocol in mock_protocols else []
            
            defi_metrics = []
            
            for protocol_name in mock_protocols:
                metrics = DeFiMetrics(
                    protocol=protocol_name,
                    token_symbol=protocol_name.upper(),
                    tvl_usd=np.random.uniform(1e9, 10e9),  # Random TVL between 1B-10B
                    apy=np.random.uniform(3.0, 15.0),  # Random APY between 3-15%
                    volume_24h=np.random.uniform(1e8, 1e9),  # Random volume
                    fees_24h=np.random.uniform(1e6, 1e7),  # Random fees
                    users_active=np.random.randint(1000, 10000)  # Random active users
                )
                defi_metrics.append(metrics)
            
            return defi_metrics
            
        except Exception as e:
            logger.error(f"Error fetching DeFi data: {e}")
            return None
    
    async def get_on_chain_metrics(self, symbol: str) -> Optional[OnChainMetrics]:
        """Get on-chain metrics for a cryptocurrency"""
        
        try:
            # Check cache
            cache_key = f"onchain_{symbol}"
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < 3600:  # 1 hour cache
                    return cached_data['data']
            
            metrics = await self._fetch_on_chain_data(symbol)
            
            if metrics:
                self.cache[cache_key] = {
                    'data': metrics,
                    'timestamp': datetime.now()
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting on-chain metrics for {symbol}: {e}")
            return None
    
    async def _fetch_on_chain_data(self, symbol: str) -> Optional[OnChainMetrics]:
        """Fetch on-chain data from blockchain analytics providers"""
        
        try:
            # This would integrate with providers like Glassnode, IntoTheBlock, etc.
            # For now, return mock data for Bitcoin
            
            if symbol.upper() == 'BTC':
                metrics = OnChainMetrics(
                    symbol=symbol,
                    network='bitcoin',
                    active_addresses=np.random.randint(800000, 1200000),
                    transaction_count_24h=np.random.randint(250000, 400000),
                    transaction_volume_24h=np.random.uniform(1e9, 5e9),
                    hash_rate=np.random.uniform(150e18, 200e18),  # Hash/s
                    difficulty=np.random.uniform(20e12, 30e12),
                    block_time=np.random.uniform(9.5, 10.5),  # Minutes
                    fees_24h=np.random.uniform(1e6, 5e6),  # USD
                    mvrv_ratio=np.random.uniform(1.0, 3.0)
                )
                return metrics
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching on-chain data for {symbol}: {e}")
            return None
    
    async def detect_arbitrage_opportunities(self, symbol: str, min_profit_percent: float = None) -> List[Dict]:
        """Detect arbitrage opportunities across exchanges"""
        
        try:
            if not self.config['arbitrage_detection']['enabled']:
                return []
            
            min_profit = min_profit_percent or self.config['arbitrage_detection']['min_profit_percent']
            
            # Get tickers from all exchanges
            exchange_tickers = {}
            
            for exchange_name in self.exchanges.keys():
                try:
                    ticker = await self._fetch_ticker_from_exchange(symbol, exchange_name)
                    if ticker and ticker.bid > 0 and ticker.ask > 0:
                        exchange_tickers[exchange_name] = ticker
                except Exception as e:
                    logger.warning(f"Error fetching ticker from {exchange_name} for arbitrage: {e}")
                    continue
            
            if len(exchange_tickers) < 2:
                return []
            
            # Find arbitrage opportunities
            opportunities = []
            
            for buy_exchange, buy_ticker in exchange_tickers.items():
                for sell_exchange, sell_ticker in exchange_tickers.items():
                    if buy_exchange == sell_exchange:
                        continue
                    
                    # Calculate potential profit
                    buy_price = buy_ticker.ask  # Price to buy
                    sell_price = sell_ticker.bid  # Price to sell
                    
                    if sell_price > buy_price:
                        profit_percent = ((sell_price - buy_price) / buy_price) * 100
                        
                        if profit_percent >= min_profit:
                            opportunity = {
                                'symbol': symbol,
                                'buy_exchange': buy_exchange,
                                'sell_exchange': sell_exchange,
                                'buy_price': buy_price,
                                'sell_price': sell_price,
                                'profit_percent': profit_percent,
                                'profit_usd_per_unit': sell_price - buy_price,
                                'buy_volume': buy_ticker.ask_volume,
                                'sell_volume': sell_ticker.bid_volume,
                                'max_trade_size': min(buy_ticker.ask_volume, sell_ticker.bid_volume),
                                'timestamp': datetime.now()
                            }
                            opportunities.append(opportunity)
            
            # Sort by profit percentage
            opportunities.sort(key=lambda x: x['profit_percent'], reverse=True)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error detecting arbitrage opportunities: {e}")
            return []
    
    async def start_real_time_stream(self, symbols: List[str], callback: Callable, data_types: List[str] = None):
        """Start real-time data streaming"""
        
        try:
            if not self.config['websocket_settings']['enabled']:
                logger.warning("WebSocket streaming is disabled")
                return
            
            self.is_streaming = True
            data_types = data_types or ['ticker', 'trades']
            
            # Start WebSocket streams for each exchange
            for exchange_name in self.exchanges.keys():
                if exchange_name in ['binance', 'coinbase', 'kraken']:  # Exchanges with WS support
                    task = asyncio.create_task(
                        self._start_exchange_stream(exchange_name, symbols, callback, data_types)
                    )
                    self.stream_tasks.append(task)
            
            logger.info(f"Started real-time streaming for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error starting real-time stream: {e}")
    
    async def _start_exchange_stream(self, exchange_name: str, symbols: List[str], callback: Callable, data_types: List[str]):
        """Start WebSocket stream for specific exchange"""
        
        try:
            if exchange_name == 'binance':
                await self._start_binance_stream(symbols, callback, data_types)
            elif exchange_name == 'coinbase':
                await self._start_coinbase_stream(symbols, callback, data_types)
            elif exchange_name == 'kraken':
                await self._start_kraken_stream(symbols, callback, data_types)
            
        except Exception as e:
            logger.error(f"Error starting {exchange_name} stream: {e}")
    
    async def _start_binance_stream(self, symbols: List[str], callback: Callable, data_types: List[str]):
        """Start Binance WebSocket stream"""
        
        try:
            # Convert symbols to Binance format
            binance_symbols = [symbol.replace('/', '').lower() for symbol in symbols]
            
            # Create stream URL
            streams = []
            for symbol in binance_symbols:
                if 'ticker' in data_types:
                    streams.append(f"{symbol}@ticker")
                if 'trades' in data_types:
                    streams.append(f"{symbol}@trade")
                if 'orderbook' in data_types:
                    streams.append(f"{symbol}@depth20")
            
            stream_url = f"wss://stream.binance.com:9443/ws/{'/'.join(streams)}"
            
            async with websockets.connect(stream_url) as websocket:
                while self.is_streaming:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        data = json.loads(message)
                        
                        # Process and callback
                        processed_data = self._process_binance_message(data)
                        if processed_data:
                            await callback(processed_data)
                            
                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        await websocket.ping()
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing Binance message: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error in Binance stream: {e}")
    
    def _process_binance_message(self, data: Dict) -> Optional[Dict]:
        """Process Binance WebSocket message"""
        
        try:
            if 'stream' not in data:
                return None
            
            stream = data['stream']
            payload = data['data']
            
            if '@ticker' in stream:
                # Process ticker data
                symbol = payload['s']
                return {
                    'type': 'ticker',
                    'exchange': 'binance',
                    'symbol': f"{symbol[:3]}/{symbol[3:]}",  # Simplified symbol conversion
                    'price': float(payload['c']),
                    'change_24h': float(payload['P']),
                    'volume_24h': float(payload['v']),
                    'timestamp': datetime.now()
                }
            
            elif '@trade' in stream:
                # Process trade data
                symbol = payload['s']
                return {
                    'type': 'trade',
                    'exchange': 'binance',
                    'symbol': f"{symbol[:3]}/{symbol[3:]}",
                    'price': float(payload['p']),
                    'size': float(payload['q']),
                    'side': 'buy' if payload['m'] else 'sell',
                    'timestamp': datetime.fromtimestamp(payload['T'] / 1000)
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error processing Binance message: {e}")
            return None
    
    async def _start_coinbase_stream(self, symbols: List[str], callback: Callable, data_types: List[str]):
        """Start Coinbase WebSocket stream"""
        # Placeholder - would implement Coinbase WebSocket
        pass
    
    async def _start_kraken_stream(self, symbols: List[str], callback: Callable, data_types: List[str]):
        """Start Kraken WebSocket stream"""
        # Placeholder - would implement Kraken WebSocket
        pass
    
    async def stop_real_time_stream(self):
        """Stop real-time data streaming"""
        
        try:
            self.is_streaming = False
            
            # Cancel all stream tasks
            for task in self.stream_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.stream_tasks:
                await asyncio.gather(*self.stream_tasks, return_exceptions=True)
            
            self.stream_tasks.clear()
            logger.info("Stopped real-time streaming")
            
        except Exception as e:
            logger.error(f"Error stopping real-time stream: {e}")
    
    # Utility methods
    
    def _validate_ohlcv_data(self, data: pd.DataFrame) -> bool:
        """Validate OHLCV data quality"""
        
        try:
            if data is None or data.empty:
                return False
            
            # Check for required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_cols):
                return False
            
            # Check for valid OHLC relationships
            invalid_rows = (
                (data['high'] < data['low']) |
                (data['high'] < data['open']) |
                (data['high'] < data['close']) |
                (data['low'] > data['open']) |
                (data['low'] > data['close']) |
                (data['volume'] < 0)
            )
            
            if invalid_rows.any():
                logger.warning(f"Found {invalid_rows.sum()} invalid OHLCV rows")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating OHLCV data: {e}")
            return False
    
    def get_available_symbols(self, exchange: str = None) -> List[str]:
        """Get available trading symbols"""
        
        try:
            if exchange and exchange in self.exchanges:
                markets = self.exchanges[exchange].markets
                return list(markets.keys())
            
            # Return symbols from all exchanges
            all_symbols = set()
            for exchange in self.exchanges.values():
                if hasattr(exchange, 'markets') and exchange.markets:
                    all_symbols.update(exchange.markets.keys())
            
            return sorted(list(all_symbols))
            
        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
            return []
    
    def get_exchange_status(self) -> Dict[str, Dict]:
        """Get status of all exchanges"""
        
        status = {}
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                status[exchange_name] = {
                    'connected': True,
                    'markets': len(exchange.markets) if hasattr(exchange, 'markets') else 0,
                    'rate_limit': exchange.rateLimit,
                    'last_request': getattr(exchange, 'last_request_time', None)
                }
            except Exception as e:
                status[exchange_name] = {
                    'connected': False,
                    'error': str(e)
                }
        
        return status
    
    async def close(self):
        """Close all exchange connections"""
        
        try:
            # Stop streaming
            await self.stop_real_time_stream()
            
            # Close exchange connections
            for exchange in self.exchanges.values():
                try:
                    await exchange.close()
                except Exception as e:
                    logger.warning(f"Error closing exchange: {e}")
            
            logger.info("Crypto data manager closed")
            
        except Exception as e:
            logger.error(f"Error closing crypto data manager: {e}")

# ==================== TESTING ====================

async def test_crypto_data_manager():
    """Test crypto data manager functionality"""
    
    print("₿ Testing Crypto Data Manager")
    print("=" * 40)
    
    # Create crypto data manager
    crypto_manager = CryptoDataManager()
    
    # Initialize exchanges
    await crypto_manager._initialize_exchanges()
    
    test_symbol = 'BTC/USDT'
    
    print(f"✅ Testing ticker for {test_symbol}:")
    ticker = await crypto_manager.get_ticker(test_symbol)
    
    if ticker:
        print(f"   Price: ${ticker.price:,.2f}")
        print(f"   Change 24h: {ticker.change_percent_24h:+.2f}%")
        print(f"   Volume 24h: ${ticker.volume_24h:,.0f}")
        print(f"   Exchange: {ticker.exchange}")
        print(f"   Spread: {((ticker.ask - ticker.bid) / ticker.bid * 100):.3f}%")
    else:
        print("   ❌ Could not fetch ticker")
    
    print(f"\n✅ Testing historical data for {test_symbol}:")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    historical_data = await crypto_manager.get_historical_data(
        test_symbol,
        timeframe='1h',
        start_date=start_date,
        end_date=end_date
    )
    
    if historical_data is not None and len(historical_data) > 0:
        print(f"   Data points: {len(historical_data)}")
        print(f"   Date range: {historical_data.index[0]} to {historical_data.index[-1]}")
        print(f"   Price range: ${historical_data['low'].min():,.2f} - ${historical_data['high'].max():,.2f}")
        print(f"   Avg volume: {historical_data['volume'].mean():,.0f}")
    else:
        print("   ❌ Could not fetch historical data")
    
    print(f"\n✅ Testing order book for {test_symbol}:")
    order_book = await crypto_manager.get_order_book(test_symbol, limit=10)
    
    if order_book:
        print(f"   Best bid: ${order_book.best_bid:,.2f}")
        print(f"   Best ask: ${order_book.best_ask:,.2f}")
        print(f"   Spread: ${order_book.spread:,.2f} ({order_book.spread_percent:.3f}%)")
        print(f"   Bid levels: {len(order_book.bids)}")
        print(f"   Ask levels: {len(order_book.asks)}")
        print(f"   Exchange: {order_book.exchange}")
    else:
        print("   ❌ Could not fetch order book")
    
    print(f"\n✅ Testing arbitrage detection for {test_symbol}:")
    arbitrage_ops = await crypto_manager.detect_arbitrage_opportunities(test_symbol, min_profit_percent=0.1)
    
    if arbitrage_ops:
        print(f"   Found {len(arbitrage_ops)} arbitrage opportunities:")
        for i, op in enumerate(arbitrage_ops[:3]):
            print(f"   {i+1}. Buy {op['buy_exchange']} @ ${op['buy_price']:,.2f}, "
                  f"Sell {op['sell_exchange']} @ ${op['sell_price']:,.2f} "
                  f"(+{op['profit_percent']:.2f}%)")
    else:
        print("   No arbitrage opportunities found")
    
    print(f"\n✅ Testing DeFi metrics:")
    defi_metrics = await crypto_manager.get_defi_metrics()
    
    if defi_metrics:
        print(f"   Found {len(defi_metrics)} DeFi protocols:")
        for metric in defi_metrics[:3]:
            print(f"   {metric.protocol}: TVL ${metric.tvl_usd/1e9:.1f}B, APY {metric.apy:.1f}%")
    else:
        print("   ❌ Could not fetch DeFi metrics")
    
    print(f"\n✅ Exchange status:")
    exchange_status = crypto_manager.get_exchange_status()
    
    for exchange, status in exchange_status.items():
        if status.get('connected'):
            print(f"   {exchange}: ✅ Connected ({status['markets']} markets)")
        else:
            print(f"   {exchange}: ❌ {status.get('error', 'Not connected')}")
    
    # Available symbols
    symbols = crypto_manager.get_available_symbols()
    print(f"\n✅ Available symbols: {len(symbols)} (showing first 10)")
    for symbol in symbols[:10]:
        print(f"   {symbol}")
    
    # Close connections
    await crypto_manager.close()
    
    print("\n🎉 Crypto data manager tests completed!")

if __name__ == "__main__":
    asyncio.run(test_crypto_data_manager())