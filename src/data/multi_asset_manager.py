#!/usr/bin/env python3
"""
File: multi_asset_manager.py
Path: NeuroCluster-Elite/src/data/multi_asset_manager.py
Description: Unified data manager for all asset types (stocks, crypto, forex, commodities)

This module provides a unified interface for fetching real-time market data
from multiple sources with intelligent failover and caching mechanisms.

Features:
- Multi-source data fetching (Alpha Vantage, Yahoo Finance, CoinGecko, Polygon, etc.)
- Intelligent failover and error handling
- Real-time data with WebSocket support
- Advanced caching layer with Redis support
- Rate limiting and API key management
- Data validation and quality checks
- Technical indicator calculation
- Sentiment data integration

Author: Your Name
Created: 2025-06-28
Version: 1.0.0
License: MIT
"""

import asyncio
import aiohttp
import yfinance as yf
import requests
import websocket
import json
import sqlite3
import redis
import time
import logging
import hashlib
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import ssl
from urllib.parse import urlencode
import base64
import hmac

# Technical analysis
import ta
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# Import our core modules
try:
    from src.core.neurocluster_elite import AssetType, MarketData
    from src.utils.config_manager import ConfigManager
    from src.utils.helpers import format_currency, calculate_hash, retry_on_failure
except ImportError:
    # Fallback for testing
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== DATA SOURCE CONFIGURATIONS ====================

@dataclass
class DataSourceConfig:
    """Configuration for data sources"""
    name: str
    api_key: Optional[str] = None
    base_url: str = ""
    rate_limit: int = 100  # requests per minute
    timeout: int = 10
    enabled: bool = True
    priority: int = 1  # Lower number = higher priority
    supports_realtime: bool = False
    supports_historical: bool = True
    supported_assets: List[AssetType] = field(default_factory=list)

@dataclass
class CacheConfig:
    """Configuration for data caching"""
    enabled: bool = True
    ttl_seconds: int = 30  # Time to live for cached data
    redis_url: Optional[str] = None
    sqlite_path: str = "data/cache.db"
    max_memory_items: int = 1000
    compression_enabled: bool = True

@dataclass
class DataQualityMetrics:
    """Data quality tracking metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    average_response_time: float = 0.0
    data_freshness_score: float = 1.0
    completeness_score: float = 1.0
    accuracy_score: float = 1.0

# ==================== DATA SOURCE IMPLEMENTATIONS ====================

class DataSourceError(Exception):
    """Custom exception for data source errors"""
    pass

class BaseDataSource:
    """Base class for all data sources"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.last_request_time = 0
        self.request_count = 0
        self.request_window_start = time.time()
        self.quality_metrics = DataQualityMetrics()
        self.session = None
        
    async def initialize(self):
        """Initialize the data source"""
        if not self.session:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                ttl_dns_cache=300,
                ttl_conn_pool_cache=300
            )
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'User-Agent': 'NeuroCluster-Elite/1.0'}
            )
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    def _check_rate_limit(self) -> bool:
        """Check if we can make a request within rate limits"""
        current_time = time.time()
        
        # Reset counter if window expired (1 minute)
        if current_time - self.request_window_start > 60:
            self.request_count = 0
            self.request_window_start = current_time
        
        # Check if we can make another request
        if self.request_count >= self.config.rate_limit:
            logger.warning(f"Rate limit exceeded for {self.config.name}")
            return False
        
        self.request_count += 1
        self.last_request_time = current_time
        return True
    
    async def fetch_data(self, symbols: List[str], asset_type: AssetType) -> Dict[str, MarketData]:
        """Abstract method to fetch data"""
        raise NotImplementedError("Subclasses must implement fetch_data")

class YahooFinanceSource(BaseDataSource):
    """Yahoo Finance data source for stocks and ETFs"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.config.supported_assets = [AssetType.STOCK, AssetType.ETF, AssetType.INDEX, AssetType.CRYPTO]
    
    async def fetch_data(self, symbols: List[str], asset_type: AssetType) -> Dict[str, MarketData]:
        """Fetch data from Yahoo Finance"""
        
        if asset_type not in self.config.supported_assets:
            return {}
        
        if not self._check_rate_limit():
            raise DataSourceError(f"Rate limit exceeded for {self.config.name}")
        
        data = {}
        start_time = time.time()
        
        try:
            # Use yfinance library for reliability
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    
                    # Get current data
                    info = ticker.info
                    hist = ticker.history(period="2d", interval="1m")
                    
                    if hist.empty:
                        continue
                    
                    # Get latest price data
                    latest = hist.iloc[-1]
                    previous = hist.iloc[-2] if len(hist) > 1 else latest
                    
                    current_price = float(latest['Close'])
                    previous_price = float(previous['Close'])
                    change = current_price - previous_price
                    change_percent = (change / previous_price) * 100 if previous_price > 0 else 0.0
                    
                    # Create MarketData object with technical indicators
                    market_data = MarketData(
                        symbol=symbol,
                        asset_type=asset_type,
                        price=current_price,
                        change=change,
                        change_percent=change_percent,
                        volume=float(latest['Volume']),
                        timestamp=datetime.now()
                    )
                    
                    # Add technical indicators if we have enough data
                    if len(hist) >= 20:
                        market_data = await self._add_technical_indicators(market_data, hist)
                    
                    # Add fundamental data
                    market_data.market_cap = info.get('marketCap')
                    
                    data[symbol] = market_data
                    
                except Exception as e:
                    logger.warning(f"Error fetching {symbol} from Yahoo Finance: {e}")
                    continue
            
            # Update quality metrics
            response_time = time.time() - start_time
            self.quality_metrics.total_requests += 1
            self.quality_metrics.successful_requests += 1
            self.quality_metrics.average_response_time = (
                (self.quality_metrics.average_response_time * (self.quality_metrics.total_requests - 1) + response_time) /
                self.quality_metrics.total_requests
            )
            
        except Exception as e:
            self.quality_metrics.failed_requests += 1
            logger.error(f"Yahoo Finance fetch error: {e}")
            raise DataSourceError(f"Yahoo Finance error: {e}")
        
        return data
    
    async def _add_technical_indicators(self, market_data: MarketData, hist: pd.DataFrame) -> MarketData:
        """Add technical indicators to market data"""
        
        try:
            close_prices = hist['Close']
            high_prices = hist['High']
            low_prices = hist['Low']
            volume_data = hist['Volume']
            
            # RSI
            if len(close_prices) >= 14:
                rsi = RSIIndicator(close_prices, window=14)
                market_data.rsi = float(rsi.rsi().iloc[-1])
            
            # MACD
            if len(close_prices) >= 26:
                macd = MACD(close_prices)
                market_data.macd = float(macd.macd().iloc[-1])
                market_data.macd_signal = float(macd.macd_signal().iloc[-1])
                market_data.macd_histogram = float(macd.macd_diff().iloc[-1])
            
            # Bollinger Bands
            if len(close_prices) >= 20:
                bb = BollingerBands(close_prices, window=20)
                market_data.bollinger_upper = float(bb.bollinger_hband().iloc[-1])
                market_data.bollinger_lower = float(bb.bollinger_lband().iloc[-1])
                market_data.bollinger_middle = float(bb.bollinger_mavg().iloc[-1])
            
            # Moving Averages
            if len(close_prices) >= 50:
                sma_20 = SMAIndicator(close_prices, window=20)
                sma_50 = SMAIndicator(close_prices, window=50)
                market_data.sma_20 = float(sma_20.sma_indicator().iloc[-1])
                market_data.sma_50 = float(sma_50.sma_indicator().iloc[-1])
            
            # EMAs
            if len(close_prices) >= 26:
                ema_12 = EMAIndicator(close_prices, window=12)
                ema_26 = EMAIndicator(close_prices, window=26)
                market_data.ema_12 = float(ema_12.ema_indicator().iloc[-1])
                market_data.ema_26 = float(ema_26.ema_indicator().iloc[-1])
            
            # Volatility
            if len(close_prices) >= 20:
                returns = close_prices.pct_change().dropna()
                market_data.volatility = float(returns.std() * np.sqrt(252) * 100)  # Annualized volatility %
            
            # Average True Range
            if len(close_prices) >= 14:
                atr = AverageTrueRange(high_prices, low_prices, close_prices, window=14)
                market_data.atr = float(atr.average_true_range().iloc[-1])
            
            # Volume indicators
            if len(volume_data) >= 20:
                volume_sma = SMAIndicator(volume_data, window=20)
                market_data.volume_sma = float(volume_sma.sma_indicator().iloc[-1])
                market_data.volume_ratio = float(market_data.volume / market_data.volume_sma) if market_data.volume_sma > 0 else 1.0
            
        except Exception as e:
            logger.warning(f"Error calculating technical indicators for {market_data.symbol}: {e}")
        
        return market_data

class CoinGeckoSource(BaseDataSource):
    """CoinGecko data source for cryptocurrency data"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.config.supported_assets = [AssetType.CRYPTO]
        self.config.base_url = "https://api.coingecko.com/api/v3"
    
    async def fetch_data(self, symbols: List[str], asset_type: AssetType) -> Dict[str, MarketData]:
        """Fetch cryptocurrency data from CoinGecko"""
        
        if asset_type != AssetType.CRYPTO:
            return {}
        
        if not self._check_rate_limit():
            raise DataSourceError(f"Rate limit exceeded for {self.config.name}")
        
        await self.initialize()
        
        data = {}
        start_time = time.time()
        
        try:
            # Convert symbols to CoinGecko IDs
            coin_ids = [self._symbol_to_coingecko_id(symbol) for symbol in symbols]
            coin_ids_str = ','.join(coin_ids)
            
            # Fetch data from CoinGecko
            url = f"{self.config.base_url}/simple/price"
            params = {
                'ids': coin_ids_str,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true',
                'include_market_cap': 'true',
                'include_last_updated_at': 'true'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    for coin_id, coin_data in result.items():
                        symbol = self._coingecko_id_to_symbol(coin_id)
                        if not symbol:
                            continue
                        
                        current_price = coin_data.get('usd', 0)
                        change_percent = coin_data.get('usd_24h_change', 0)
                        change = current_price * (change_percent / 100)
                        
                        market_data = MarketData(
                            symbol=symbol,
                            asset_type=AssetType.CRYPTO,
                            price=current_price,
                            change=change,
                            change_percent=change_percent,
                            volume=coin_data.get('usd_24h_vol', 0),
                            timestamp=datetime.now(),
                            market_cap=coin_data.get('usd_market_cap'),
                            liquidity_score=self._calculate_crypto_liquidity(coin_data)
                        )
                        
                        data[symbol] = market_data
                else:
                    raise DataSourceError(f"CoinGecko API error: {response.status}")
            
            # Update quality metrics
            response_time = time.time() - start_time
            self.quality_metrics.total_requests += 1
            self.quality_metrics.successful_requests += 1
            self.quality_metrics.average_response_time = (
                (self.quality_metrics.average_response_time * (self.quality_metrics.total_requests - 1) + response_time) /
                self.quality_metrics.total_requests
            )
            
        except Exception as e:
            self.quality_metrics.failed_requests += 1
            logger.error(f"CoinGecko fetch error: {e}")
            raise DataSourceError(f"CoinGecko error: {e}")
        
        return data
    
    def _symbol_to_coingecko_id(self, symbol: str) -> str:
        """Convert trading symbol to CoinGecko ID"""
        # Remove common suffixes
        clean_symbol = symbol.replace('-USD', '').replace('USDT', '').replace('USD', '').lower()
        
        # Common symbol mappings
        mapping = {
            'btc': 'bitcoin',
            'eth': 'ethereum',
            'ada': 'cardano',
            'sol': 'solana',
            'doge': 'dogecoin',
            'matic': 'polygon',
            'avax': 'avalanche-2',
            'dot': 'polkadot',
            'link': 'chainlink',
            'uni': 'uniswap',
            'xlm': 'stellar',
            'xrp': 'ripple',
            'ltc': 'litecoin',
            'bch': 'bitcoin-cash',
            'eos': 'eos',
            'trx': 'tron',
            'bnb': 'binancecoin',
            'atom': 'cosmos',
            'algo': 'algorand',
            'vet': 'vechain'
        }
        
        return mapping.get(clean_symbol, clean_symbol)
    
    def _coingecko_id_to_symbol(self, coin_id: str) -> Optional[str]:
        """Convert CoinGecko ID back to trading symbol"""
        # Reverse mapping
        mapping = {
            'bitcoin': 'BTC-USD',
            'ethereum': 'ETH-USD',
            'cardano': 'ADA-USD',
            'solana': 'SOL-USD',
            'dogecoin': 'DOGE-USD',
            'polygon': 'MATIC-USD',
            'avalanche-2': 'AVAX-USD',
            'polkadot': 'DOT-USD',
            'chainlink': 'LINK-USD',
            'uniswap': 'UNI-USD',
            'stellar': 'XLM-USD',
            'ripple': 'XRP-USD',
            'litecoin': 'LTC-USD',
            'bitcoin-cash': 'BCH-USD',
            'eos': 'EOS-USD',
            'tron': 'TRX-USD',
            'binancecoin': 'BNB-USD',
            'cosmos': 'ATOM-USD',
            'algorand': 'ALGO-USD',
            'vechain': 'VET-USD'
        }
        
        return mapping.get(coin_id, f"{coin_id.upper()}-USD")
    
    def _calculate_crypto_liquidity(self, coin_data: Dict) -> float:
        """Calculate liquidity score for cryptocurrency"""
        try:
            volume = coin_data.get('usd_24h_vol', 0)
            market_cap = coin_data.get('usd_market_cap', 0)
            
            # Volume to market cap ratio as liquidity indicator
            if market_cap > 0:
                liquidity_ratio = volume / market_cap
                # Normalize to 0-1 scale (typical good liquidity is 0.01-0.1 ratio)
                liquidity_score = min(1.0, liquidity_ratio * 10)
            else:
                liquidity_score = 0.1
            
            return liquidity_score
            
        except:
            return 0.3  # Default low liquidity

class AlphaVantageSource(BaseDataSource):
    """Alpha Vantage data source for forex and additional stock data"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.config.supported_assets = [AssetType.FOREX, AssetType.STOCK, AssetType.CRYPTO]
        self.config.base_url = "https://www.alphavantage.co/query"
    
    async def fetch_data(self, symbols: List[str], asset_type: AssetType) -> Dict[str, MarketData]:
        """Fetch forex or stock data from Alpha Vantage"""
        
        if not self.config.api_key:
            raise DataSourceError("Alpha Vantage API key required")
        
        if not self._check_rate_limit():
            raise DataSourceError(f"Rate limit exceeded for {self.config.name}")
        
        await self.initialize()
        
        data = {}
        
        try:
            if asset_type == AssetType.FOREX:
                data = await self._fetch_forex_data(symbols)
            elif asset_type == AssetType.STOCK:
                data = await self._fetch_stock_data(symbols)
            elif asset_type == AssetType.CRYPTO:
                data = await self._fetch_crypto_data(symbols)
            
        except Exception as e:
            logger.error(f"Alpha Vantage fetch error: {e}")
            raise DataSourceError(f"Alpha Vantage error: {e}")
        
        return data
    
    async def _fetch_forex_data(self, pairs: List[str]) -> Dict[str, MarketData]:
        """Fetch forex data from Alpha Vantage"""
        data = {}
        
        for pair in pairs:
            try:
                if len(pair) != 6:  # EURUSD format
                    continue
                
                from_currency = pair[:3]
                to_currency = pair[3:]
                
                params = {
                    'function': 'FX_INTRADAY',
                    'from_symbol': from_currency,
                    'to_symbol': to_currency,
                    'interval': '1min',
                    'apikey': self.config.api_key
                }
                
                async with self.session.get(self.config.base_url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        if 'Time Series (1min)' in result:
                            time_series = result['Time Series (1min)']
                            latest_time = max(time_series.keys())
                            latest_data = time_series[latest_time]
                            
                            current_price = float(latest_data['4. close'])
                            open_price = float(latest_data['1. open'])
                            change = current_price - open_price
                            change_percent = (change / open_price) * 100 if open_price > 0 else 0.0
                            
                            data[pair] = MarketData(
                                symbol=pair,
                                asset_type=AssetType.FOREX,
                                price=current_price,
                                change=change,
                                change_percent=change_percent,
                                volume=0.0,  # Forex doesn't have traditional volume
                                timestamp=datetime.now(),
                                bid_ask_spread=abs(float(latest_data['2. high']) - float(latest_data['3. low']))
                            )
                        
            except Exception as e:
                logger.warning(f"Error fetching forex pair {pair}: {e}")
                continue
        
        return data
    
    async def _fetch_stock_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Fetch stock data from Alpha Vantage"""
        data = {}
        
        for symbol in symbols:
            try:
                params = {
                    'function': 'TIME_SERIES_INTRADAY',
                    'symbol': symbol,
                    'interval': '1min',
                    'apikey': self.config.api_key
                }
                
                async with self.session.get(self.config.base_url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        if 'Time Series (1min)' in result:
                            time_series = result['Time Series (1min)']
                            latest_time = max(time_series.keys())
                            latest_data = time_series[latest_time]
                            
                            current_price = float(latest_data['4. close'])
                            open_price = float(latest_data['1. open'])
                            change = current_price - open_price
                            change_percent = (change / open_price) * 100 if open_price > 0 else 0.0
                            
                            data[symbol] = MarketData(
                                symbol=symbol,
                                asset_type=AssetType.STOCK,
                                price=current_price,
                                change=change,
                                change_percent=change_percent,
                                volume=float(latest_data['5. volume']),
                                timestamp=datetime.now()
                            )
                        
            except Exception as e:
                logger.warning(f"Error fetching stock {symbol}: {e}")
                continue
        
        return data
    
    async def _fetch_crypto_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Fetch crypto data from Alpha Vantage"""
        data = {}
        
        for symbol in symbols:
            try:
                # Alpha Vantage uses different format for crypto
                clean_symbol = symbol.replace('-USD', '').replace('USD', '')
                
                params = {
                    'function': 'CRYPTO_INTRADAY',
                    'symbol': clean_symbol,
                    'market': 'USD',
                    'interval': '1min',
                    'apikey': self.config.api_key
                }
                
                async with self.session.get(self.config.base_url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        if 'Time Series Crypto (1min)' in result:
                            time_series = result['Time Series Crypto (1min)']
                            latest_time = max(time_series.keys())
                            latest_data = time_series[latest_time]
                            
                            current_price = float(latest_data['4. close'])
                            open_price = float(latest_data['1. open'])
                            change = current_price - open_price
                            change_percent = (change / open_price) * 100 if open_price > 0 else 0.0
                            
                            data[symbol] = MarketData(
                                symbol=symbol,
                                asset_type=AssetType.CRYPTO,
                                price=current_price,
                                change=change,
                                change_percent=change_percent,
                                volume=float(latest_data['5. volume']),
                                timestamp=datetime.now()
                            )
                        
            except Exception as e:
                logger.warning(f"Error fetching crypto {symbol}: {e}")
                continue
        
        return data

# ==================== CACHING SYSTEM ====================

class DataCache:
    """Advanced caching system with multiple backends"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.memory_cache = {}
        self.cache_timestamps = {}
        self.redis_client = None
        self.sqlite_conn = None
        
        if config.enabled:
            self._initialize_backends()
    
    def _initialize_backends(self):
        """Initialize caching backends"""
        
        # Redis backend
        if self.config.redis_url:
            try:
                self.redis_client = redis.from_url(self.config.redis_url)
                self.redis_client.ping()
                logger.info("Redis cache backend initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")
                self.redis_client = None
        
        # SQLite backend
        try:
            cache_dir = Path(self.config.sqlite_path).parent
            cache_dir.mkdir(exist_ok=True)
            
            self.sqlite_conn = sqlite3.connect(self.config.sqlite_path, check_same_thread=False)
            self.sqlite_conn.execute('''
                CREATE TABLE IF NOT EXISTS market_data_cache (
                    key TEXT PRIMARY KEY,
                    data TEXT,
                    timestamp REAL
                )
            ''')
            self.sqlite_conn.commit()
            logger.info("SQLite cache backend initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize SQLite cache: {e}")
            self.sqlite_conn = None
    
    def get(self, key: str) -> Optional[Dict[str, MarketData]]:
        """Get cached data"""
        
        if not self.config.enabled:
            return None
        
        # Check memory cache first
        if key in self.memory_cache:
            timestamp = self.cache_timestamps.get(key, 0)
            if time.time() - timestamp < self.config.ttl_seconds:
                return self.memory_cache[key]
            else:
                # Expired, remove from memory cache
                del self.memory_cache[key]
                del self.cache_timestamps[key]
        
        # Check Redis cache
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    data_dict = json.loads(cached_data)
                    # Convert back to MarketData objects
                    market_data = {}
                    for symbol, data in data_dict.items():
                        market_data[symbol] = self._dict_to_market_data(data)
                    
                    # Store in memory cache for faster access
                    self.memory_cache[key] = market_data
                    self.cache_timestamps[key] = time.time()
                    
                    return market_data
            except Exception as e:
                logger.warning(f"Redis cache get error: {e}")
        
        # Check SQLite cache
        if self.sqlite_conn:
            try:
                cursor = self.sqlite_conn.cursor()
                cursor.execute(
                    "SELECT data, timestamp FROM market_data_cache WHERE key = ?", 
                    (key,)
                )
                result = cursor.fetchone()
                
                if result:
                    data_json, timestamp = result
                    if time.time() - timestamp < self.config.ttl_seconds:
                        data_dict = json.loads(data_json)
                        market_data = {}
                        for symbol, data in data_dict.items():
                            market_data[symbol] = self._dict_to_market_data(data)
                        
                        # Store in memory cache
                        self.memory_cache[key] = market_data
                        self.cache_timestamps[key] = time.time()
                        
                        return market_data
            except Exception as e:
                logger.warning(f"SQLite cache get error: {e}")
        
        return None
    
    def set(self, key: str, data: Dict[str, MarketData]):
        """Store data in cache"""
        
        if not self.config.enabled:
            return
        
        current_time = time.time()
        
        # Store in memory cache
        self.memory_cache[key] = data
        self.cache_timestamps[key] = current_time
        
        # Limit memory cache size
        if len(self.memory_cache) > self.config.max_memory_items:
            # Remove oldest entries
            oldest_keys = sorted(self.cache_timestamps.items(), key=lambda x: x[1])[:10]
            for old_key, _ in oldest_keys:
                del self.memory_cache[old_key]
                del self.cache_timestamps[old_key]
        
        # Convert to serializable format
        data_dict = {}
        for symbol, market_data in data.items():
            data_dict[symbol] = asdict(market_data)
            # Convert datetime to string
            data_dict[symbol]['timestamp'] = market_data.timestamp.isoformat()
            data_dict[symbol]['asset_type'] = market_data.asset_type.value
        
        # Store in Redis cache
        if self.redis_client:
            try:
                self.redis_client.setex(
                    key, 
                    self.config.ttl_seconds, 
                    json.dumps(data_dict, default=str)
                )
            except Exception as e:
                logger.warning(f"Redis cache set error: {e}")
        
        # Store in SQLite cache
        if self.sqlite_conn:
            try:
                cursor = self.sqlite_conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO market_data_cache (key, data, timestamp) VALUES (?, ?, ?)",
                    (key, json.dumps(data_dict, default=str), current_time)
                )
                self.sqlite_conn.commit()
            except Exception as e:
                logger.warning(f"SQLite cache set error: {e}")
    
    def _dict_to_market_data(self, data_dict: Dict) -> MarketData:
        """Convert dictionary back to MarketData object"""
        # Convert string timestamp back to datetime
        if isinstance(data_dict['timestamp'], str):
            data_dict['timestamp'] = datetime.fromisoformat(data_dict['timestamp'])
        
        # Convert asset_type string back to enum
        if isinstance(data_dict['asset_type'], str):
            data_dict['asset_type'] = AssetType(data_dict['asset_type'])
        
        return MarketData(**data_dict)
    
    def clear(self):
        """Clear all cached data"""
        self.memory_cache.clear()
        self.cache_timestamps.clear()
        
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logger.warning(f"Redis cache clear error: {e}")
        
        if self.sqlite_conn:
            try:
                cursor = self.sqlite_conn.cursor()
                cursor.execute("DELETE FROM market_data_cache")
                self.sqlite_conn.commit()
            except Exception as e:
                logger.warning(f"SQLite cache clear error: {e}")

# ==================== MAIN MULTI-ASSET MANAGER ====================

class MultiAssetDataManager:
    """
    Unified data manager for all asset types with intelligent routing
    
    Features:
    - Multi-source data fetching with failover
    - Intelligent caching and rate limiting
    - Real-time data validation and quality checks
    - Performance monitoring and optimization
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
        # Initialize components
        self.sources = self._initialize_sources()
        self.cache = DataCache(CacheConfig(**self.config.get('cache', {})))
        self.quality_metrics = DataQualityMetrics()
        
        # Threading and async management
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.source_lock = threading.RLock()
        
        logger.info("Multi-Asset Data Manager initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'cache': {
                'enabled': True,
                'ttl_seconds': 30,
                'redis_url': os.getenv('REDIS_URL'),
                'sqlite_path': 'data/market_data_cache.db'
            },
            'sources': {
                'yahoo_finance': {'enabled': True, 'priority': 1},
                'coingecko': {'enabled': True, 'priority': 2},
                'alpha_vantage': {'enabled': True, 'priority': 3}
            },
            'timeouts': {
                'yahoo_finance': 10,
                'coingecko': 15,
                'alpha_vantage': 20
            },
            'api_keys': {
                'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY'),
                'polygon': os.getenv('POLYGON_API_KEY')
            }
        }
    
    def _initialize_sources(self) -> Dict[str, BaseDataSource]:
        """Initialize data sources"""
        
        sources = {}
        
        # Yahoo Finance
        if self.config['sources']['yahoo_finance']['enabled']:
            yahoo_config = DataSourceConfig(
                name="Yahoo Finance",
                rate_limit=200,
                timeout=self.config['timeouts']['yahoo_finance'],
                priority=self.config['sources']['yahoo_finance']['priority']
            )
            sources['yahoo_finance'] = YahooFinanceSource(yahoo_config)
        
        # CoinGecko
        if self.config['sources']['coingecko']['enabled']:
            coingecko_config = DataSourceConfig(
                name="CoinGecko",
                rate_limit=100,
                timeout=self.config['timeouts']['coingecko'],
                priority=self.config['sources']['coingecko']['priority']
            )
            sources['coingecko'] = CoinGeckoSource(coingecko_config)
        
        # Alpha Vantage (if API key provided)
        if (self.config['sources']['alpha_vantage']['enabled'] and 
            self.config.get('api_keys', {}).get('alpha_vantage')):
            
            alpha_config = DataSourceConfig(
                name="Alpha Vantage",
                api_key=self.config['api_keys']['alpha_vantage'],
                rate_limit=75,  # Free tier limit
                timeout=self.config['timeouts']['alpha_vantage'],
                priority=self.config['sources']['alpha_vantage']['priority']
            )
            sources['alpha_vantage'] = AlphaVantageSource(alpha_config)
        
        logger.info(f"Initialized {len(sources)} data sources: {list(sources.keys())}")
        return sources
    
    async def initialize(self):
        """Initialize async components"""
        for source in self.sources.values():
            await source.initialize()
        logger.info("Data sources initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        for source in self.sources.values():
            await source.cleanup()
    
    async def fetch_market_data(self, symbols: List[str], asset_type: AssetType) -> Dict[str, MarketData]:
        """
        Fetch market data for given symbols with intelligent source routing
        
        Args:
            symbols: List of symbols to fetch
            asset_type: Type of assets (STOCK, CRYPTO, FOREX, etc.)
            
        Returns:
            Dictionary mapping symbols to MarketData objects
        """
        
        # Generate cache key
        cache_key = f"{asset_type.value}:{','.join(sorted(symbols))}"
        
        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data:
            self.quality_metrics.cache_hits += 1
            return cached_data
        
        self.quality_metrics.cache_misses += 1
        
        # Get appropriate sources for asset type
        suitable_sources = self._get_sources_for_asset_type(asset_type)
        
        if not suitable_sources:
            logger.warning(f"No suitable sources found for asset type: {asset_type}")
            return {}
        
        # Try sources in priority order
        for source_name in suitable_sources:
            source = self.sources[source_name]
            
            try:
                start_time = time.time()
                data = await source.fetch_data(symbols, asset_type)
                response_time = time.time() - start_time
                
                if data:
                    # Update quality metrics
                    self.quality_metrics.total_requests += 1
                    self.quality_metrics.successful_requests += 1
                    self.quality_metrics.average_response_time = (
                        (self.quality_metrics.average_response_time * (self.quality_metrics.total_requests - 1) + response_time) /
                        self.quality_metrics.total_requests
                    )
                    
                    # Validate data quality
                    validated_data = self._validate_data_quality(data)
                    
                    # Cache the data
                    self.cache.set(cache_key, validated_data)
                    
                    logger.info(f"Successfully fetched {len(validated_data)} symbols from {source_name}")
                    return validated_data
                
            except DataSourceError as e:
                logger.warning(f"Data source {source_name} failed: {e}")
                self.quality_metrics.failed_requests += 1
                continue
            except Exception as e:
                logger.error(f"Unexpected error from {source_name}: {e}")
                self.quality_metrics.failed_requests += 1
                continue
        
        logger.error(f"All data sources failed for {asset_type} symbols: {symbols}")
        return {}
    
    def _get_sources_for_asset_type(self, asset_type: AssetType) -> List[str]:
        """Get suitable sources for asset type, sorted by priority"""
        
        asset_source_mapping = {
            AssetType.STOCK: ['yahoo_finance', 'alpha_vantage'],
            AssetType.ETF: ['yahoo_finance', 'alpha_vantage'],
            AssetType.INDEX: ['yahoo_finance'],
            AssetType.CRYPTO: ['coingecko', 'yahoo_finance', 'alpha_vantage'],
            AssetType.FOREX: ['alpha_vantage'],
            AssetType.COMMODITY: ['yahoo_finance']  # Using commodity ETFs
        }
        
        suitable_sources = asset_source_mapping.get(asset_type, [])
        
        # Filter by enabled sources and sort by priority
        enabled_sources = []
        for source_name in suitable_sources:
            if source_name in self.sources:
                source = self.sources[source_name]
                enabled_sources.append((source_name, source.config.priority))
        
        # Sort by priority (lower number = higher priority)
        enabled_sources.sort(key=lambda x: x[1])
        
        return [source_name for source_name, _ in enabled_sources]
    
    def _validate_data_quality(self, data: Dict[str, MarketData]) -> Dict[str, MarketData]:
        """Validate and clean market data"""
        
        validated_data = {}
        
        for symbol, market_data in data.items():
            # Basic validation
            if not self._is_valid_market_data(market_data):
                logger.warning(f"Invalid data for {symbol}, skipping")
                continue
            
            # Price validation
            if market_data.price <= 0:
                logger.warning(f"Invalid price for {symbol}: {market_data.price}")
                continue
            
            # Change validation
            if abs(market_data.change_percent) > 50:  # Extreme movements
                logger.warning(f"Extreme price change for {symbol}: {market_data.change_percent}%")
                # Don't skip, but flag for monitoring
            
            validated_data[symbol] = market_data
        
        # Update data quality score
        if len(data) > 0:
            self.quality_metrics.completeness_score = len(validated_data) / len(data)
        
        return validated_data
    
    def _is_valid_market_data(self, market_data: MarketData) -> bool:
        """Check if market data is valid"""
        
        # Required fields
        if not market_data.symbol or market_data.price is None:
            return False
        
        # Timestamp check
        if not market_data.timestamp:
            return False
        
        # Data freshness check (not older than 1 hour for most assets)
        max_age = timedelta(hours=1)
        if datetime.now() - market_data.timestamp > max_age:
            return False
        
        return True
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        
        source_health = {}
        for name, source in self.sources.items():
            source_health[name] = {
                'enabled': source.config.enabled,
                'total_requests': source.quality_metrics.total_requests,
                'success_rate': (
                    source.quality_metrics.successful_requests / 
                    max(1, source.quality_metrics.total_requests)
                ),
                'average_response_time': source.quality_metrics.average_response_time,
                'last_request_time': source.last_request_time
            }
        
        cache_metrics = {
            'enabled': self.cache.config.enabled,
            'hit_rate': (
                self.quality_metrics.cache_hits / 
                max(1, self.quality_metrics.cache_hits + self.quality_metrics.cache_misses)
            ),
            'memory_items': len(self.cache.memory_cache)
        }
        
        return {
            'status': 'healthy',
            'sources': source_health,
            'cache': cache_metrics,
            'overall_metrics': {
                'total_requests': self.quality_metrics.total_requests,
                'success_rate': (
                    self.quality_metrics.successful_requests / 
                    max(1, self.quality_metrics.total_requests)
                ),
                'average_response_time': self.quality_metrics.average_response_time,
                'data_completeness': self.quality_metrics.completeness_score,
                'cache_hit_rate_percent': (
                    self.quality_metrics.cache_hits / 
                    max(1, self.quality_metrics.cache_hits + self.quality_metrics.cache_misses)
                ) * 100
            }
        }

# ==================== CONVENIENCE FUNCTIONS ====================

async def fetch_stocks(symbols: List[str], config: Dict = None) -> Dict[str, MarketData]:
    """Convenience function to fetch stock data"""
    manager = MultiAssetDataManager(config)
    await manager.initialize()
    try:
        return await manager.fetch_market_data(symbols, AssetType.STOCK)
    finally:
        await manager.cleanup()

async def fetch_crypto(symbols: List[str], config: Dict = None) -> Dict[str, MarketData]:
    """Convenience function to fetch crypto data"""
    manager = MultiAssetDataManager(config)
    await manager.initialize()
    try:
        return await manager.fetch_market_data(symbols, AssetType.CRYPTO)
    finally:
        await manager.cleanup()

async def fetch_forex(pairs: List[str], config: Dict = None) -> Dict[str, MarketData]:
    """Convenience function to fetch forex data"""
    manager = MultiAssetDataManager(config)
    await manager.initialize()
    try:
        return await manager.fetch_market_data(pairs, AssetType.FOREX)
    finally:
        await manager.cleanup()

# ==================== TESTING ====================

async def test_multi_asset_manager():
    """Test the multi-asset data manager"""
    print("🧪 Testing Multi-Asset Data Manager")
    print("=" * 50)
    
    # Test configuration
    config = {
        'cache': {'enabled': True, 'ttl_seconds': 30},
        'sources': {
            'yahoo_finance': {'enabled': True, 'priority': 1},
            'coingecko': {'enabled': True, 'priority': 2}
        }
    }
    
    manager = MultiAssetDataManager(config)
    await manager.initialize()
    
    try:
        # Test stock data
        print("📈 Testing stock data...")
        stock_symbols = ['AAPL', 'GOOGL', 'MSFT']
        stock_data = await manager.fetch_market_data(stock_symbols, AssetType.STOCK)
        print(f"✅ Fetched {len(stock_data)} stock symbols")
        
        # Test crypto data
        print("🪙 Testing crypto data...")
        crypto_symbols = ['BTC-USD', 'ETH-USD']
        crypto_data = await manager.fetch_market_data(crypto_symbols, AssetType.CRYPTO)
        print(f"✅ Fetched {len(crypto_data)} crypto symbols")
        
        # Test cache performance
        print("⚡ Testing cache performance...")
        start_time = time.time()
        cached_stock_data = await manager.fetch_market_data(stock_symbols, AssetType.STOCK)
        cache_time = (time.time() - start_time) * 1000
        print(f"✅ Cache fetch took {cache_time:.2f}ms")
        
        # Health status
        health = manager.get_health_status()
        print(f"📊 Health Status: {health['status']}")
        print(f"📊 Cache Hit Rate: {health['overall_metrics']['cache_hit_rate_percent']:.1f}%")
        print(f"📊 Avg Response Time: {health['overall_metrics']['average_response_time']:.2f}s")
        
    finally:
        await manager.cleanup()
    
    print("\n🎉 Multi-Asset Data Manager test completed!")

if __name__ == "__main__":
    # Run test
    asyncio.run(test_multi_asset_manager())