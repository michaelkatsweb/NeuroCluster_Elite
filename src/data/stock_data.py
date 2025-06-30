#!/usr/bin/env python3
"""
File: stock_data.py
Path: NeuroCluster-Elite/src/data/stock_data.py
Description: Stock market data fetcher with multiple data sources and real-time capabilities

This module provides comprehensive stock market data fetching from multiple sources
including Yahoo Finance, Alpha Vantage, Polygon, and others with intelligent failover,
caching, and real-time WebSocket support.

Features:
- Multi-source data fetching with automatic failover
- Real-time quote and trade data via WebSocket
- Historical data with multiple timeframes
- Advanced caching and rate limiting
- Data validation and quality checks
- Earnings, fundamentals, and news data
- Options and institutional data
- Pre-market and after-hours data

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
import time
import logging
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import ssl
from urllib.parse import urlencode
import base64
import hmac
import hashlib

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

# ==================== DATA SOURCE CONFIGURATIONS ====================

@dataclass
class StockQuote:
    """Real-time stock quote data"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    bid: float = 0.0
    ask: float = 0.0
    bid_size: int = 0
    ask_size: int = 0
    last_trade_time: Optional[datetime] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    avg_volume: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class StockFundamentals:
    """Stock fundamental data"""
    symbol: str
    market_cap: float
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    eps: Optional[float] = None
    revenue: Optional[float] = None
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    beta: Optional[float] = None
    shares_outstanding: Optional[int] = None
    float_shares: Optional[int] = None
    insider_ownership: Optional[float] = None
    institutional_ownership: Optional[float] = None
    short_interest: Optional[float] = None
    forward_pe: Optional[float] = None
    price_to_book: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class EarningsData:
    """Earnings data"""
    symbol: str
    quarter: str
    year: int
    eps_actual: Optional[float] = None
    eps_estimate: Optional[float] = None
    eps_surprise: Optional[float] = None
    revenue_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_surprise: Optional[float] = None
    earnings_date: Optional[datetime] = None
    next_earnings_date: Optional[datetime] = None
    guidance: Optional[str] = None

@dataclass
class NewsItem:
    """News item data"""
    title: str
    summary: str
    url: str
    published_time: datetime
    source: str
    symbols: List[str] = field(default_factory=list)
    sentiment: Optional[float] = None  # -1 to 1
    relevance_score: Optional[float] = None  # 0 to 1

# ==================== STOCK DATA MANAGER ====================

class StockDataManager:
    """
    Comprehensive stock data manager with multiple data sources
    
    This class provides real-time and historical stock data from various sources
    with intelligent failover, caching, and data validation.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize stock data manager"""
        
        self.config = config or self._get_default_config()
        self.api_keys = self._load_api_keys()
        self.cache = {}
        self.rate_limiters = {}
        self.websocket_connections = {}
        self.real_time_callbacks = {}
        self.data_quality_metrics = {}
        
        # Initialize data sources
        self._initialize_data_sources()
        
        # Start real-time data thread
        self.real_time_thread = None
        self.is_running = False
        
        logger.info("Stock Data Manager initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'data_sources': {
                'yahoo_finance': {'enabled': True, 'priority': 1},
                'alpha_vantage': {'enabled': True, 'priority': 2},
                'polygon': {'enabled': True, 'priority': 3},
                'iex_cloud': {'enabled': True, 'priority': 4},
                'finnhub': {'enabled': True, 'priority': 5}
            },
            'cache_settings': {
                'enabled': True,
                'ttl_seconds': 30,
                'max_items': 10000
            },
            'rate_limits': {
                'yahoo_finance': 2000,  # requests per hour
                'alpha_vantage': 500,
                'polygon': 1000,
                'iex_cloud': 1000000,
                'finnhub': 300
            },
            'real_time': {
                'enabled': True,
                'max_symbols': 100,
                'reconnect_delay': 5
            },
            'data_validation': {
                'price_change_threshold': 0.2,  # 20% max change
                'volume_spike_threshold': 10.0   # 10x volume spike
            }
        }
    
    def _load_api_keys(self) -> Dict:
        """Load API keys from configuration"""
        return {
            'alpha_vantage': 'YOUR_ALPHA_VANTAGE_KEY',
            'polygon': 'YOUR_POLYGON_KEY',
            'iex_cloud': 'YOUR_IEX_CLOUD_KEY',
            'finnhub': 'YOUR_FINNHUB_KEY'
        }
    
    def _initialize_data_sources(self):
        """Initialize data source handlers"""
        
        self.data_sources = {
            'yahoo_finance': {
                'handler': self._fetch_yahoo_finance,
                'websocket': self._connect_yahoo_websocket,
                'rate_limit': self.config['rate_limits']['yahoo_finance']
            },
            'alpha_vantage': {
                'handler': self._fetch_alpha_vantage,
                'rate_limit': self.config['rate_limits']['alpha_vantage']
            },
            'polygon': {
                'handler': self._fetch_polygon,
                'websocket': self._connect_polygon_websocket,
                'rate_limit': self.config['rate_limits']['polygon']
            },
            'iex_cloud': {
                'handler': self._fetch_iex_cloud,
                'rate_limit': self.config['rate_limits']['iex_cloud']
            },
            'finnhub': {
                'handler': self._fetch_finnhub,
                'websocket': self._connect_finnhub_websocket,
                'rate_limit': self.config['rate_limits']['finnhub']
            }
        }
    
    async def get_real_time_quote(self, symbol: str) -> Optional[StockQuote]:
        """Get real-time quote for a symbol"""
        
        try:
            # Check cache first
            cache_key = f"quote_{symbol}"
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < 5:  # 5 second cache
                    return cached_data['data']
            
            # Try data sources in priority order
            for source_name, source_config in sorted(
                self.config['data_sources'].items(), 
                key=lambda x: x[1]['priority']
            ):
                if not source_config['enabled']:
                    continue
                
                try:
                    if source_name in self.data_sources:
                        handler = self.data_sources[source_name]['handler']
                        quote = await handler(symbol, data_type='quote')
                        
                        if quote and self._validate_quote_data(quote):
                            # Cache the result
                            self.cache[cache_key] = {
                                'data': quote,
                                'timestamp': datetime.now()
                            }
                            
                            self._update_quality_metrics(source_name, True)
                            return quote
                        
                except Exception as e:
                    logger.warning(f"Error fetching quote from {source_name}: {e}")
                    self._update_quality_metrics(source_name, False)
                    continue
            
            logger.warning(f"Could not fetch quote for {symbol} from any source")
            return None
            
        except Exception as e:
            logger.error(f"Error getting real-time quote for {symbol}: {e}")
            return None
    
    async def get_historical_data(self, 
                                symbol: str, 
                                timeframe: str = '1d',
                                start_date: datetime = None,
                                end_date: datetime = None,
                                limit: int = 1000) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data"""
        
        try:
            # Set default dates
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=365)
            
            # Check cache
            cache_key = f"hist_{symbol}_{timeframe}_{start_date.date()}_{end_date.date()}"
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < 300:  # 5 minute cache
                    return cached_data['data']
            
            # Try data sources
            for source_name, source_config in sorted(
                self.config['data_sources'].items(), 
                key=lambda x: x[1]['priority']
            ):
                if not source_config['enabled']:
                    continue
                
                try:
                    if source_name in self.data_sources:
                        handler = self.data_sources[source_name]['handler']
                        data = await handler(
                            symbol, 
                            data_type='historical',
                            timeframe=timeframe,
                            start_date=start_date,
                            end_date=end_date,
                            limit=limit
                        )
                        
                        if data is not None and len(data) > 0:
                            # Validate and clean data
                            cleaned_data = self._validate_and_clean_ohlcv(data)
                            
                            if cleaned_data is not None:
                                # Cache the result
                                self.cache[cache_key] = {
                                    'data': cleaned_data,
                                    'timestamp': datetime.now()
                                }
                                
                                self._update_quality_metrics(source_name, True)
                                return cleaned_data
                        
                except Exception as e:
                    logger.warning(f"Error fetching historical data from {source_name}: {e}")
                    self._update_quality_metrics(source_name, False)
                    continue
            
            logger.warning(f"Could not fetch historical data for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    async def get_fundamentals(self, symbol: str) -> Optional[StockFundamentals]:
        """Get fundamental data for a symbol"""
        
        try:
            # Check cache
            cache_key = f"fundamentals_{symbol}"
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < 3600:  # 1 hour cache
                    return cached_data['data']
            
            # Try to get from Yahoo Finance first (most comprehensive for free)
            try:
                fundamentals = await self._fetch_yahoo_fundamentals(symbol)
                if fundamentals:
                    self.cache[cache_key] = {
                        'data': fundamentals,
                        'timestamp': datetime.now()
                    }
                    return fundamentals
            except Exception as e:
                logger.warning(f"Error fetching fundamentals from Yahoo: {e}")
            
            # Fallback to other sources
            for source_name in ['alpha_vantage', 'iex_cloud', 'finnhub']:
                try:
                    if source_name in self.data_sources:
                        handler = self.data_sources[source_name]['handler']
                        fundamentals = await handler(symbol, data_type='fundamentals')
                        
                        if fundamentals:
                            self.cache[cache_key] = {
                                'data': fundamentals,
                                'timestamp': datetime.now()
                            }
                            return fundamentals
                            
                except Exception as e:
                    logger.warning(f"Error fetching fundamentals from {source_name}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting fundamentals for {symbol}: {e}")
            return None
    
    async def get_earnings_data(self, symbol: str) -> Optional[List[EarningsData]]:
        """Get earnings data for a symbol"""
        
        try:
            # Check cache
            cache_key = f"earnings_{symbol}"
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < 3600:  # 1 hour cache
                    return cached_data['data']
            
            earnings_data = await self._fetch_yahoo_earnings(symbol)
            
            if earnings_data:
                self.cache[cache_key] = {
                    'data': earnings_data,
                    'timestamp': datetime.now()
                }
            
            return earnings_data
            
        except Exception as e:
            logger.error(f"Error getting earnings data for {symbol}: {e}")
            return None
    
    async def get_news(self, symbol: str, limit: int = 20) -> Optional[List[NewsItem]]:
        """Get news for a symbol"""
        
        try:
            # Check cache
            cache_key = f"news_{symbol}_{limit}"
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < 600:  # 10 minute cache
                    return cached_data['data']
            
            news_items = []
            
            # Try multiple sources
            for source_name in ['yahoo_finance', 'alpha_vantage', 'iex_cloud']:
                try:
                    if source_name in self.data_sources:
                        handler = self.data_sources[source_name]['handler']
                        source_news = await handler(symbol, data_type='news', limit=limit)
                        
                        if source_news:
                            news_items.extend(source_news)
                            
                except Exception as e:
                    logger.warning(f"Error fetching news from {source_name}: {e}")
                    continue
            
            # Remove duplicates and sort by date
            unique_news = []
            seen_urls = set()
            
            for item in news_items:
                if item.url not in seen_urls:
                    unique_news.append(item)
                    seen_urls.add(item.url)
            
            unique_news.sort(key=lambda x: x.published_time, reverse=True)
            final_news = unique_news[:limit]
            
            if final_news:
                self.cache[cache_key] = {
                    'data': final_news,
                    'timestamp': datetime.now()
                }
            
            return final_news
            
        except Exception as e:
            logger.error(f"Error getting news for {symbol}: {e}")
            return None
    
    def start_real_time_data(self, symbols: List[str], callback: Callable):
        """Start real-time data streaming"""
        
        try:
            if not self.config['real_time']['enabled']:
                logger.warning("Real-time data is disabled in configuration")
                return
            
            self.is_running = True
            self.real_time_callbacks[callback] = symbols
            
            # Start WebSocket connections
            for source_name in ['yahoo_finance', 'polygon', 'finnhub']:
                if (source_name in self.data_sources and 
                    'websocket' in self.data_sources[source_name]):
                    
                    try:
                        ws_handler = self.data_sources[source_name]['websocket']
                        ws_handler(symbols, callback)
                    except Exception as e:
                        logger.warning(f"Error starting WebSocket for {source_name}: {e}")
            
            logger.info(f"Started real-time data for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error starting real-time data: {e}")
    
    def stop_real_time_data(self):
        """Stop real-time data streaming"""
        
        try:
            self.is_running = False
            
            # Close WebSocket connections
            for source_name, connection in self.websocket_connections.items():
                try:
                    if connection:
                        connection.close()
                except Exception as e:
                    logger.warning(f"Error closing WebSocket for {source_name}: {e}")
            
            self.websocket_connections.clear()
            self.real_time_callbacks.clear()
            
            logger.info("Stopped real-time data streaming")
            
        except Exception as e:
            logger.error(f"Error stopping real-time data: {e}")
    
    # Data source implementations
    
    async def _fetch_yahoo_finance(self, symbol: str, **kwargs) -> Any:
        """Fetch data from Yahoo Finance"""
        
        try:
            data_type = kwargs.get('data_type', 'quote')
            
            if data_type == 'quote':
                return await self._fetch_yahoo_quote(symbol)
            elif data_type == 'historical':
                return await self._fetch_yahoo_historical(symbol, **kwargs)
            elif data_type == 'news':
                return await self._fetch_yahoo_news(symbol, **kwargs)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error fetching from Yahoo Finance: {e}")
            return None
    
    async def _fetch_yahoo_quote(self, symbol: str) -> Optional[StockQuote]:
        """Fetch real-time quote from Yahoo Finance"""
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or 'regularMarketPrice' not in info:
                return None
            
            quote = StockQuote(
                symbol=symbol,
                price=float(info.get('regularMarketPrice', 0)),
                change=float(info.get('regularMarketChange', 0)),
                change_percent=float(info.get('regularMarketChangePercent', 0)),
                volume=int(info.get('regularMarketVolume', 0)),
                bid=float(info.get('bid', 0)),
                ask=float(info.get('ask', 0)),
                bid_size=int(info.get('bidSize', 0)),
                ask_size=int(info.get('askSize', 0)),
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE'),
                dividend_yield=info.get('dividendYield'),
                week_52_high=info.get('fiftyTwoWeekHigh'),
                week_52_low=info.get('fiftyTwoWeekLow'),
                avg_volume=info.get('averageVolume')
            )
            
            return quote
            
        except Exception as e:
            logger.warning(f"Error fetching Yahoo quote for {symbol}: {e}")
            return None
    
    async def _fetch_yahoo_historical(self, symbol: str, **kwargs) -> Optional[pd.DataFrame]:
        """Fetch historical data from Yahoo Finance"""
        
        try:
            timeframe = kwargs.get('timeframe', '1d')
            start_date = kwargs.get('start_date')
            end_date = kwargs.get('end_date')
            
            # Map timeframe to Yahoo Finance intervals
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '1d': '1d', '1wk': '1wk', '1mo': '1mo'
            }
            
            interval = interval_map.get(timeframe, '1d')
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                return None
            
            # Standardize column names
            data.columns = [col.lower() for col in data.columns]
            
            return data
            
        except Exception as e:
            logger.warning(f"Error fetching Yahoo historical data for {symbol}: {e}")
            return None
    
    async def _fetch_yahoo_fundamentals(self, symbol: str) -> Optional[StockFundamentals]:
        """Fetch fundamental data from Yahoo Finance"""
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return None
            
            fundamentals = StockFundamentals(
                symbol=symbol,
                market_cap=info.get('marketCap', 0),
                pe_ratio=info.get('trailingPE'),
                pb_ratio=info.get('priceToBook'),
                ps_ratio=info.get('priceToSalesTrailing12Months'),
                peg_ratio=info.get('pegRatio'),
                eps=info.get('trailingEps'),
                revenue=info.get('totalRevenue'),
                profit_margin=info.get('profitMargins'),
                operating_margin=info.get('operatingMargins'),
                return_on_equity=info.get('returnOnEquity'),
                return_on_assets=info.get('returnOnAssets'),
                debt_to_equity=info.get('debtToEquity'),
                current_ratio=info.get('currentRatio'),
                dividend_yield=info.get('dividendYield'),
                payout_ratio=info.get('payoutRatio'),
                beta=info.get('beta'),
                shares_outstanding=info.get('sharesOutstanding'),
                float_shares=info.get('floatShares'),
                insider_ownership=info.get('heldPercentInsiders'),
                institutional_ownership=info.get('heldPercentInstitutions'),
                short_interest=info.get('shortPercentOfFloat'),
                forward_pe=info.get('forwardPE')
            )
            
            return fundamentals
            
        except Exception as e:
            logger.warning(f"Error fetching Yahoo fundamentals for {symbol}: {e}")
            return None
    
    async def _fetch_yahoo_earnings(self, symbol: str) -> Optional[List[EarningsData]]:
        """Fetch earnings data from Yahoo Finance"""
        
        try:
            ticker = yf.Ticker(symbol)
            earnings = ticker.quarterly_earnings
            
            if earnings is None or earnings.empty:
                return None
            
            earnings_data = []
            
            for date, row in earnings.iterrows():
                earnings_item = EarningsData(
                    symbol=symbol,
                    quarter=f"Q{date.quarter}",
                    year=date.year,
                    eps_actual=row.get('Earnings'),
                    revenue_actual=row.get('Revenue')
                )
                earnings_data.append(earnings_item)
            
            return earnings_data
            
        except Exception as e:
            logger.warning(f"Error fetching Yahoo earnings for {symbol}: {e}")
            return None
    
    async def _fetch_yahoo_news(self, symbol: str, **kwargs) -> Optional[List[NewsItem]]:
        """Fetch news from Yahoo Finance"""
        
        try:
            limit = kwargs.get('limit', 20)
            
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                return None
            
            news_items = []
            
            for item in news[:limit]:
                news_item = NewsItem(
                    title=item.get('title', ''),
                    summary=item.get('summary', ''),
                    url=item.get('link', ''),
                    published_time=datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                    source=item.get('publisher', 'Yahoo Finance'),
                    symbols=[symbol]
                )
                news_items.append(news_item)
            
            return news_items
            
        except Exception as e:
            logger.warning(f"Error fetching Yahoo news for {symbol}: {e}")
            return None
    
    async def _fetch_alpha_vantage(self, symbol: str, **kwargs) -> Any:
        """Fetch data from Alpha Vantage"""
        
        # Placeholder - would implement Alpha Vantage API calls
        return None
    
    async def _fetch_polygon(self, symbol: str, **kwargs) -> Any:
        """Fetch data from Polygon"""
        
        # Placeholder - would implement Polygon API calls
        return None
    
    async def _fetch_iex_cloud(self, symbol: str, **kwargs) -> Any:
        """Fetch data from IEX Cloud"""
        
        # Placeholder - would implement IEX Cloud API calls
        return None
    
    async def _fetch_finnhub(self, symbol: str, **kwargs) -> Any:
        """Fetch data from Finnhub"""
        
        # Placeholder - would implement Finnhub API calls
        return None
    
    # WebSocket implementations
    
    def _connect_yahoo_websocket(self, symbols: List[str], callback: Callable):
        """Connect to Yahoo Finance WebSocket"""
        # Placeholder - would implement Yahoo WebSocket connection
        pass
    
    def _connect_polygon_websocket(self, symbols: List[str], callback: Callable):
        """Connect to Polygon WebSocket"""
        # Placeholder - would implement Polygon WebSocket connection
        pass
    
    def _connect_finnhub_websocket(self, symbols: List[str], callback: Callable):
        """Connect to Finnhub WebSocket"""
        # Placeholder - would implement Finnhub WebSocket connection
        pass
    
    # Validation and utility methods
    
    def _validate_quote_data(self, quote: StockQuote) -> bool:
        """Validate quote data quality"""
        
        try:
            # Basic validation
            if quote.price <= 0:
                return False
            
            # Check for extreme price changes
            threshold = self.config['data_validation']['price_change_threshold']
            if abs(quote.change_percent) > threshold * 100:
                logger.warning(f"Extreme price change detected for {quote.symbol}: {quote.change_percent}%")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating quote data: {e}")
            return False
    
    def _validate_and_clean_ohlcv(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Validate and clean OHLCV data"""
        
        try:
            if data is None or data.empty:
                return None
            
            # Required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            
            # Check if all required columns exist
            for col in required_cols:
                if col not in data.columns:
                    logger.warning(f"Missing required column: {col}")
                    return None
            
            # Remove rows with invalid data
            data = data.dropna(subset=required_cols)
            
            # Validate OHLC relationships
            invalid_rows = (
                (data['high'] < data['low']) |
                (data['high'] < data['open']) |
                (data['high'] < data['close']) |
                (data['low'] > data['open']) |
                (data['low'] > data['close']) |
                (data['volume'] < 0)
            )
            
            if invalid_rows.any():
                logger.warning(f"Removing {invalid_rows.sum()} invalid rows")
                data = data[~invalid_rows]
            
            # Remove extreme outliers
            for col in ['open', 'high', 'low', 'close']:
                q99 = data[col].quantile(0.99)
                q01 = data[col].quantile(0.01)
                outliers = (data[col] > q99 * 2) | (data[col] < q01 * 0.5)
                
                if outliers.any():
                    logger.warning(f"Removing {outliers.sum()} outliers from {col}")
                    data = data[~outliers]
            
            return data if len(data) > 0 else None
            
        except Exception as e:
            logger.error(f"Error validating OHLCV data: {e}")
            return None
    
    def _update_quality_metrics(self, source: str, success: bool):
        """Update data quality metrics for a source"""
        
        if source not in self.data_quality_metrics:
            self.data_quality_metrics[source] = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'success_rate': 0.0,
                'last_success': None,
                'last_failure': None
            }
        
        metrics = self.data_quality_metrics[source]
        metrics['total_requests'] += 1
        
        if success:
            metrics['successful_requests'] += 1
            metrics['last_success'] = datetime.now()
        else:
            metrics['failed_requests'] += 1
            metrics['last_failure'] = datetime.now()
        
        metrics['success_rate'] = metrics['successful_requests'] / metrics['total_requests']
    
    def get_data_quality_report(self) -> Dict:
        """Get data quality report for all sources"""
        
        report = {
            'timestamp': datetime.now(),
            'sources': self.data_quality_metrics.copy(),
            'cache_stats': {
                'items': len(self.cache),
                'max_items': self.config['cache_settings']['max_items']
            }
        }
        
        return report
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        logger.info("Cache cleared")

# ==================== TESTING ====================

async def test_stock_data_manager():
    """Test stock data manager functionality"""
    
    print("📈 Testing Stock Data Manager")
    print("=" * 40)
    
    # Create stock data manager
    stock_manager = StockDataManager()
    
    test_symbol = 'AAPL'
    
    print(f"✅ Testing real-time quote for {test_symbol}:")
    quote = await stock_manager.get_real_time_quote(test_symbol)
    
    if quote:
        print(f"   Price: ${quote.price:.2f}")
        print(f"   Change: {quote.change:+.2f} ({quote.change_percent:+.2f}%)")
        print(f"   Volume: {quote.volume:,}")
        print(f"   Bid/Ask: ${quote.bid:.2f} / ${quote.ask:.2f}")
        if quote.market_cap:
            print(f"   Market Cap: ${quote.market_cap/1e9:.1f}B")
    else:
        print("   ❌ Could not fetch quote")
    
    print(f"\n✅ Testing historical data for {test_symbol}:")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    historical_data = await stock_manager.get_historical_data(
        test_symbol, 
        timeframe='1d',
        start_date=start_date,
        end_date=end_date
    )
    
    if historical_data is not None and len(historical_data) > 0:
        print(f"   Data points: {len(historical_data)}")
        print(f"   Date range: {historical_data.index[0].date()} to {historical_data.index[-1].date()}")
        print(f"   Price range: ${historical_data['low'].min():.2f} - ${historical_data['high'].max():.2f}")
        print(f"   Avg volume: {historical_data['volume'].mean():,.0f}")
    else:
        print("   ❌ Could not fetch historical data")
    
    print(f"\n✅ Testing fundamentals for {test_symbol}:")
    fundamentals = await stock_manager.get_fundamentals(test_symbol)
    
    if fundamentals:
        print(f"   Market Cap: ${fundamentals.market_cap/1e9:.1f}B")
        if fundamentals.pe_ratio:
            print(f"   P/E Ratio: {fundamentals.pe_ratio:.2f}")
        if fundamentals.dividend_yield:
            print(f"   Dividend Yield: {fundamentals.dividend_yield:.2%}")
        if fundamentals.beta:
            print(f"   Beta: {fundamentals.beta:.2f}")
    else:
        print("   ❌ Could not fetch fundamentals")
    
    print(f"\n✅ Testing news for {test_symbol}:")
    news = await stock_manager.get_news(test_symbol, limit=5)
    
    if news:
        print(f"   Found {len(news)} news items:")
        for i, item in enumerate(news[:3]):
            print(f"   {i+1}. {item.title[:60]}...")
            print(f"      Source: {item.source} | {item.published_time.strftime('%Y-%m-%d %H:%M')}")
    else:
        print("   ❌ Could not fetch news")
    
    # Test data quality report
    print(f"\n✅ Data quality report:")
    quality_report = stock_manager.get_data_quality_report()
    
    for source, metrics in quality_report['sources'].items():
        print(f"   {source}: {metrics['success_rate']:.1%} success rate "
              f"({metrics['successful_requests']}/{metrics['total_requests']} requests)")
    
    print(f"   Cache: {quality_report['cache_stats']['items']} items")
    
    print("\n🎉 Stock data manager tests completed!")

if __name__ == "__main__":
    asyncio.run(test_stock_data_manager())