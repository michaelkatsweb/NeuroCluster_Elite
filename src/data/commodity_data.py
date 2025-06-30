#!/usr/bin/env python3
"""
File: commodity_data.py
Path: NeuroCluster-Elite/src/data/commodity_data.py
Description: Commodity data fetcher with multiple sources and futures market integration

This module provides comprehensive commodity data fetching from multiple sources
including futures exchanges, ETF proxies, and commodity-specific APIs with real-time
pricing, supply/demand analysis, and weather/geopolitical impact assessment.

Features:
- Multi-source commodity data aggregation
- Futures contracts and spot prices
- ETF proxy data for retail access
- Supply/demand fundamentals tracking
- Weather and seasonal impact analysis
- Storage levels and inventory data
- Geopolitical risk assessment
- Agricultural, energy, and metals coverage
- Real-time price alerts and notifications

Author: Your Name
Created: 2025-06-29
Version: 1.0.0
License: MIT
"""

import asyncio
import aiohttp
import requests
import yfinance as yf
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

# Import our modules
try:
    from src.core.neurocluster_elite import AssetType, MarketData
    from src.utils.config_manager import ConfigManager
    from src.utils.helpers import retry_on_failure, format_currency, calculate_hash
except ImportError:
    # Fallback for testing
    pass

# Configure logging
logger = logging.getLogger(__name__)

# ==================== COMMODITY DATA STRUCTURES ====================

@dataclass
class CommodityQuote:
    """Real-time commodity quote data"""
    symbol: str
    name: str
    category: str  # "Energy", "Metals", "Agriculture", "Livestock"
    price: float
    currency: str  # "USD", "EUR", etc.
    unit: str  # "per barrel", "per ounce", "per bushel", etc.
    change_24h: float
    change_percent_24h: float
    high_24h: float
    low_24h: float
    volume_24h: Optional[float] = None
    open_interest: Optional[int] = None  # For futures
    settlement_date: Optional[datetime] = None  # For futures
    spot_price: Optional[float] = None
    futures_price: Optional[float] = None
    basis: Optional[float] = None  # Futures - Spot
    bid: Optional[float] = None
    ask: Optional[float] = None
    last_trade_time: Optional[datetime] = None
    exchange: str = ""
    provider: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class CommodityInfo:
    """Commodity metadata and characteristics"""
    symbol: str
    name: str
    category: str
    subcategory: str = ""
    description: str = ""
    unit: str = ""
    currency: str = "USD"
    major_exchanges: List[str] = field(default_factory=list)
    trading_hours: Optional[Dict] = None
    contract_months: List[str] = field(default_factory=list)
    tick_size: float = 0.01
    contract_size: Optional[str] = None
    margin_requirements: Optional[Dict] = None
    seasonal_patterns: Optional[Dict] = None
    supply_factors: List[str] = field(default_factory=list)
    demand_factors: List[str] = field(default_factory=list)
    related_etfs: List[str] = field(default_factory=list)
    correlation_assets: List[str] = field(default_factory=list)

@dataclass
class SupplyDemandData:
    """Supply and demand fundamentals"""
    commodity: str
    report_date: datetime
    supply_data: Dict[str, Any] = field(default_factory=dict)  # production, inventory, imports
    demand_data: Dict[str, Any] = field(default_factory=dict)  # consumption, exports, industrial use
    inventory_levels: Optional[Dict[str, float]] = None
    seasonal_adjustment: Optional[float] = None
    weather_impact: Optional[Dict] = None
    geopolitical_risk: Optional[float] = None  # 0-1 scale
    price_forecast: Optional[Dict] = None

@dataclass
class WeatherData:
    """Weather impact on agricultural commodities"""
    commodity: str
    regions: List[str]
    weather_conditions: Dict[str, Any]
    growing_stage: str = ""
    temperature_anomaly: Optional[float] = None
    precipitation_anomaly: Optional[float] = None
    drought_severity: Optional[str] = None  # "None", "Mild", "Moderate", "Severe", "Extreme"
    frost_risk: Optional[float] = None  # 0-1 probability
    harvest_forecast: Optional[Dict] = None
    impact_assessment: str = ""  # "Bullish", "Bearish", "Neutral"

@dataclass
class InventoryReport:
    """Commodity inventory/storage report"""
    commodity: str
    report_type: str  # "Weekly", "Monthly", "Quarterly"
    current_level: float
    previous_level: float
    change: float
    change_percent: float
    seasonal_average: Optional[float] = None
    five_year_average: Optional[float] = None
    days_of_supply: Optional[float] = None
    storage_capacity: Optional[float] = None
    utilization_rate: Optional[float] = None
    report_date: datetime = field(default_factory=datetime.now)

# ==================== DATA SOURCE CONFIGURATIONS ====================

@dataclass
class CommodityDataSource:
    """Commodity data source configuration"""
    name: str
    api_key: Optional[str] = None
    base_url: str = ""
    rate_limit: int = 60
    timeout: int = 10
    enabled: bool = True
    priority: int = 1
    supports_realtime: bool = False
    supports_historical: bool = True
    supports_futures: bool = False
    supports_fundamentals: bool = False
    supported_categories: List[str] = field(default_factory=list)
    cost_per_request: float = 0.0

# ==================== COMMODITY DATA MANAGER ====================

class CommodityDataManager:
    """
    Comprehensive commodity data manager with multiple sources
    
    This class provides real-time and historical commodity data from various sources
    including futures exchanges, ETF proxies, and fundamental data providers.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
        # Data sources
        self.data_sources = self._initialize_data_sources()
        
        # Commodity metadata
        self.commodities = self._initialize_commodity_info()
        
        # ETF mappings for retail access
        self.etf_mappings = self._initialize_etf_mappings()
        
        # Caching
        self.quote_cache: Dict[str, CommodityQuote] = {}
        self.fundamental_cache: Dict[str, SupplyDemandData] = {}
        self.cache_ttl = self.config.get('cache_ttl', 60)  # seconds
        self.cache_lock = threading.RLock()
        
        # Rate limiting
        self.request_counts: Dict[str, List[float]] = {}
        self.rate_limit_lock = threading.Lock()
        
        # Performance tracking
        self.fetch_times: List[float] = []
        self.error_counts: Dict[str, int] = {}
        
        logger.info("🥇 Commodity Data Manager initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration for commodity data manager"""
        return {
            'cache_ttl': 60,
            'max_retries': 3,
            'timeout': 15,
            'rate_limit_buffer': 0.9,
            'preferred_provider': 'yahoo',
            'use_etf_proxies': True,
            'include_futures': True,
            'fundamental_analysis': True,
            'weather_integration': True,
            'inventory_tracking': True,
            'categories': ['Energy', 'Metals', 'Agriculture', 'Livestock']
        }
    
    def _initialize_data_sources(self) -> Dict[str, CommodityDataSource]:
        """Initialize commodity data sources"""
        
        sources = {
            'yahoo': CommodityDataSource(
                name="Yahoo Finance",
                base_url="https://finance.yahoo.com",
                rate_limit=2000,
                priority=1,
                supports_realtime=True,
                supports_historical=True,
                supports_futures=True,
                supported_categories=['Energy', 'Metals', 'Agriculture']
            ),
            'quandl': CommodityDataSource(
                name="Quandl",
                api_key=self.config.get('quandl_api_key'),
                base_url="https://www.quandl.com/api/v3",
                rate_limit=50,
                priority=2,
                supports_realtime=False,
                supports_historical=True,
                supports_fundamentals=True,
                supported_categories=['Energy', 'Metals', 'Agriculture']
            ),
            'alpha_vantage': CommodityDataSource(
                name="Alpha Vantage",
                api_key=self.config.get('alpha_vantage_api_key'),
                base_url="https://www.alphavantage.co/query",
                rate_limit=5,
                priority=3,
                supports_realtime=True,
                supports_historical=True,
                supported_categories=['Energy', 'Metals']
            ),
            'eia': CommodityDataSource(
                name="EIA (Energy Information Administration)",
                api_key=self.config.get('eia_api_key'),
                base_url="https://api.eia.gov/v2",
                rate_limit=1000,
                priority=4,
                supports_realtime=False,
                supports_historical=True,
                supports_fundamentals=True,
                supported_categories=['Energy']
            ),
            'usda': CommodityDataSource(
                name="USDA",
                base_url="https://apps.fas.usda.gov/psdonline/circulars",
                rate_limit=100,
                priority=5,
                supports_realtime=False,
                supports_historical=True,
                supports_fundamentals=True,
                supported_categories=['Agriculture']
            ),
            'cme': CommodityDataSource(
                name="CME Group",
                base_url="https://www.cmegroup.com/market-data",
                rate_limit=1000,
                priority=6,
                supports_realtime=True,
                supports_historical=True,
                supports_futures=True,
                supported_categories=['Energy', 'Metals', 'Agriculture']
            )
        }
        
        # Filter enabled sources
        enabled_sources = {k: v for k, v in sources.items() if v.enabled}
        
        logger.info(f"Initialized {len(enabled_sources)} commodity data sources")
        return enabled_sources
    
    def _initialize_commodity_info(self) -> Dict[str, CommodityInfo]:
        """Initialize commodity information database"""
        
        commodities = {
            # Energy Commodities
            'CL': CommodityInfo(
                symbol='CL',
                name='Crude Oil (WTI)',
                category='Energy',
                subcategory='Petroleum',
                description='West Texas Intermediate crude oil futures',
                unit='per barrel',
                major_exchanges=['NYMEX'],
                contract_months=['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z'],
                tick_size=0.01,
                contract_size='1,000 barrels',
                related_etfs=['USO', 'OIL', 'UCO'],
                supply_factors=['OPEC production', 'Shale production', 'Refinery capacity'],
                demand_factors=['Economic growth', 'Seasonal driving', 'Industrial demand']
            ),
            'BZ': CommodityInfo(
                symbol='BZ',
                name='Crude Oil (Brent)',
                category='Energy',
                subcategory='Petroleum',
                description='Brent crude oil futures',
                unit='per barrel',
                major_exchanges=['ICE'],
                related_etfs=['BNO'],
                supply_factors=['North Sea production', 'OPEC policy', 'Geopolitical events'],
                demand_factors=['European demand', 'Asian imports', 'Refinery margins']
            ),
            'NG': CommodityInfo(
                symbol='NG',
                name='Natural Gas',
                category='Energy',
                subcategory='Natural Gas',
                description='Natural gas futures',
                unit='per MMBtu',
                major_exchanges=['NYMEX'],
                related_etfs=['UNG', 'BOIL', 'KOLD'],
                supply_factors=['Production levels', 'Storage injections', 'LNG exports'],
                demand_factors=['Weather patterns', 'Power generation', 'Industrial use']
            ),
            'HO': CommodityInfo(
                symbol='HO',
                name='Heating Oil',
                category='Energy',
                subcategory='Refined Products',
                description='No. 2 heating oil futures',
                unit='per gallon',
                major_exchanges=['NYMEX'],
                seasonal_patterns={'winter_premium': True, 'summer_discount': True}
            ),
            'RB': CommodityInfo(
                symbol='RB',
                name='Gasoline',
                category='Energy',
                subcategory='Refined Products',
                description='RBOB gasoline futures',
                unit='per gallon',
                major_exchanges=['NYMEX'],
                seasonal_patterns={'summer_driving': True, 'winter_blends': True}
            ),
            
            # Precious Metals
            'GC': CommodityInfo(
                symbol='GC',
                name='Gold',
                category='Metals',
                subcategory='Precious',
                description='Gold futures',
                unit='per troy ounce',
                major_exchanges=['COMEX'],
                related_etfs=['GLD', 'IAU', 'SGOL'],
                supply_factors=['Mining production', 'Central bank sales', 'Recycling'],
                demand_factors=['Investment demand', 'Jewelry', 'Central bank purchases'],
                correlation_assets=['USD', 'Real rates', 'VIX']
            ),
            'SI': CommodityInfo(
                symbol='SI',
                name='Silver',
                category='Metals',
                subcategory='Precious',
                description='Silver futures',
                unit='per troy ounce',
                major_exchanges=['COMEX'],
                related_etfs=['SLV', 'PSLV'],
                supply_factors=['Mining production', 'Recycling', 'Government sales'],
                demand_factors=['Industrial use', 'Investment', 'Photography']
            ),
            'PL': CommodityInfo(
                symbol='PL',
                name='Platinum',
                category='Metals',
                subcategory='Precious',
                description='Platinum futures',
                unit='per troy ounce',
                major_exchanges=['NYMEX'],
                related_etfs=['PPLT'],
                supply_factors=['South African production', 'Russian exports'],
                demand_factors=['Auto catalysts', 'Industrial', 'Investment']
            ),
            'PA': CommodityInfo(
                symbol='PA',
                name='Palladium',
                category='Metals',
                subcategory='Precious',
                description='Palladium futures',
                unit='per troy ounce',
                major_exchanges=['NYMEX'],
                related_etfs=['PALL'],
                supply_factors=['Russian production', 'South African mines'],
                demand_factors=['Auto catalysts', 'Electronics', 'Dental']
            ),
            
            # Base Metals
            'HG': CommodityInfo(
                symbol='HG',
                name='Copper',
                category='Metals',
                subcategory='Industrial',
                description='Copper futures',
                unit='per pound',
                major_exchanges=['COMEX'],
                related_etfs=['CPER', 'JJC'],
                supply_factors=['Mine production', 'Recycling', 'Smelter capacity'],
                demand_factors=['Construction', 'Electronics', 'Transportation'],
                correlation_assets=['Economic growth', 'China PMI', 'USD']
            ),
            
            # Agricultural Commodities
            'C': CommodityInfo(
                symbol='C',
                name='Corn',
                category='Agriculture',
                subcategory='Grains',
                description='Corn futures',
                unit='per bushel',
                major_exchanges=['CBOT'],
                related_etfs=['CORN'],
                supply_factors=['US production', 'Weather', 'Planted acreage'],
                demand_factors=['Ethanol', 'Feed', 'Exports'],
                seasonal_patterns={'planting_season': 'March-May', 'harvest': 'September-November'}
            ),
            'S': CommodityInfo(
                symbol='S',
                name='Soybeans',
                category='Agriculture',
                subcategory='Grains',
                description='Soybean futures',
                unit='per bushel',
                major_exchanges=['CBOT'],
                related_etfs=['SOYB'],
                supply_factors=['US/Brazil production', 'Weather', 'Yields'],
                demand_factors=['China imports', 'Crush margins', 'Meal/oil demand']
            ),
            'W': CommodityInfo(
                symbol='W',
                name='Wheat',
                category='Agriculture',
                subcategory='Grains',
                description='Wheat futures',
                unit='per bushel',
                major_exchanges=['CBOT'],
                related_etfs=['WEAT'],
                supply_factors=['Global production', 'Weather', 'Quality'],
                demand_factors=['Food consumption', 'Feed use', 'Exports']
            ),
            'SB': CommodityInfo(
                symbol='SB',
                name='Sugar',
                category='Agriculture',
                subcategory='Softs',
                description='Sugar No. 11 futures',
                unit='per pound',
                major_exchanges=['ICE'],
                related_etfs=['SGG'],
                supply_factors=['Brazil production', 'Weather', 'Ethanol competition'],
                demand_factors=['Food industry', 'Beverage', 'Emerging markets']
            ),
            'CC': CommodityInfo(
                symbol='CC',
                name='Cocoa',
                category='Agriculture',
                subcategory='Softs',
                description='Cocoa futures',
                unit='per metric ton',
                major_exchanges=['ICE'],
                related_etfs=['NIB'],
                supply_factors=['West Africa production', 'Weather', 'Disease'],
                demand_factors=['Chocolate consumption', 'Grinding', 'Seasonal']
            ),
            'KC': CommodityInfo(
                symbol='KC',
                name='Coffee',
                category='Agriculture',
                subcategory='Softs',
                description='Coffee C futures',
                unit='per pound',
                major_exchanges=['ICE'],
                related_etfs=['JO'],
                supply_factors=['Brazil/Colombia production', 'Weather', 'Frost'],
                demand_factors=['Global consumption', 'Economic growth', 'Arabica premium']
            ),
            'CT': CommodityInfo(
                symbol='CT',
                name='Cotton',
                category='Agriculture',
                subcategory='Softs',
                description='Cotton No. 2 futures',
                unit='per pound',
                major_exchanges=['ICE'],
                related_etfs=['BAL'],
                supply_factors=['US/China production', 'Weather', 'Acreage'],
                demand_factors=['Textile industry', 'China imports', 'Synthetic competition']
            ),
            
            # Livestock
            'LC': CommodityInfo(
                symbol='LC',
                name='Live Cattle',
                category='Livestock',
                subcategory='Cattle',
                description='Live cattle futures',
                unit='per pound',
                major_exchanges=['CME'],
                related_etfs=['COW'],
                supply_factors=['Cattle inventory', 'Feed costs', 'Weather'],
                demand_factors=['Beef consumption', 'Export demand', 'Food service']
            ),
            'FC': CommodityInfo(
                symbol='FC',
                name='Feeder Cattle',
                category='Livestock',
                subcategory='Cattle',
                description='Feeder cattle futures',
                unit='per pound',
                major_exchanges=['CME'],
                supply_factors=['Calf crop', 'Pasture conditions', 'Corn prices'],
                demand_factors=['Feedlot placements', 'Weight gains', 'Beef prices']
            ),
            'LH': CommodityInfo(
                symbol='LH',
                name='Lean Hogs',
                category='Livestock',
                subcategory='Hogs',
                description='Lean hog futures',
                unit='per pound',
                major_exchanges=['CME'],
                supply_factors=['Pig inventory', 'Feed costs', 'Disease'],
                demand_factors=['Pork consumption', 'Export demand', 'Processing capacity']
            )
        }
        
        return commodities
    
    def _initialize_etf_mappings(self) -> Dict[str, str]:
        """Initialize ETF mappings for commodity exposure"""
        
        return {
            # Energy
            'CL': 'USO',    # Crude Oil -> United States Oil Fund
            'NG': 'UNG',    # Natural Gas -> United States Natural Gas Fund
            'BZ': 'BNO',    # Brent Oil -> United States Brent Oil Fund
            
            # Precious Metals
            'GC': 'GLD',    # Gold -> SPDR Gold Trust
            'SI': 'SLV',    # Silver -> iShares Silver Trust
            'PL': 'PPLT',   # Platinum -> Aberdeen Standard Platinum ETF
            'PA': 'PALL',   # Palladium -> Aberdeen Standard Palladium ETF
            
            # Base Metals
            'HG': 'CPER',   # Copper -> United States Copper Index Fund
            
            # Agriculture
            'C': 'CORN',    # Corn -> Teucrium Corn Fund
            'S': 'SOYB',    # Soybeans -> Teucrium Soybean Fund
            'W': 'WEAT',    # Wheat -> Teucrium Wheat Fund
            'SB': 'SGG',    # Sugar -> iShares S&P GSCI Sugar Index Fund
            'CC': 'NIB',    # Cocoa -> iPath Bloomberg Cocoa ETN
            'KC': 'JO',     # Coffee -> iPath Bloomberg Coffee ETN
            'CT': 'BAL',    # Cotton -> iPath Bloomberg Cotton ETN
            
            # Livestock
            'LC': 'COW',    # Live Cattle -> iPath Bloomberg Livestock ETN
            
            # Broad Commodity Exposure
            'DJP': 'DJP',   # iPath Bloomberg Commodity Index ETN
            'DBA': 'DBA',   # Invesco DB Agriculture Fund
            'DBE': 'DBE',   # Invesco DB Energy Fund
            'DBB': 'DBB',   # Invesco DB Base Metals Fund
            'DBP': 'DBP',   # Invesco DB Precious Metals Fund
            'PDBC': 'PDBC', # Invesco Optimum Yield Diversified Commodity
            'COMM': 'COMM', # iShares MSCI KLD 400 Social ETF
            'GSG': 'GSG',   # iShares S&P GSCI Commodity-Indexed Trust
            'RJI': 'RJI',   # Elements Rogers International Commodity Index ETN
        }
    
    async def fetch_real_time_quotes(self, symbols: List[str]) -> Dict[str, MarketData]:
        """
        Fetch real-time commodity quotes
        
        Args:
            symbols: List of commodity symbols or ETF tickers
            
        Returns:
            Dictionary mapping symbols to MarketData objects
        """
        
        start_time = time.time()
        quotes = {}
        
        try:
            # Check cache first
            cached_quotes = self._get_cached_quotes(symbols)
            quotes.update(cached_quotes)
            
            # Identify symbols that need fresh data
            fresh_symbols = [sym for sym in symbols if sym not in quotes]
            
            if fresh_symbols:
                # Separate futures symbols from ETF symbols
                futures_symbols = [s for s in fresh_symbols if s in self.commodities]
                etf_symbols = [s for s in fresh_symbols if s not in self.commodities]
                
                # Add ETF proxies for futures if enabled
                if self.config.get('use_etf_proxies', True):
                    for future_sym in futures_symbols:
                        if future_sym in self.etf_mappings:
                            etf_symbols.append(self.etf_mappings[future_sym])
                
                # Fetch ETF data using Yahoo Finance
                if etf_symbols:
                    etf_quotes = await self._fetch_etf_quotes(etf_symbols)
                    quotes.update(etf_quotes)
                
                # Fetch futures data (if available)
                if futures_symbols and self.config.get('include_futures', True):
                    futures_quotes = await self._fetch_futures_quotes(futures_symbols)
                    quotes.update(futures_quotes)
                
                # Cache the results
                self._cache_quotes(quotes)
            
            # Track performance
            fetch_time = (time.time() - start_time) * 1000
            self.fetch_times.append(fetch_time)
            
            logger.debug(f"Fetched {len(quotes)} commodity quotes in {fetch_time:.2f}ms")
            
            return quotes
            
        except Exception as e:
            logger.error(f"Error fetching commodity quotes: {e}")
            return {}
    
    async def fetch_historical_data(self, symbol: str, period: str = "1Y", 
                                  interval: str = "1d", limit: int = 252) -> pd.DataFrame:
        """
        Fetch historical commodity data
        
        Args:
            symbol: Commodity symbol or ETF ticker
            period: Time period ('1M', '3M', '6M', '1Y', '2Y', '5Y')
            interval: Data interval ('1d', '1wk', '1mo')
            limit: Maximum number of data points
            
        Returns:
            DataFrame with OHLCV data
        """
        
        try:
            # Determine if this is a futures symbol or ETF
            if symbol in self.commodities:
                # Try to fetch futures data
                data = await self._fetch_futures_historical(symbol, period, interval, limit)
                
                if data.empty and self.config.get('use_etf_proxies', True):
                    # Fallback to ETF proxy
                    etf_symbol = self.etf_mappings.get(symbol)
                    if etf_symbol:
                        data = await self._fetch_etf_historical(etf_symbol, period, interval, limit)
            else:
                # Direct ETF fetch
                data = await self._fetch_etf_historical(symbol, period, interval, limit)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def fetch_supply_demand_data(self, commodity: str) -> Optional[SupplyDemandData]:
        """
        Fetch supply and demand fundamentals
        
        Args:
            commodity: Commodity symbol
            
        Returns:
            SupplyDemandData object or None
        """
        
        try:
            # Check cache first
            if commodity in self.fundamental_cache:
                cached_data = self.fundamental_cache[commodity]
                if (datetime.now() - cached_data.report_date).days < 7:  # 1 week cache
                    return cached_data
            
            # Fetch fresh data based on commodity category
            commodity_info = self.commodities.get(commodity)
            if not commodity_info:
                return None
            
            if commodity_info.category == 'Energy':
                data = await self._fetch_energy_fundamentals(commodity)
            elif commodity_info.category == 'Agriculture':
                data = await self._fetch_agricultural_fundamentals(commodity)
            elif commodity_info.category == 'Metals':
                data = await self._fetch_metals_fundamentals(commodity)
            else:
                data = None
            
            if data:
                self.fundamental_cache[commodity] = data
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching supply/demand data for {commodity}: {e}")
            return None
    
    async def fetch_inventory_reports(self, commodity: str) -> List[InventoryReport]:
        """
        Fetch inventory/storage reports
        
        Args:
            commodity: Commodity symbol
            
        Returns:
            List of inventory reports
        """
        
        try:
            commodity_info = self.commodities.get(commodity)
            if not commodity_info:
                return []
            
            if commodity_info.category == 'Energy':
                return await self._fetch_energy_inventories(commodity)
            elif commodity_info.category == 'Agriculture':
                return await self._fetch_agricultural_inventories(commodity)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error fetching inventory reports for {commodity}: {e}")
            return []
    
    async def fetch_weather_data(self, commodity: str) -> Optional[WeatherData]:
        """
        Fetch weather data for agricultural commodities
        
        Args:
            commodity: Agricultural commodity symbol
            
        Returns:
            WeatherData object or None
        """
        
        try:
            commodity_info = self.commodities.get(commodity)
            if not commodity_info or commodity_info.category != 'Agriculture':
                return None
            
            # Mock weather data - would integrate with weather APIs
            weather_data = WeatherData(
                commodity=commodity,
                regions=self._get_production_regions(commodity),
                weather_conditions={
                    'temperature': 'Normal',
                    'precipitation': 'Below Average',
                    'soil_moisture': 'Adequate'
                },
                growing_stage=self._get_growing_stage(commodity),
                temperature_anomaly=0.5,  # +0.5°C above normal
                precipitation_anomaly=-15.0,  # -15% below normal
                drought_severity='Mild',
                frost_risk=0.1,  # 10% probability
                impact_assessment='Neutral'
            )
            
            return weather_data
            
        except Exception as e:
            logger.error(f"Error fetching weather data for {commodity}: {e}")
            return None
    
    def get_commodity_info(self, symbol: str) -> Optional[CommodityInfo]:
        """Get commodity information"""
        return self.commodities.get(symbol.upper())
    
    def get_supported_commodities(self) -> Dict[str, List[str]]:
        """Get supported commodities by category"""
        
        categories = {}
        for symbol, info in self.commodities.items():
            if info.category not in categories:
                categories[info.category] = []
            categories[info.category].append(symbol)
        
        return categories
    
    def get_etf_proxy(self, commodity_symbol: str) -> Optional[str]:
        """Get ETF proxy for commodity"""
        return self.etf_mappings.get(commodity_symbol)
    
    # ==================== PRIVATE METHODS ====================
    
    def _get_cached_quotes(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Get cached quotes that are still valid"""
        
        cached = {}
        current_time = datetime.now()
        
        with self.cache_lock:
            for symbol in symbols:
                if symbol in self.quote_cache:
                    quote = self.quote_cache[symbol]
                    if (current_time - quote.timestamp).seconds < self.cache_ttl:
                        cached[symbol] = self._convert_to_market_data(quote)
        
        return cached
    
    def _cache_quotes(self, quotes: Dict[str, MarketData]):
        """Cache quotes"""
        
        with self.cache_lock:
            for symbol, market_data in quotes.items():
                # Convert MarketData back to CommodityQuote for caching
                commodity_quote = CommodityQuote(
                    symbol=symbol,
                    name=self.commodities.get(symbol, CommodityInfo(symbol, symbol, 'Unknown')).name,
                    category=self.commodities.get(symbol, CommodityInfo(symbol, symbol, 'Unknown')).category,
                    price=market_data.price,
                    currency='USD',
                    unit='per unit',
                    change_24h=market_data.change,
                    change_percent_24h=market_data.change_percent,
                    high_24h=market_data.high or market_data.price,
                    low_24h=market_data.low or market_data.price,
                    volume_24h=market_data.volume,
                    bid=market_data.bid,
                    ask=market_data.ask
                )
                
                self.quote_cache[symbol] = commodity_quote
            
            # Limit cache size
            if len(self.quote_cache) > 500:
                # Remove oldest entries
                sorted_quotes = sorted(
                    self.quote_cache.items(),
                    key=lambda x: x[1].timestamp
                )
                self.quote_cache = dict(sorted_quotes[-400:])
    
    def _convert_to_market_data(self, quote: CommodityQuote) -> MarketData:
        """Convert CommodityQuote to MarketData format"""
        
        return MarketData(
            symbol=quote.symbol,
            asset_type=AssetType.COMMODITY,
            price=quote.price,
            change=quote.change_24h,
            change_percent=quote.change_percent_24h,
            volume=quote.volume_24h or 0.0,
            high=quote.high_24h,
            low=quote.low_24h,
            bid=quote.bid,
            ask=quote.ask,
            timestamp=quote.timestamp,
            # Commodity-specific fields
            unit=quote.unit,
            currency=quote.currency,
            exchange=quote.exchange
        )
    
    async def _fetch_etf_quotes(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Fetch ETF quotes using Yahoo Finance"""
        
        quotes = {}
        
        try:
            # Use yfinance for reliable ETF data
            tickers = yf.Tickers(' '.join(symbols))
            
            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.info
                    hist = ticker.history(period="2d", interval="1d")
                    
                    if not hist.empty:
                        latest = hist.iloc[-1]
                        previous = hist.iloc[-2] if len(hist) > 1 else latest
                        
                        current_price = float(latest['Close'])
                        previous_price = float(previous['Close'])
                        change = current_price - previous_price
                        change_percent = (change / previous_price) * 100 if previous_price > 0 else 0.0
                        
                        market_data = MarketData(
                            symbol=symbol,
                            asset_type=AssetType.COMMODITY,
                            price=current_price,
                            change=change,
                            change_percent=change_percent,
                            volume=float(latest['Volume']),
                            high=float(latest['High']),
                            low=float(latest['Low']),
                            timestamp=datetime.now(),
                            currency='USD'
                        )
                        
                        quotes[symbol] = market_data
                        
                except Exception as e:
                    logger.warning(f"Error fetching ETF data for {symbol}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"ETF data fetch error: {e}")
        
        return quotes
    
    async def _fetch_futures_quotes(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Fetch futures quotes (placeholder implementation)"""
        
        # This would integrate with futures data providers like CME, ICE, etc.
        # For now, return empty dict as futures data requires specialized APIs
        logger.info(f"Futures data fetching not yet implemented for {symbols}")
        return {}
    
    async def _fetch_etf_historical(self, symbol: str, period: str, 
                                  interval: str, limit: int) -> pd.DataFrame:
        """Fetch historical ETF data"""
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Convert period format
            yf_period = period.lower().replace('y', 'y').replace('m', 'mo')
            if yf_period not in ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']:
                yf_period = '1y'  # Default fallback
            
            hist = ticker.history(period=yf_period, interval=interval)
            
            if not hist.empty:
                # Standardize column names
                hist = hist.rename(columns={
                    'Open': 'open',
                    'High': 'high', 
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # Add timestamp column
                hist['timestamp'] = hist.index
                hist = hist.reset_index(drop=True)
                
                # Limit results
                if len(hist) > limit:
                    hist = hist.tail(limit)
                
                return hist
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching ETF historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _fetch_futures_historical(self, symbol: str, period: str, 
                                      interval: str, limit: int) -> pd.DataFrame:
        """Fetch historical futures data (placeholder)"""
        
        # Placeholder for futures historical data
        # Would integrate with CME, ICE, or other futures data providers
        return pd.DataFrame()
    
    async def _fetch_energy_fundamentals(self, commodity: str) -> Optional[SupplyDemandData]:
        """Fetch energy commodity fundamentals"""
        
        # Mock implementation - would integrate with EIA API
        supply_data = {
            'production': 12000000,  # barrels/day
            'imports': 6000000,
            'inventory': 400000000,  # barrels
            'refinery_utilization': 0.85
        }
        
        demand_data = {
            'consumption': 20000000,  # barrels/day
            'exports': 2000000,
            'gasoline_demand': 9000000,
            'distillate_demand': 4000000
        }
        
        return SupplyDemandData(
            commodity=commodity,
            report_date=datetime.now(),
            supply_data=supply_data,
            demand_data=demand_data,
            inventory_levels={'total': 400000000, 'spr': 650000000},
            geopolitical_risk=0.3
        )
    
    async def _fetch_agricultural_fundamentals(self, commodity: str) -> Optional[SupplyDemandData]:
        """Fetch agricultural commodity fundamentals"""
        
        # Mock implementation - would integrate with USDA data
        supply_data = {
            'production': 14000,  # million bushels
            'imports': 25,
            'beginning_stocks': 1100,
            'total_supply': 15125
        }
        
        demand_data = {
            'feed_use': 5500,
            'food_seed_industrial': 6500,
            'exports': 2400,
            'total_demand': 14400
        }
        
        return SupplyDemandData(
            commodity=commodity,
            report_date=datetime.now(),
            supply_data=supply_data,
            demand_data=demand_data,
            inventory_levels={'ending_stocks': 725},
            weather_impact={'drought_risk': 0.2, 'frost_risk': 0.1}
        )
    
    async def _fetch_metals_fundamentals(self, commodity: str) -> Optional[SupplyDemandData]:
        """Fetch metals commodity fundamentals"""
        
        # Mock implementation - would integrate with metals industry data
        supply_data = {
            'mine_production': 3000,  # metric tons
            'recycling': 1200,
            'imports': 800,
            'total_supply': 5000
        }
        
        demand_data = {
            'industrial': 2800,
            'jewelry': 1500,
            'investment': 400,
            'technology': 300
        }
        
        return SupplyDemandData(
            commodity=commodity,
            report_date=datetime.now(),
            supply_data=supply_data,
            demand_data=demand_data,
            inventory_levels={'exchange_stocks': 150, 'etf_holdings': 2000}
        )
    
    async def _fetch_energy_inventories(self, commodity: str) -> List[InventoryReport]:
        """Fetch energy inventory reports"""
        
        # Mock EIA weekly petroleum status report
        reports = [
            InventoryReport(
                commodity=commodity,
                report_type='Weekly',
                current_level=400.5,  # million barrels
                previous_level=398.2,
                change=2.3,
                change_percent=0.58,
                seasonal_average=420.0,
                five_year_average=450.0,
                days_of_supply=25.2
            )
        ]
        
        return reports
    
    async def _fetch_agricultural_inventories(self, commodity: str) -> List[InventoryReport]:
        """Fetch agricultural inventory reports"""
        
        # Mock USDA grain stocks report
        reports = [
            InventoryReport(
                commodity=commodity,
                report_type='Quarterly',
                current_level=1100.0,  # million bushels
                previous_level=1250.0,
                change=-150.0,
                change_percent=-12.0,
                seasonal_average=1000.0,
                five_year_average=1050.0
            )
        ]
        
        return reports
    
    def _get_production_regions(self, commodity: str) -> List[str]:
        """Get major production regions for commodity"""
        
        region_mapping = {
            'C': ['Iowa', 'Illinois', 'Nebraska', 'Minnesota'],  # Corn
            'S': ['Illinois', 'Iowa', 'Minnesota', 'Indiana'],   # Soybeans
            'W': ['Kansas', 'North Dakota', 'Montana', 'Washington'],  # Wheat
            'SB': ['Brazil', 'India', 'China', 'Thailand'],     # Sugar
            'CC': ['Ivory Coast', 'Ghana', 'Ecuador', 'Brazil'], # Cocoa
            'KC': ['Brazil', 'Vietnam', 'Colombia', 'Indonesia'], # Coffee
            'CT': ['China', 'India', 'United States', 'Pakistan'] # Cotton
        }
        
        return region_mapping.get(commodity, ['Unknown'])
    
    def _get_growing_stage(self, commodity: str) -> str:
        """Get current growing stage for agricultural commodity"""
        
        # Simplified - would use actual calendar and location data
        current_month = datetime.now().month
        
        if commodity in ['C', 'S']:  # Corn, Soybeans
            if current_month in [3, 4, 5]:
                return 'Planting'
            elif current_month in [6, 7, 8]:
                return 'Growing'
            elif current_month in [9, 10, 11]:
                return 'Harvest'
            else:
                return 'Dormant'
        
        return 'Unknown'

# ==================== TESTING ====================

def test_commodity_data_manager():
    """Test commodity data manager functionality"""
    
    print("🥇 Testing Commodity Data Manager")
    print("=" * 50)
    
    # Create commodity data manager
    commodity_manager = CommodityDataManager()
    
    # Test commodity info
    gold_info = commodity_manager.get_commodity_info('GC')
    print(f"✅ Gold Info: {gold_info.name} - {gold_info.category}")
    print(f"   Unit: {gold_info.unit}")
    print(f"   Exchanges: {gold_info.major_exchanges}")
    print(f"   Related ETFs: {gold_info.related_etfs}")
    
    # Test supported commodities
    supported = commodity_manager.get_supported_commodities()
    print(f"\n✅ Supported commodities by category:")
    for category, symbols in supported.items():
        print(f"   {category}: {len(symbols)} commodities")
    
    # Test ETF mappings
    gold_etf = commodity_manager.get_etf_proxy('GC')
    oil_etf = commodity_manager.get_etf_proxy('CL')
    print(f"\n✅ ETF Proxies:")
    print(f"   Gold (GC) -> {gold_etf}")
    print(f"   Oil (CL) -> {oil_etf}")
    
    # Test data sources
    print(f"\n✅ Data sources initialized: {len(commodity_manager.data_sources)}")
    for name, source in commodity_manager.data_sources.items():
        print(f"   - {name}: Priority {source.priority}, Categories: {source.supported_categories}")
    
    print("\n🎉 Commodity data manager tests completed!")

if __name__ == "__main__":
    test_commodity_data_manager()