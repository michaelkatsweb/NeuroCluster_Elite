#!/usr/bin/env python3
"""
File: market_scanner.py
Path: NeuroCluster-Elite/src/analysis/market_scanner.py
Description: Advanced market scanning system for opportunity detection

This module implements comprehensive market scanning capabilities to identify
trading opportunities, unusual activity, breakouts, and market anomalies across
multiple asset classes using the NeuroCluster algorithm.

Features:
- Real-time market scanning across multiple asset types
- Technical pattern recognition and breakout detection
- Volume and price anomaly detection
- Momentum and trend analysis
- Volatility spike detection
- Gap analysis and unusual market movements
- Integration with NeuroCluster regime detection
- Multi-timeframe analysis and screening
- Custom screening criteria and filters
- Alert generation for discovered opportunities

Author: Your Name
Created: 2025-06-29
Version: 1.0.0
License: MIT
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import time
import sqlite3
from pathlib import Path
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import math

# Technical analysis
import talib
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Import our modules
try:
    from src.core.neurocluster_elite import NeuroClusterElite, RegimeType, AssetType, MarketData
    from src.data.multi_asset_manager import MultiAssetDataManager
    from src.analysis.technical_indicators import TechnicalIndicators
    from src.utils.config_manager import ConfigManager
    from src.utils.logger import get_enhanced_logger, LogCategory
    from src.utils.helpers import format_currency, format_percentage, calculate_sharpe_ratio
except ImportError:
    # Fallback for testing
    pass

# Configure logging
logger = get_enhanced_logger(__name__, LogCategory.ANALYSIS)

# ==================== ENUMS AND DATA STRUCTURES ====================

class ScanType(Enum):
    """Types of market scans"""
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    VOLUME_SPIKE = "volume_spike"
    GAP_UP = "gap_up"
    GAP_DOWN = "gap_down"
    UNUSUAL_ACTIVITY = "unusual_activity"
    REGIME_CHANGE = "regime_change"
    VOLATILITY_SPIKE = "volatility_spike"
    PATTERN_RECOGNITION = "pattern_recognition"
    ARBITRAGE = "arbitrage"
    EARNINGS_MOMENTUM = "earnings_momentum"
    NEWS_CATALYST = "news_catalyst"

class AlertLevel(Enum):
    """Alert levels for scan results"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ScanStatus(Enum):
    """Status of scan operations"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class PatternType(Enum):
    """Technical pattern types"""
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    FLAG = "flag"
    PENNANT = "pennant"
    WEDGE = "wedge"
    CHANNEL = "channel"
    RECTANGLE = "rectangle"

@dataclass
class ScanResult:
    """Market scan result"""
    symbol: str
    asset_type: AssetType
    scan_type: ScanType
    alert_level: AlertLevel
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    
    # Market data
    current_price: float
    price_change: float
    price_change_pct: float
    volume: float
    volume_change_pct: float
    
    # Technical indicators
    rsi: Optional[float] = None
    macd: Optional[float] = None
    bollinger_position: Optional[float] = None
    
    # Regime information
    current_regime: Optional[RegimeType] = None
    regime_confidence: float = 0.0
    
    # Pattern information
    pattern_type: Optional[PatternType] = None
    pattern_confidence: float = 0.0
    
    # Analysis details
    description: str = ""
    reasoning: str = ""
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    
    # Metadata
    scan_duration: float = 0.0
    data_quality: float = 1.0
    scan_id: str = field(default_factory=lambda: str(int(time.time() * 1000)))

@dataclass
class ScanCriteria:
    """Criteria for market scanning"""
    asset_types: List[AssetType] = field(default_factory=lambda: [AssetType.STOCK, AssetType.CRYPTO])
    scan_types: List[ScanType] = field(default_factory=lambda: [ScanType.BREAKOUT, ScanType.MOMENTUM])
    min_price: float = 1.0
    max_price: float = 1000000.0
    min_volume: float = 0.0
    min_market_cap: float = 0.0
    min_score: float = 0.5
    min_confidence: float = 0.6
    timeframes: List[str] = field(default_factory=lambda: ["1h", "4h", "1d"])
    max_results: int = 50
    
    # Technical filters
    rsi_min: Optional[float] = None
    rsi_max: Optional[float] = None
    volume_spike_threshold: float = 2.0
    price_change_min: Optional[float] = None
    price_change_max: Optional[float] = None

@dataclass
class ScanConfiguration:
    """Scanner configuration"""
    name: str
    criteria: ScanCriteria
    enabled: bool = True
    schedule: Optional[str] = None  # Cron-like schedule
    alerts_enabled: bool = True
    last_run: Optional[datetime] = None
    total_runs: int = 0
    successful_runs: int = 0

@dataclass
class MarketScannerStats:
    """Scanner statistics"""
    total_scans: int = 0
    successful_scans: int = 0
    failed_scans: int = 0
    total_symbols_scanned: int = 0
    total_opportunities_found: int = 0
    high_priority_alerts: int = 0
    average_scan_time: float = 0.0
    last_scan_time: Optional[datetime] = None

# ==================== MARKET SCANNER ====================

class AdvancedMarketScanner:
    """Advanced market scanning system with multiple scan types"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize market scanner
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize core components
        self.neurocluster = None
        self.data_manager = None
        self.technical_indicators = None
        
        # Scanner configuration
        self.scan_configurations = {}
        self.active_scans = {}
        self.scan_results_cache = {}
        
        # Database and storage
        self.db_path = self.config.get('db_path', 'data/market_scanner.db')
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes
        
        # Performance settings
        self.max_concurrent_scans = self.config.get('max_concurrent_scans', 10)
        self.scan_timeout = self.config.get('scan_timeout', 60)  # seconds
        
        # Statistics
        self.stats = MarketScannerStats()
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_scans)
        self.scan_lock = threading.Lock()
        
        # Default scan configurations
        self._create_default_scan_configs()
        
        # Initialize components
        self._initialize_components()
        
        # Initialize database
        self._initialize_database()
        
        logger.info("🔍 Advanced Market Scanner initialized")
    
    def _initialize_components(self):
        """Initialize core components"""
        
        try:
            # Initialize NeuroCluster algorithm
            self.neurocluster = NeuroClusterElite(self.config.get('neurocluster', {}))
            
            # Initialize data manager
            self.data_manager = MultiAssetDataManager(self.config.get('data', {}))
            
            # Initialize technical indicators
            self.technical_indicators = TechnicalIndicators()
            
            logger.info("📊 Scanner components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize scanner components: {e}")
    
    def _create_default_scan_configs(self):
        """Create default scan configurations"""
        
        # Breakout scanner
        breakout_criteria = ScanCriteria(
            scan_types=[ScanType.BREAKOUT],
            min_volume=100000,
            volume_spike_threshold=1.5,
            min_score=0.7,
            timeframes=["1h", "4h"]
        )
        
        self.scan_configurations["breakout_scanner"] = ScanConfiguration(
            name="Breakout Scanner",
            criteria=breakout_criteria,
            schedule="*/15 * * * *"  # Every 15 minutes
        )
        
        # Momentum scanner
        momentum_criteria = ScanCriteria(
            scan_types=[ScanType.MOMENTUM],
            rsi_min=60,
            rsi_max=80,
            price_change_min=0.02,  # 2% minimum change
            min_score=0.6
        )
        
        self.scan_configurations["momentum_scanner"] = ScanConfiguration(
            name="Momentum Scanner",
            criteria=momentum_criteria,
            schedule="*/30 * * * *"  # Every 30 minutes
        )
        
        # Volume spike scanner
        volume_criteria = ScanCriteria(
            scan_types=[ScanType.VOLUME_SPIKE],
            volume_spike_threshold=3.0,
            min_score=0.8
        )
        
        self.scan_configurations["volume_scanner"] = ScanConfiguration(
            name="Volume Spike Scanner",
            criteria=volume_criteria,
            schedule="*/10 * * * *"  # Every 10 minutes
        )
        
        # Gap scanner
        gap_criteria = ScanCriteria(
            scan_types=[ScanType.GAP_UP, ScanType.GAP_DOWN],
            price_change_min=0.05,  # 5% minimum gap
            min_score=0.7
        )
        
        self.scan_configurations["gap_scanner"] = ScanConfiguration(
            name="Gap Scanner",
            criteria=gap_criteria,
            schedule="0 9,16 * * 1-5"  # Market open and close
        )
    
    def _initialize_database(self):
        """Initialize scanner database"""
        
        try:
            # Create data directory
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create database tables
            with sqlite3.connect(self.db_path) as conn:
                # Scan results table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS scan_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        scan_id TEXT UNIQUE NOT NULL,
                        symbol TEXT NOT NULL,
                        asset_type TEXT NOT NULL,
                        scan_type TEXT NOT NULL,
                        alert_level TEXT NOT NULL,
                        score REAL NOT NULL,
                        confidence REAL NOT NULL,
                        timestamp DATETIME NOT NULL,
                        current_price REAL,
                        price_change REAL,
                        price_change_pct REAL,
                        volume REAL,
                        volume_change_pct REAL,
                        rsi REAL,
                        macd REAL,
                        current_regime TEXT,
                        regime_confidence REAL,
                        pattern_type TEXT,
                        pattern_confidence REAL,
                        description TEXT,
                        reasoning TEXT,
                        target_price REAL,
                        stop_loss REAL,
                        risk_reward_ratio REAL,
                        data_quality REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Scan configurations table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS scan_configs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        config_data TEXT NOT NULL,
                        enabled BOOLEAN DEFAULT TRUE,
                        last_run DATETIME,
                        total_runs INTEGER DEFAULT 0,
                        successful_runs INTEGER DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes
                conn.execute('CREATE INDEX IF NOT EXISTS idx_results_timestamp ON scan_results(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_results_symbol ON scan_results(symbol)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_results_scan_type ON scan_results(scan_type)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_results_alert_level ON scan_results(alert_level)')
                
                logger.info("📊 Scanner database initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize scanner database: {e}")
    
    async def run_scan(self, scan_name: str = None, criteria: ScanCriteria = None, 
                      symbols: List[str] = None) -> List[ScanResult]:
        """
        Run market scan with specified criteria
        
        Args:
            scan_name: Name of predefined scan configuration
            criteria: Custom scan criteria
            symbols: Specific symbols to scan (optional)
            
        Returns:
            List of scan results
        """
        
        start_time = time.time()
        
        try:
            # Get scan criteria
            if scan_name and scan_name in self.scan_configurations:
                criteria = self.scan_configurations[scan_name].criteria
                config = self.scan_configurations[scan_name]
            elif criteria is None:
                criteria = ScanCriteria()  # Default criteria
                config = None
            
            # Generate scan ID
            scan_id = f"scan_{int(time.time() * 1000)}"
            
            with self.scan_lock:
                self.active_scans[scan_id] = {
                    'status': ScanStatus.RUNNING,
                    'start_time': start_time,
                    'criteria': criteria
                }
            
            logger.info(f"🔍 Starting market scan: {scan_name or 'custom'}")
            
            # Get symbols to scan
            if symbols is None:
                symbols = await self._get_scan_universe(criteria)
            
            # Filter symbols by criteria
            filtered_symbols = await self._filter_symbols(symbols, criteria)
            
            logger.info(f"📊 Scanning {len(filtered_symbols)} symbols")
            
            # Run scans on filtered symbols
            scan_results = []
            
            # Process symbols in batches
            batch_size = min(50, self.max_concurrent_scans)
            
            for i in range(0, len(filtered_symbols), batch_size):
                batch = filtered_symbols[i:i + batch_size]
                
                # Create scan tasks
                tasks = []
                for symbol in batch:
                    for scan_type in criteria.scan_types:
                        task = self._scan_symbol(symbol, scan_type, criteria)
                        tasks.append(task)
                
                # Execute batch
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in batch_results:
                    if isinstance(result, ScanResult):
                        # Apply filters
                        if (result.score >= criteria.min_score and 
                            result.confidence >= criteria.min_confidence):
                            scan_results.append(result)
                    elif isinstance(result, Exception):
                        logger.warning(f"Scan task failed: {result}")
                        self.stats.failed_scans += 1
            
            # Sort results by score and alert level
            scan_results.sort(key=lambda x: (x.alert_level.value, -x.score))
            
            # Limit results
            scan_results = scan_results[:criteria.max_results]
            
            # Store results
            await self._store_scan_results(scan_results)
            
            # Update statistics
            scan_duration = time.time() - start_time
            self.stats.total_scans += 1
            self.stats.successful_scans += 1
            self.stats.total_symbols_scanned += len(filtered_symbols)
            self.stats.total_opportunities_found += len(scan_results)
            self.stats.high_priority_alerts += len([r for r in scan_results if r.alert_level in [AlertLevel.HIGH, AlertLevel.CRITICAL]])
            self.stats.average_scan_time = (self.stats.average_scan_time * (self.stats.successful_scans - 1) + scan_duration) / self.stats.successful_scans
            self.stats.last_scan_time = datetime.now()
            
            # Update scan configuration
            if config:
                config.last_run = datetime.now()
                config.total_runs += 1
                config.successful_runs += 1
            
            # Update scan status
            with self.scan_lock:
                self.active_scans[scan_id]['status'] = ScanStatus.COMPLETED
                self.active_scans[scan_id]['results'] = scan_results
                self.active_scans[scan_id]['duration'] = scan_duration
            
            logger.info(f"✅ Scan completed: {len(scan_results)} opportunities found in {scan_duration:.2f}s")
            
            return scan_results
            
        except Exception as e:
            logger.error(f"Market scan failed: {e}")
            
            # Update statistics
            self.stats.failed_scans += 1
            
            # Update scan status
            with self.scan_lock:
                if scan_id in self.active_scans:
                    self.active_scans[scan_id]['status'] = ScanStatus.FAILED
                    self.active_scans[scan_id]['error'] = str(e)
            
            return []
    
    async def _get_scan_universe(self, criteria: ScanCriteria) -> List[str]:
        """Get universe of symbols to scan based on criteria"""
        
        try:
            # This would typically fetch from a symbol universe database
            # For now, return a default set of popular symbols
            
            stock_symbols = [
                'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
                'AMD', 'INTC', 'CRM', 'PYPL', 'ADBE', 'QCOM', 'AVGO', 'TXN',
                'ORCL', 'NOW', 'INTU', 'AMAT', 'MU', 'ADI', 'LRCX', 'KLAC'
            ]
            
            crypto_symbols = [
                'BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD',
                'AVAX-USD', 'MATIC-USD', 'LINK-USD', 'UNI-USD', 'ATOM-USD'
            ]
            
            forex_symbols = [
                'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
                'NZDUSD', 'USDCHF', 'EURGBP', 'EURJPY', 'GBPJPY'
            ]
            
            symbols = []
            
            if AssetType.STOCK in criteria.asset_types:
                symbols.extend(stock_symbols)
            
            if AssetType.CRYPTO in criteria.asset_types:
                symbols.extend(crypto_symbols)
            
            if AssetType.FOREX in criteria.asset_types:
                symbols.extend(forex_symbols)
            
            return symbols
            
        except Exception as e:
            logger.error(f"Failed to get scan universe: {e}")
            return []
    
    async def _filter_symbols(self, symbols: List[str], criteria: ScanCriteria) -> List[str]:
        """Filter symbols based on basic criteria"""
        
        filtered_symbols = []
        
        try:
            # Get market data for all symbols
            market_data = {}
            
            for asset_type in criteria.asset_types:
                type_symbols = [s for s in symbols if self._get_symbol_asset_type(s) == asset_type]
                if type_symbols:
                    data = await self.data_manager.fetch_market_data(type_symbols, asset_type)
                    market_data.update(data)
            
            # Apply filters
            for symbol, data in market_data.items():
                if data is None:
                    continue
                
                # Price filters
                if data.price < criteria.min_price or data.price > criteria.max_price:
                    continue
                
                # Volume filter
                if data.volume < criteria.min_volume:
                    continue
                
                # Price change filters
                if criteria.price_change_min is not None and data.change_percent < criteria.price_change_min * 100:
                    continue
                
                if criteria.price_change_max is not None and data.change_percent > criteria.price_change_max * 100:
                    continue
                
                filtered_symbols.append(symbol)
            
            return filtered_symbols
            
        except Exception as e:
            logger.warning(f"Symbol filtering failed: {e}")
            return symbols  # Return unfiltered if filtering fails
    
    def _get_symbol_asset_type(self, symbol: str) -> AssetType:
        """Determine asset type from symbol"""
        
        if '-USD' in symbol or 'USD' in symbol:
            return AssetType.CRYPTO
        elif any(forex in symbol for forex in ['EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']):
            return AssetType.FOREX
        else:
            return AssetType.STOCK
    
    async def _scan_symbol(self, symbol: str, scan_type: ScanType, 
                          criteria: ScanCriteria) -> Optional[ScanResult]:
        """Scan individual symbol for specific pattern/opportunity"""
        
        try:
            # Get market data
            asset_type = self._get_symbol_asset_type(symbol)
            market_data = await self.data_manager.fetch_market_data([symbol], asset_type)
            
            if symbol not in market_data or market_data[symbol] is None:
                return None
            
            data = market_data[symbol]
            
            # Get historical data for technical analysis
            historical_data = await self._get_historical_data(symbol, asset_type)
            
            if historical_data is None or len(historical_data) < 20:
                return None
            
            # Detect regime using NeuroCluster
            regime, regime_confidence = None, 0.0
            if self.neurocluster:
                try:
                    regime, regime_confidence = self.neurocluster.detect_regime({symbol: data})
                except Exception as e:
                    logger.warning(f"Regime detection failed for {symbol}: {e}")
            
            # Run specific scan type
            if scan_type == ScanType.BREAKOUT:
                return await self._scan_breakout(symbol, asset_type, data, historical_data, regime, regime_confidence)
            elif scan_type == ScanType.MOMENTUM:
                return await self._scan_momentum(symbol, asset_type, data, historical_data, regime, regime_confidence)
            elif scan_type == ScanType.VOLUME_SPIKE:
                return await self._scan_volume_spike(symbol, asset_type, data, historical_data, criteria)
            elif scan_type == ScanType.GAP_UP or scan_type == ScanType.GAP_DOWN:
                return await self._scan_gaps(symbol, asset_type, data, historical_data, scan_type)
            elif scan_type == ScanType.REVERSAL:
                return await self._scan_reversal(symbol, asset_type, data, historical_data, regime, regime_confidence)
            elif scan_type == ScanType.PATTERN_RECOGNITION:
                return await self._scan_patterns(symbol, asset_type, data, historical_data)
            elif scan_type == ScanType.VOLATILITY_SPIKE:
                return await self._scan_volatility_spike(symbol, asset_type, data, historical_data)
            else:
                logger.warning(f"Scan type {scan_type.value} not implemented")
                return None
                
        except Exception as e:
            logger.warning(f"Symbol scan failed for {symbol}: {e}")
            return None
    
    async def _get_historical_data(self, symbol: str, asset_type: AssetType) -> Optional[pd.DataFrame]:
        """Get historical price data for technical analysis"""
        
        try:
            # This would fetch historical data from the data manager
            # For now, return simulated data
            
            dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')
            
            # Generate realistic price data
            np.random.seed(hash(symbol) % 2**32)
            returns = np.random.normal(0.001, 0.02, 100)
            prices = 100 * np.exp(np.cumsum(returns))
            volumes = np.random.lognormal(10, 1, 100)
            
            df = pd.DataFrame({
                'timestamp': dates,
                'open': prices * np.random.uniform(0.99, 1.01, 100),
                'high': prices * np.random.uniform(1.00, 1.05, 100),
                'low': prices * np.random.uniform(0.95, 1.00, 100),
                'close': prices,
                'volume': volumes
            })
            
            return df
            
        except Exception as e:
            logger.warning(f"Failed to get historical data for {symbol}: {e}")
            return None
    
    async def _scan_breakout(self, symbol: str, asset_type: AssetType, data: MarketData,
                           historical_data: pd.DataFrame, regime: RegimeType, 
                           regime_confidence: float) -> Optional[ScanResult]:
        """Scan for breakout patterns"""
        
        try:
            prices = historical_data['close'].values
            volumes = historical_data['volume'].values
            
            # Calculate resistance level (recent highs)
            resistance = np.max(prices[-20:])  # 20-period high
            
            # Check if current price is breaking above resistance
            current_price = data.price
            breakout_threshold = resistance * 1.005  # 0.5% above resistance
            
            if current_price <= breakout_threshold:
                return None
            
            # Calculate volume confirmation
            avg_volume = np.mean(volumes[-20:])
            volume_spike = data.volume / avg_volume if avg_volume > 0 else 1
            
            # Calculate RSI for momentum confirmation
            rsi = self.technical_indicators.calculate_rsi(prices)[-1] if len(prices) > 14 else 50
            
            # Calculate score
            breakout_strength = (current_price - resistance) / resistance
            volume_score = min(1.0, volume_spike / 2.0)
            momentum_score = (rsi - 50) / 50 if rsi > 50 else 0
            
            score = (breakout_strength * 0.4 + volume_score * 0.4 + momentum_score * 0.2)
            score = max(0.0, min(1.0, score))
            
            if score < 0.5:
                return None
            
            # Determine alert level
            if score > 0.8 and volume_spike > 3:
                alert_level = AlertLevel.HIGH
            elif score > 0.6 and volume_spike > 2:
                alert_level = AlertLevel.MEDIUM
            else:
                alert_level = AlertLevel.LOW
            
            # Calculate targets
            target_price = current_price * 1.05  # 5% target
            stop_loss = resistance * 0.98  # 2% below resistance
            risk_reward = (target_price - current_price) / (current_price - stop_loss) if current_price > stop_loss else 0
            
            return ScanResult(
                symbol=symbol,
                asset_type=asset_type,
                scan_type=ScanType.BREAKOUT,
                alert_level=alert_level,
                score=score,
                confidence=min(regime_confidence, volume_score),
                timestamp=datetime.now(),
                current_price=current_price,
                price_change=data.change,
                price_change_pct=data.change_percent,
                volume=data.volume,
                volume_change_pct=(volume_spike - 1) * 100,
                rsi=rsi,
                current_regime=regime,
                regime_confidence=regime_confidence,
                description=f"Breakout above resistance at {format_currency(resistance)}",
                reasoning=f"Price broke above {format_currency(resistance)} with {volume_spike:.1f}x volume",
                target_price=target_price,
                stop_loss=stop_loss,
                risk_reward_ratio=risk_reward
            )
            
        except Exception as e:
            logger.warning(f"Breakout scan failed for {symbol}: {e}")
            return None
    
    async def _scan_momentum(self, symbol: str, asset_type: AssetType, data: MarketData,
                           historical_data: pd.DataFrame, regime: RegimeType, 
                           regime_confidence: float) -> Optional[ScanResult]:
        """Scan for momentum patterns"""
        
        try:
            prices = historical_data['close'].values
            volumes = historical_data['volume'].values
            
            # Calculate technical indicators
            rsi = self.technical_indicators.calculate_rsi(prices)[-1] if len(prices) > 14 else 50
            macd, signal, histogram = self.technical_indicators.calculate_macd(prices)
            
            current_macd = macd[-1] if len(macd) > 0 else 0
            current_signal = signal[-1] if len(signal) > 0 else 0
            
            # Check momentum conditions
            momentum_conditions = {
                'rsi_momentum': 60 < rsi < 80,  # Strong but not overbought
                'macd_bullish': current_macd > current_signal,
                'price_momentum': data.change_percent > 2.0,  # 2% price increase
                'volume_confirmation': data.volume > np.mean(volumes[-10:]) * 1.2
            }
            
            conditions_met = sum(momentum_conditions.values())
            
            if conditions_met < 2:
                return None
            
            # Calculate momentum score
            rsi_score = (rsi - 50) / 30 if rsi > 50 else 0  # Normalize RSI
            macd_score = 1.0 if momentum_conditions['macd_bullish'] else 0.0
            price_score = min(1.0, data.change_percent / 10.0)  # Normalize price change
            volume_score = 1.0 if momentum_conditions['volume_confirmation'] else 0.5
            
            score = (rsi_score * 0.3 + macd_score * 0.2 + price_score * 0.3 + volume_score * 0.2)
            score = max(0.0, min(1.0, score))
            
            if score < 0.5:
                return None
            
            # Determine alert level
            if conditions_met >= 4 and score > 0.8:
                alert_level = AlertLevel.HIGH
            elif conditions_met >= 3 and score > 0.6:
                alert_level = AlertLevel.MEDIUM
            else:
                alert_level = AlertLevel.LOW
            
            return ScanResult(
                symbol=symbol,
                asset_type=asset_type,
                scan_type=ScanType.MOMENTUM,
                alert_level=alert_level,
                score=score,
                confidence=regime_confidence,
                timestamp=datetime.now(),
                current_price=data.price,
                price_change=data.change,
                price_change_pct=data.change_percent,
                volume=data.volume,
                rsi=rsi,
                macd=current_macd,
                current_regime=regime,
                regime_confidence=regime_confidence,
                description=f"Strong momentum with {conditions_met}/4 conditions met",
                reasoning=f"RSI: {rsi:.1f}, MACD bullish: {momentum_conditions['macd_bullish']}, Price up {data.change_percent:.1f}%"
            )
            
        except Exception as e:
            logger.warning(f"Momentum scan failed for {symbol}: {e}")
            return None
    
    async def _scan_volume_spike(self, symbol: str, asset_type: AssetType, data: MarketData,
                               historical_data: pd.DataFrame, criteria: ScanCriteria) -> Optional[ScanResult]:
        """Scan for volume spikes"""
        
        try:
            volumes = historical_data['volume'].values
            
            # Calculate average volume
            avg_volume_20 = np.mean(volumes[-20:])
            avg_volume_50 = np.mean(volumes[-50:]) if len(volumes) >= 50 else avg_volume_20
            
            # Check volume spike
            current_volume = data.volume
            volume_spike_20 = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
            volume_spike_50 = current_volume / avg_volume_50 if avg_volume_50 > 0 else 1
            
            if volume_spike_20 < criteria.volume_spike_threshold:
                return None
            
            # Calculate score based on volume spike magnitude
            spike_score = min(1.0, volume_spike_20 / 5.0)  # Normalize to max 5x volume
            consistency_score = min(1.0, volume_spike_50 / volume_spike_20)  # Consistency across timeframes
            
            score = spike_score * 0.7 + consistency_score * 0.3
            
            # Price movement confirmation
            price_movement_score = min(1.0, abs(data.change_percent) / 5.0)  # Normalize to 5%
            score = score * 0.8 + price_movement_score * 0.2
            
            if score < 0.5:
                return None
            
            # Determine alert level
            if volume_spike_20 > 5 and abs(data.change_percent) > 3:
                alert_level = AlertLevel.CRITICAL
            elif volume_spike_20 > 3 and abs(data.change_percent) > 2:
                alert_level = AlertLevel.HIGH
            elif volume_spike_20 > 2:
                alert_level = AlertLevel.MEDIUM
            else:
                alert_level = AlertLevel.LOW
            
            return ScanResult(
                symbol=symbol,
                asset_type=asset_type,
                scan_type=ScanType.VOLUME_SPIKE,
                alert_level=alert_level,
                score=score,
                confidence=0.8,  # Volume spikes are relatively reliable
                timestamp=datetime.now(),
                current_price=data.price,
                price_change=data.change,
                price_change_pct=data.change_percent,
                volume=current_volume,
                volume_change_pct=(volume_spike_20 - 1) * 100,
                description=f"Volume spike: {volume_spike_20:.1f}x average",
                reasoning=f"Current volume {format_currency(current_volume)} vs 20-day avg {format_currency(avg_volume_20)}"
            )
            
        except Exception as e:
            logger.warning(f"Volume spike scan failed for {symbol}: {e}")
            return None
    
    async def _scan_gaps(self, symbol: str, asset_type: AssetType, data: MarketData,
                        historical_data: pd.DataFrame, scan_type: ScanType) -> Optional[ScanResult]:
        """Scan for gap up/down patterns"""
        
        try:
            if len(historical_data) < 2:
                return None
            
            # Get previous close and current open (approximated)
            prev_close = historical_data['close'].iloc[-2]
            current_price = data.price
            
            # Calculate gap percentage
            gap_pct = (current_price - prev_close) / prev_close * 100
            
            # Check gap conditions
            if scan_type == ScanType.GAP_UP and gap_pct < 3.0:  # 3% minimum gap up
                return None
            elif scan_type == ScanType.GAP_DOWN and gap_pct > -3.0:  # 3% minimum gap down
                return None
            
            # Calculate score
            gap_magnitude = abs(gap_pct)
            score = min(1.0, gap_magnitude / 10.0)  # Normalize to 10% max
            
            # Volume confirmation
            volumes = historical_data['volume'].values
            avg_volume = np.mean(volumes[-20:])
            volume_spike = data.volume / avg_volume if avg_volume > 0 else 1
            
            if volume_spike > 1.5:
                score *= 1.2  # Boost score for volume confirmation
            
            score = min(1.0, score)
            
            if score < 0.5:
                return None
            
            # Determine alert level
            if gap_magnitude > 8:
                alert_level = AlertLevel.CRITICAL
            elif gap_magnitude > 5:
                alert_level = AlertLevel.HIGH
            else:
                alert_level = AlertLevel.MEDIUM
            
            direction = "up" if gap_pct > 0 else "down"
            
            return ScanResult(
                symbol=symbol,
                asset_type=asset_type,
                scan_type=scan_type,
                alert_level=alert_level,
                score=score,
                confidence=0.7,
                timestamp=datetime.now(),
                current_price=current_price,
                price_change=data.change,
                price_change_pct=data.change_percent,
                volume=data.volume,
                volume_change_pct=(volume_spike - 1) * 100,
                description=f"Gap {direction}: {gap_pct:.1f}% from previous close",
                reasoning=f"Price gapped {direction} from {format_currency(prev_close)} to {format_currency(current_price)}"
            )
            
        except Exception as e:
            logger.warning(f"Gap scan failed for {symbol}: {e}")
            return None
    
    async def _scan_reversal(self, symbol: str, asset_type: AssetType, data: MarketData,
                           historical_data: pd.DataFrame, regime: RegimeType, 
                           regime_confidence: float) -> Optional[ScanResult]:
        """Scan for reversal patterns"""
        
        try:
            prices = historical_data['close'].values
            
            if len(prices) < 20:
                return None
            
            # Calculate technical indicators for reversal signals
            rsi = self.technical_indicators.calculate_rsi(prices)[-1] if len(prices) > 14 else 50
            
            # Check for oversold/overbought conditions
            oversold_reversal = rsi < 30 and data.change_percent > 1.0  # Oversold bounce
            overbought_reversal = rsi > 70 and data.change_percent < -1.0  # Overbought decline
            
            if not (oversold_reversal or overbought_reversal):
                return None
            
            # Calculate Bollinger Bands for additional confirmation
            bb_upper, bb_middle, bb_lower = self.technical_indicators.calculate_bollinger_bands(prices)
            current_price = data.price
            
            # Check if price is reversing from extreme levels
            bb_reversal = False
            if oversold_reversal and current_price < bb_lower[-1]:
                bb_reversal = True
            elif overbought_reversal and current_price > bb_upper[-1]:
                bb_reversal = True
            
            # Calculate score
            rsi_score = abs(rsi - 50) / 50  # Distance from neutral
            price_change_score = min(1.0, abs(data.change_percent) / 5.0)
            bb_score = 0.5 if bb_reversal else 0.2
            
            score = (rsi_score * 0.4 + price_change_score * 0.4 + bb_score * 0.2)
            score = max(0.0, min(1.0, score))
            
            if score < 0.5:
                return None
            
            # Determine alert level and direction
            if oversold_reversal:
                alert_level = AlertLevel.HIGH if rsi < 25 else AlertLevel.MEDIUM
                reversal_type = "Oversold Bounce"
            else:
                alert_level = AlertLevel.HIGH if rsi > 75 else AlertLevel.MEDIUM
                reversal_type = "Overbought Decline"
            
            return ScanResult(
                symbol=symbol,
                asset_type=asset_type,
                scan_type=ScanType.REVERSAL,
                alert_level=alert_level,
                score=score,
                confidence=regime_confidence,
                timestamp=datetime.now(),
                current_price=current_price,
                price_change=data.change,
                price_change_pct=data.change_percent,
                volume=data.volume,
                rsi=rsi,
                current_regime=regime,
                regime_confidence=regime_confidence,
                description=f"{reversal_type} - RSI: {rsi:.1f}",
                reasoning=f"RSI at extreme level ({rsi:.1f}) with price reversal signal"
            )
            
        except Exception as e:
            logger.warning(f"Reversal scan failed for {symbol}: {e}")
            return None
    
    async def _scan_patterns(self, symbol: str, asset_type: AssetType, data: MarketData,
                           historical_data: pd.DataFrame) -> Optional[ScanResult]:
        """Scan for technical chart patterns"""
        
        try:
            prices = historical_data['close'].values
            highs = historical_data['high'].values
            lows = historical_data['low'].values
            
            if len(prices) < 30:
                return None
            
            # Simple pattern recognition (can be expanded)
            patterns_detected = []
            
            # Double bottom pattern
            if self._detect_double_bottom(lows, prices):
                patterns_detected.append((PatternType.DOUBLE_BOTTOM, 0.7))
            
            # Double top pattern  
            if self._detect_double_top(highs, prices):
                patterns_detected.append((PatternType.DOUBLE_TOP, 0.7))
            
            # Triangle patterns
            triangle_pattern, triangle_confidence = self._detect_triangle_patterns(highs, lows)
            if triangle_pattern:
                patterns_detected.append((triangle_pattern, triangle_confidence))
            
            if not patterns_detected:
                return None
            
            # Get best pattern
            best_pattern, pattern_confidence = max(patterns_detected, key=lambda x: x[1])
            
            # Calculate overall score
            price_confirmation = min(1.0, abs(data.change_percent) / 3.0)
            score = pattern_confidence * 0.7 + price_confirmation * 0.3
            
            if score < 0.5:
                return None
            
            # Determine alert level
            if pattern_confidence > 0.8 and abs(data.change_percent) > 2:
                alert_level = AlertLevel.HIGH
            elif pattern_confidence > 0.6:
                alert_level = AlertLevel.MEDIUM
            else:
                alert_level = AlertLevel.LOW
            
            return ScanResult(
                symbol=symbol,
                asset_type=asset_type,
                scan_type=ScanType.PATTERN_RECOGNITION,
                alert_level=alert_level,
                score=score,
                confidence=pattern_confidence,
                timestamp=datetime.now(),
                current_price=data.price,
                price_change=data.change,
                price_change_pct=data.change_percent,
                volume=data.volume,
                pattern_type=best_pattern,
                pattern_confidence=pattern_confidence,
                description=f"Pattern detected: {best_pattern.value.replace('_', ' ').title()}",
                reasoning=f"Technical pattern with {pattern_confidence:.1%} confidence"
            )
            
        except Exception as e:
            logger.warning(f"Pattern scan failed for {symbol}: {e}")
            return None
    
    async def _scan_volatility_spike(self, symbol: str, asset_type: AssetType, data: MarketData,
                                   historical_data: pd.DataFrame) -> Optional[ScanResult]:
        """Scan for volatility spikes"""
        
        try:
            prices = historical_data['close'].values
            
            if len(prices) < 20:
                return None
            
            # Calculate recent volatility vs historical
            returns = np.diff(np.log(prices))
            recent_vol = np.std(returns[-5:]) * np.sqrt(252)  # Annualized volatility, last 5 periods
            historical_vol = np.std(returns[-20:-5]) * np.sqrt(252)  # Previous 15 periods
            
            if historical_vol == 0:
                return None
            
            vol_ratio = recent_vol / historical_vol
            
            if vol_ratio < 1.5:  # 50% increase in volatility
                return None
            
            # Calculate score
            vol_score = min(1.0, vol_ratio / 3.0)  # Normalize to 3x volatility
            price_move_score = min(1.0, abs(data.change_percent) / 5.0)
            
            score = vol_score * 0.6 + price_move_score * 0.4
            
            if score < 0.5:
                return None
            
            # Determine alert level
            if vol_ratio > 2.5:
                alert_level = AlertLevel.HIGH
            elif vol_ratio > 2.0:
                alert_level = AlertLevel.MEDIUM
            else:
                alert_level = AlertLevel.LOW
            
            return ScanResult(
                symbol=symbol,
                asset_type=asset_type,
                scan_type=ScanType.VOLATILITY_SPIKE,
                alert_level=alert_level,
                score=score,
                confidence=0.6,
                timestamp=datetime.now(),
                current_price=data.price,
                price_change=data.change,
                price_change_pct=data.change_percent,
                volume=data.volume,
                description=f"Volatility spike: {vol_ratio:.1f}x normal",
                reasoning=f"Recent volatility {recent_vol:.1%} vs historical {historical_vol:.1%}"
            )
            
        except Exception as e:
            logger.warning(f"Volatility spike scan failed for {symbol}: {e}")
            return None
    
    def _detect_double_bottom(self, lows: np.ndarray, prices: np.ndarray) -> bool:
        """Detect double bottom pattern"""
        
        try:
            if len(lows) < 20:
                return False
            
            # Find recent lows
            recent_lows = lows[-20:]
            min_indices = []
            
            for i in range(2, len(recent_lows) - 2):
                if (recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i-2] and
                    recent_lows[i] < recent_lows[i+1] and recent_lows[i] < recent_lows[i+2]):
                    min_indices.append(i)
            
            if len(min_indices) < 2:
                return False
            
            # Check if two lows are similar
            last_two_lows = recent_lows[min_indices[-2:]]
            if abs(last_two_lows[1] - last_two_lows[0]) / last_two_lows[0] < 0.02:  # Within 2%
                # Check if current price is above both lows
                current_price = prices[-1]
                if current_price > max(last_two_lows) * 1.01:  # 1% above lows
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _detect_double_top(self, highs: np.ndarray, prices: np.ndarray) -> bool:
        """Detect double top pattern"""
        
        try:
            if len(highs) < 20:
                return False
            
            # Find recent highs
            recent_highs = highs[-20:]
            max_indices = []
            
            for i in range(2, len(recent_highs) - 2):
                if (recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i-2] and
                    recent_highs[i] > recent_highs[i+1] and recent_highs[i] > recent_highs[i+2]):
                    max_indices.append(i)
            
            if len(max_indices) < 2:
                return False
            
            # Check if two highs are similar
            last_two_highs = recent_highs[max_indices[-2:]]
            if abs(last_two_highs[1] - last_two_highs[0]) / last_two_highs[0] < 0.02:  # Within 2%
                # Check if current price is below both highs
                current_price = prices[-1]
                if current_price < min(last_two_highs) * 0.99:  # 1% below highs
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _detect_triangle_patterns(self, highs: np.ndarray, lows: np.ndarray) -> Tuple[Optional[PatternType], float]:
        """Detect triangle patterns"""
        
        try:
            if len(highs) < 15:
                return None, 0.0
            
            recent_highs = highs[-15:]
            recent_lows = lows[-15:]
            
            # Calculate trend lines
            x = np.arange(len(recent_highs))
            
            # High trend line
            high_slope, high_intercept, high_r_value, _, _ = stats.linregress(x, recent_highs)
            
            # Low trend line  
            low_slope, low_intercept, low_r_value, _, _ = stats.linregress(x, recent_lows)
            
            # Check pattern type based on slopes
            if abs(high_slope) < 0.01 and low_slope > 0.01:  # Flat highs, rising lows
                if high_r_value**2 > 0.5 and low_r_value**2 > 0.5:
                    return PatternType.ASCENDING_TRIANGLE, min(high_r_value**2, low_r_value**2)
            
            elif high_slope < -0.01 and abs(low_slope) < 0.01:  # Falling highs, flat lows
                if high_r_value**2 > 0.5 and low_r_value**2 > 0.5:
                    return PatternType.DESCENDING_TRIANGLE, min(high_r_value**2, low_r_value**2)
            
            elif high_slope < -0.01 and low_slope > 0.01:  # Converging lines
                if high_r_value**2 > 0.4 and low_r_value**2 > 0.4:
                    return PatternType.SYMMETRICAL_TRIANGLE, min(high_r_value**2, low_r_value**2)
            
            return None, 0.0
            
        except Exception:
            return None, 0.0
    
    async def _store_scan_results(self, results: List[ScanResult]):
        """Store scan results in database"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for result in results:
                    conn.execute('''
                        INSERT OR REPLACE INTO scan_results (
                            scan_id, symbol, asset_type, scan_type, alert_level,
                            score, confidence, timestamp, current_price, price_change,
                            price_change_pct, volume, volume_change_pct, rsi, macd,
                            current_regime, regime_confidence, pattern_type,
                            pattern_confidence, description, reasoning, target_price,
                            stop_loss, risk_reward_ratio, data_quality
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        result.scan_id,
                        result.symbol,
                        result.asset_type.value,
                        result.scan_type.value,
                        result.alert_level.value,
                        result.score,
                        result.confidence,
                        result.timestamp.isoformat(),
                        result.current_price,
                        result.price_change,
                        result.price_change_pct,
                        result.volume,
                        result.volume_change_pct,
                        result.rsi,
                        result.macd,
                        result.current_regime.value if result.current_regime else None,
                        result.regime_confidence,
                        result.pattern_type.value if result.pattern_type else None,
                        result.pattern_confidence,
                        result.description,
                        result.reasoning,
                        result.target_price,
                        result.stop_loss,
                        result.risk_reward_ratio,
                        result.data_quality
                    ))
                
        except Exception as e:
            logger.warning(f"Failed to store scan results: {e}")
    
    async def get_scan_history(self, symbol: str = None, scan_type: ScanType = None,
                              timeframe: str = "1d", limit: int = 100) -> List[ScanResult]:
        """Get historical scan results"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM scan_results WHERE 1=1"
                params = []
                
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                
                if scan_type:
                    query += " AND scan_type = ?"
                    params.append(scan_type.value)
                
                # Time filter
                if timeframe == "1h":
                    query += " AND timestamp > datetime('now', '-1 hours')"
                elif timeframe == "1d":
                    query += " AND timestamp > datetime('now', '-1 days')"
                elif timeframe == "1w":
                    query += " AND timestamp > datetime('now', '-7 days')"
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                results = []
                
                for row in cursor.fetchall():
                    # Reconstruct ScanResult object (simplified)
                    result = ScanResult(
                        scan_id=row[1],
                        symbol=row[2],
                        asset_type=AssetType(row[3]),
                        scan_type=ScanType(row[4]),
                        alert_level=AlertLevel(row[5]),
                        score=row[6],
                        confidence=row[7],
                        timestamp=datetime.fromisoformat(row[8]),
                        current_price=row[9],
                        price_change=row[10],
                        price_change_pct=row[11],
                        volume=row[12],
                        description=row[19] or "",
                        reasoning=row[20] or ""
                    )
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get scan history: {e}")
            return []
    
    def get_scanner_stats(self) -> Dict[str, Any]:
        """Get scanner statistics"""
        
        return {
            'total_scans': self.stats.total_scans,
            'successful_scans': self.stats.successful_scans,
            'failed_scans': self.stats.failed_scans,
            'success_rate': (self.stats.successful_scans / max(1, self.stats.total_scans)) * 100,
            'total_symbols_scanned': self.stats.total_symbols_scanned,
            'total_opportunities_found': self.stats.total_opportunities_found,
            'high_priority_alerts': self.stats.high_priority_alerts,
            'average_scan_time': self.stats.average_scan_time,
            'last_scan_time': self.stats.last_scan_time.isoformat() if self.stats.last_scan_time else None,
            'active_scans': len(self.active_scans),
            'configured_scans': len(self.scan_configurations)
        }

# ==================== CONVENIENCE FUNCTIONS ====================

async def quick_market_scan(scan_types: List[ScanType] = None, 
                           asset_types: List[AssetType] = None) -> List[ScanResult]:
    """Convenience function for quick market scan"""
    
    scanner = AdvancedMarketScanner()
    
    criteria = ScanCriteria(
        scan_types=scan_types or [ScanType.BREAKOUT, ScanType.MOMENTUM],
        asset_types=asset_types or [AssetType.STOCK, AssetType.CRYPTO],
        max_results=20
    )
    
    return await scanner.run_scan(criteria=criteria)

async def scan_for_breakouts(min_volume: float = 100000) -> List[ScanResult]:
    """Scan specifically for breakout opportunities"""
    
    scanner = AdvancedMarketScanner()
    
    criteria = ScanCriteria(
        scan_types=[ScanType.BREAKOUT],
        min_volume=min_volume,
        min_score=0.7,
        volume_spike_threshold=1.5
    )
    
    return await scanner.run_scan(criteria=criteria)

async def scan_high_momentum_stocks() -> List[ScanResult]:
    """Scan for high momentum stock opportunities"""
    
    scanner = AdvancedMarketScanner()
    
    criteria = ScanCriteria(
        asset_types=[AssetType.STOCK],
        scan_types=[ScanType.MOMENTUM],
        rsi_min=60,
        rsi_max=80,
        price_change_min=0.03,  # 3% minimum price change
        min_score=0.6
    )
    
    return await scanner.run_scan(criteria=criteria)

def create_custom_scanner(name: str, scan_types: List[ScanType], 
                         filters: Dict[str, Any]) -> ScanConfiguration:
    """Create custom scanner configuration"""
    
    criteria = ScanCriteria(
        scan_types=scan_types,
        **filters
    )
    
    return ScanConfiguration(
        name=name,
        criteria=criteria,
        enabled=True
    )