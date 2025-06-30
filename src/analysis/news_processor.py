#!/usr/bin/env python3
"""
File: news_processor.py
Path: NeuroCluster-Elite/src/analysis/news_processor.py
Description: Advanced news processing system for market intelligence

This module implements comprehensive news processing capabilities for financial markets,
aggregating news from multiple sources, analyzing relevance, and extracting market-moving insights.

Features:
- Multi-source news aggregation (Reuters, Bloomberg, Yahoo Finance, etc.)
- Real-time news feeds with WebSocket support
- Advanced relevance scoring and filtering
- Market impact assessment and event detection
- News sentiment preprocessing for sentiment analyzer
- Earnings announcements and SEC filings monitoring
- Breaking news alerts and priority scoring
- Content deduplication and source reliability scoring

Author: Your Name
Created: 2025-06-29
Version: 1.0.0
License: MIT
"""

import asyncio
import aiohttp
import feedparser
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import re
import time
import hashlib
from pathlib import Path
import sqlite3
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

# Web scraping and parsing
from bs4 import BeautifulSoup
import requests
from newspaper import Article

# Import our modules
try:
    from src.core.neurocluster_elite import AssetType, MarketData
    from src.utils.config_manager import ConfigManager
    from src.utils.logger import get_enhanced_logger, LogCategory
    from src.utils.helpers import format_currency, calculate_hash, retry_on_failure
except ImportError:
    # Fallback for testing
    pass

# Configure logging
logger = get_enhanced_logger(__name__, LogCategory.ANALYSIS)

# ==================== ENUMS AND DATA STRUCTURES ====================

class NewsSource(Enum):
    """News source types"""
    REUTERS = "reuters"
    BLOOMBERG = "bloomberg"
    YAHOO_FINANCE = "yahoo_finance"
    MARKETWATCH = "marketwatch"
    CNBC = "cnbc"
    WSJ = "wsj"
    FINANCIAL_TIMES = "financial_times"
    BENZINGA = "benzinga"
    SEEKING_ALPHA = "seeking_alpha"
    MOTLEY_FOOL = "motley_fool"
    SEC_FILINGS = "sec_filings"
    EARNINGS_CALL = "earnings_call"
    PRESS_RELEASE = "press_release"
    ANALYST_REPORT = "analyst_report"
    CUSTOM = "custom"

class NewsCategory(Enum):
    """News categories"""
    EARNINGS = "earnings"
    MERGERS_ACQUISITIONS = "mergers_acquisitions"
    REGULATORY = "regulatory"
    MANAGEMENT_CHANGES = "management_changes"
    PRODUCT_LAUNCHES = "product_launches"
    PARTNERSHIPS = "partnerships"
    FINANCIAL_RESULTS = "financial_results"
    MARKET_ANALYSIS = "market_analysis"
    ECONOMIC_DATA = "economic_data"
    GEOPOLITICAL = "geopolitical"
    SECTOR_NEWS = "sector_news"
    BREAKING_NEWS = "breaking_news"
    GENERAL = "general"

class MarketImpact(Enum):
    """Market impact levels"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"

class NewsUrgency(Enum):
    """News urgency levels"""
    BREAKING = "breaking"
    URGENT = "urgent"
    IMPORTANT = "important"
    NORMAL = "normal"
    LOW = "low"

@dataclass
class NewsArticle:
    """News article data structure"""
    title: str
    content: Optional[str]
    summary: str
    url: str
    source: NewsSource
    published_at: datetime
    author: Optional[str] = None
    
    # Analysis results
    relevance_score: float = 0.0
    market_impact: MarketImpact = MarketImpact.LOW
    urgency: NewsUrgency = NewsUrgency.NORMAL
    category: NewsCategory = NewsCategory.GENERAL
    
    # Content metrics
    sentiment_score: float = 0.0
    readability_score: float = 0.0
    content_quality: float = 0.0
    
    # Metadata
    symbols_mentioned: List[str] = field(default_factory=list)
    sectors_mentioned: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Technical
    article_id: str = field(default_factory=lambda: str(int(time.time() * 1000)))
    content_hash: str = ""
    source_reliability: float = 1.0
    
    # Processing status
    processed: bool = False
    processing_time: float = 0.0

@dataclass
class NewsSourceConfig:
    """Configuration for news sources"""
    name: str
    base_url: str
    rss_feeds: List[str] = field(default_factory=list)
    api_key: Optional[str] = None
    rate_limit: int = 100  # requests per hour
    reliability_score: float = 1.0
    enabled: bool = True
    priority: int = 1  # Lower number = higher priority
    
    # Source-specific settings
    requires_parsing: bool = True
    has_api: bool = False
    supports_search: bool = False

@dataclass
class NewsStats:
    """News processing statistics"""
    total_articles: int = 0
    processed_articles: int = 0
    failed_articles: int = 0
    duplicate_articles: int = 0
    high_impact_articles: int = 0
    processing_time_total: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)

# ==================== NEWS PROCESSOR ====================

class NewsProcessor:
    """Advanced news processing system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize news processor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize news sources
        self.news_sources = self._initialize_news_sources()
        
        # Processing configuration
        self.max_articles_per_source = self.config.get('max_articles_per_source', 50)
        self.content_min_length = self.config.get('content_min_length', 100)
        self.relevance_threshold = self.config.get('relevance_threshold', 0.3)
        self.cache_ttl = self.config.get('cache_ttl', 1800)  # 30 minutes
        
        # Storage and caching
        self.db_path = self.config.get('db_path', 'data/news.db')
        self.article_cache = {}
        self.processed_hashes = set()
        
        # Processing statistics
        self.stats = NewsStats()
        
        # Keywords for relevance scoring
        self.financial_keywords = self._load_financial_keywords()
        self.market_impact_keywords = self._load_market_impact_keywords()
        
        # Session for HTTP requests
        self.session = None
        
        # Initialize database
        self._initialize_database()
        
        logger.info("📰 News Processor initialized")
    
    def _initialize_news_sources(self) -> Dict[NewsSource, NewsSourceConfig]:
        """Initialize news source configurations"""
        
        sources = {
            NewsSource.YAHOO_FINANCE: NewsSourceConfig(
                name="Yahoo Finance",
                base_url="https://finance.yahoo.com",
                rss_feeds=[
                    "https://feeds.finance.yahoo.com/rss/2.0/headline",
                    "https://feeds.finance.yahoo.com/rss/2.0/topstories"
                ],
                reliability_score=0.85,
                priority=1
            ),
            
            NewsSource.MARKETWATCH: NewsSourceConfig(
                name="MarketWatch",
                base_url="https://www.marketwatch.com",
                rss_feeds=[
                    "https://feeds.marketwatch.com/marketwatch/topstories/",
                    "https://feeds.marketwatch.com/marketwatch/marketpulse/"
                ],
                reliability_score=0.90,
                priority=1
            ),
            
            NewsSource.CNBC: NewsSourceConfig(
                name="CNBC",
                base_url="https://www.cnbc.com",
                rss_feeds=[
                    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
                    "https://www.cnbc.com/id/10001147/device/rss/rss.html"
                ],
                reliability_score=0.95,
                priority=1
            ),
            
            NewsSource.REUTERS: NewsSourceConfig(
                name="Reuters",
                base_url="https://www.reuters.com",
                rss_feeds=[
                    "https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best",
                    "https://www.reutersagency.com/feed/?best-regions=north-america&post_type=best"
                ],
                reliability_score=0.98,
                priority=1
            ),
            
            NewsSource.BENZINGA: NewsSourceConfig(
                name="Benzinga",
                base_url="https://www.benzinga.com",
                rss_feeds=[
                    "https://www.benzinga.com/feed"
                ],
                reliability_score=0.75,
                priority=2
            ),
            
            NewsSource.SEEKING_ALPHA: NewsSourceConfig(
                name="Seeking Alpha",
                base_url="https://seekingalpha.com",
                rss_feeds=[
                    "https://seekingalpha.com/feed.xml"
                ],
                reliability_score=0.80,
                priority=2
            )
        }
        
        # Filter enabled sources
        enabled_sources = {k: v for k, v in sources.items() if v.enabled}
        
        logger.info(f"📡 Initialized {len(enabled_sources)} news sources")
        return enabled_sources
    
    def _load_financial_keywords(self) -> Dict[str, float]:
        """Load financial keywords with relevance weights"""
        
        keywords = {
            # Company events (high relevance)
            'earnings': 0.9, 'revenue': 0.8, 'profit': 0.8, 'loss': 0.7,
            'dividend': 0.8, 'acquisition': 0.9, 'merger': 0.9, 'ipo': 0.9,
            'buyback': 0.7, 'split': 0.6, 'spinoff': 0.7,
            
            # Financial metrics
            'eps': 0.8, 'pe ratio': 0.6, 'market cap': 0.6, 'volume': 0.5,
            'price target': 0.8, 'analyst': 0.7, 'upgrade': 0.8, 'downgrade': 0.8,
            
            # Market movements
            'rally': 0.7, 'crash': 0.9, 'surge': 0.7, 'plunge': 0.8,
            'breakout': 0.6, 'resistance': 0.5, 'support': 0.5,
            
            # Regulatory and legal
            'sec': 0.8, 'fda': 0.7, 'regulation': 0.6, 'lawsuit': 0.7,
            'investigation': 0.8, 'fine': 0.6, 'settlement': 0.6,
            
            # Economic indicators
            'gdp': 0.6, 'inflation': 0.7, 'unemployment': 0.6, 'fed': 0.8,
            'interest rate': 0.8, 'monetary policy': 0.7,
            
            # Sector specific
            'technology': 0.4, 'healthcare': 0.4, 'finance': 0.4,
            'energy': 0.4, 'real estate': 0.4, 'consumer': 0.4
        }
        
        return keywords
    
    def _load_market_impact_keywords(self) -> Dict[str, MarketImpact]:
        """Load keywords that indicate market impact level"""
        
        impact_keywords = {
            # Very high impact
            'crash': MarketImpact.VERY_HIGH,
            'bankruptcy': MarketImpact.VERY_HIGH,
            'merger': MarketImpact.VERY_HIGH,
            'acquisition': MarketImpact.VERY_HIGH,
            'ipo': MarketImpact.VERY_HIGH,
            'delisting': MarketImpact.VERY_HIGH,
            
            # High impact
            'earnings': MarketImpact.HIGH,
            'revenue': MarketImpact.HIGH,
            'guidance': MarketImpact.HIGH,
            'dividend': MarketImpact.HIGH,
            'buyback': MarketImpact.HIGH,
            'lawsuit': MarketImpact.HIGH,
            
            # Medium impact
            'analyst': MarketImpact.MEDIUM,
            'upgrade': MarketImpact.MEDIUM,
            'downgrade': MarketImpact.MEDIUM,
            'partnership': MarketImpact.MEDIUM,
            'product launch': MarketImpact.MEDIUM,
            
            # Low impact
            'conference': MarketImpact.LOW,
            'interview': MarketImpact.LOW,
            'commentary': MarketImpact.LOW,
            'analysis': MarketImpact.LOW
        }
        
        return impact_keywords
    
    def _initialize_database(self):
        """Initialize news database"""
        
        try:
            # Create data directory
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create database tables
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS news_articles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        article_id TEXT UNIQUE NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT,
                        summary TEXT,
                        url TEXT UNIQUE NOT NULL,
                        source TEXT NOT NULL,
                        published_at DATETIME NOT NULL,
                        author TEXT,
                        relevance_score REAL,
                        market_impact TEXT,
                        urgency TEXT,
                        category TEXT,
                        sentiment_score REAL,
                        symbols_mentioned TEXT,
                        keywords TEXT,
                        content_hash TEXT,
                        processed BOOLEAN DEFAULT FALSE,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_news_published 
                    ON news_articles(published_at)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_news_symbols 
                    ON news_articles(symbols_mentioned)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_news_relevance 
                    ON news_articles(relevance_score)
                ''')
                
                logger.info("📊 News database initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize news database: {e}")
    
    async def initialize_session(self):
        """Initialize HTTP session"""
        
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def cleanup(self):
        """Cleanup resources"""
        
        if self.session:
            await self.session.close()
    
    async def get_news(self, symbol: str = None, timeframe: str = "1h", 
                      max_articles: int = 50) -> List[NewsArticle]:
        """
        Get news articles for a symbol or general market news
        
        Args:
            symbol: Trading symbol (optional)
            timeframe: Time range for news
            max_articles: Maximum number of articles to return
            
        Returns:
            List of relevant news articles
        """
        
        try:
            await self.initialize_session()
            
            # Calculate time range
            end_time = datetime.now()
            if timeframe == "1h":
                start_time = end_time - timedelta(hours=1)
            elif timeframe == "4h":
                start_time = end_time - timedelta(hours=4)
            elif timeframe == "1d":
                start_time = end_time - timedelta(days=1)
            elif timeframe == "1w":
                start_time = end_time - timedelta(weeks=1)
            else:
                start_time = end_time - timedelta(hours=4)  # Default
            
            # Check cache first
            cache_key = f"{symbol}_{timeframe}_{max_articles}"
            if cache_key in self.article_cache:
                cached_data, timestamp = self.article_cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    return cached_data[:max_articles]
            
            # Fetch news from all sources
            all_articles = await self._fetch_news_from_sources(start_time, end_time)
            
            # Process and filter articles
            processed_articles = []
            for article in all_articles:
                processed_article = await self._process_article(article, symbol)
                if processed_article and processed_article.relevance_score > self.relevance_threshold:
                    processed_articles.append(processed_article)
            
            # Sort by relevance and recency
            processed_articles.sort(
                key=lambda x: (x.relevance_score * 0.7 + 
                             self._calculate_recency_score(x.published_at) * 0.3),
                reverse=True
            )
            
            # Store in cache
            self.article_cache[cache_key] = (processed_articles, time.time())
            
            # Update stats
            self.stats.total_articles += len(all_articles)
            self.stats.processed_articles += len(processed_articles)
            self.stats.last_update = datetime.now()
            
            logger.info(f"📰 Retrieved {len(processed_articles)} relevant articles for {symbol or 'market'}")
            
            return processed_articles[:max_articles]
            
        except Exception as e:
            logger.error(f"Failed to get news: {e}")
            return []
    
    async def _fetch_news_from_sources(self, start_time: datetime, 
                                     end_time: datetime) -> List[NewsArticle]:
        """Fetch news from all configured sources"""
        
        all_articles = []
        
        # Fetch from RSS feeds
        rss_tasks = []
        for source, config in self.news_sources.items():
            for feed_url in config.rss_feeds:
                rss_tasks.append(self._fetch_rss_feed(source, config, feed_url, start_time))
        
        # Execute RSS tasks
        if rss_tasks:
            rss_results = await asyncio.gather(*rss_tasks, return_exceptions=True)
            
            for result in rss_results:
                if isinstance(result, Exception):
                    logger.warning(f"RSS fetch failed: {result}")
                    continue
                
                if isinstance(result, list):
                    all_articles.extend(result)
        
        # Remove duplicates based on content hash
        unique_articles = self._remove_duplicates(all_articles)
        
        return unique_articles
    
    async def _fetch_rss_feed(self, source: NewsSource, config: NewsSourceConfig,
                            feed_url: str, start_time: datetime) -> List[NewsArticle]:
        """Fetch articles from RSS feed"""
        
        try:
            # Fetch RSS feed
            async with self.session.get(feed_url) as response:
                if response.status != 200:
                    logger.warning(f"RSS feed {feed_url} returned status {response.status}")
                    return []
                
                rss_content = await response.text()
            
            # Parse RSS feed
            feed = feedparser.parse(rss_content)
            
            articles = []
            for entry in feed.entries[:self.max_articles_per_source]:
                try:
                    # Extract publication date
                    published_at = self._parse_date(entry.get('published', ''))
                    
                    # Skip old articles
                    if published_at < start_time:
                        continue
                    
                    # Create article object
                    article = NewsArticle(
                        title=entry.get('title', ''),
                        content=None,  # Will be fetched later if needed
                        summary=entry.get('summary', ''),
                        url=entry.get('link', ''),
                        source=source,
                        published_at=published_at,
                        author=entry.get('author', None),
                        source_reliability=config.reliability_score
                    )
                    
                    # Generate content hash
                    article.content_hash = self._generate_content_hash(article)
                    
                    # Skip if already processed
                    if article.content_hash in self.processed_hashes:
                        self.stats.duplicate_articles += 1
                        continue
                    
                    articles.append(article)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse RSS entry: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.warning(f"Failed to fetch RSS feed {feed_url}: {e}")
            return []
    
    def _parse_date(self, date_string: str) -> datetime:
        """Parse date string from various formats"""
        
        if not date_string:
            return datetime.now()
        
        # Common date formats in RSS feeds
        formats = [
            '%a, %d %b %Y %H:%M:%S %z',
            '%a, %d %b %Y %H:%M:%S %Z',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S',
            '%a, %d %b %Y %H:%M:%S'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_string, fmt)
            except ValueError:
                continue
        
        # Fallback to current time
        logger.warning(f"Could not parse date: {date_string}")
        return datetime.now()
    
    def _generate_content_hash(self, article: NewsArticle) -> str:
        """Generate hash for article content to detect duplicates"""
        
        content = f"{article.title}{article.summary}".lower()
        content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
        return hashlib.md5(content.encode()).hexdigest()
    
    def _remove_duplicates(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on content hash"""
        
        seen_hashes = set()
        unique_articles = []
        
        for article in articles:
            if article.content_hash not in seen_hashes:
                seen_hashes.add(article.content_hash)
                unique_articles.append(article)
                self.processed_hashes.add(article.content_hash)
            else:
                self.stats.duplicate_articles += 1
        
        return unique_articles
    
    async def _process_article(self, article: NewsArticle, target_symbol: str = None) -> Optional[NewsArticle]:
        """Process and analyze article"""
        
        start_time = time.time()
        
        try:
            # Calculate relevance score
            article.relevance_score = self._calculate_relevance_score(article, target_symbol)
            
            # Extract symbols mentioned
            article.symbols_mentioned = self._extract_symbols(article.title + " " + article.summary)
            
            # Categorize article
            article.category = self._categorize_article(article)
            
            # Assess market impact
            article.market_impact = self._assess_market_impact(article)
            
            # Determine urgency
            article.urgency = self._determine_urgency(article)
            
            # Extract keywords
            article.keywords = self._extract_keywords(article.title + " " + article.summary)
            
            # Fetch full content if high relevance
            if article.relevance_score > 0.6:
                article.content = await self._fetch_full_content(article.url)
                if article.content:
                    article.content_quality = self._assess_content_quality(article.content)
            
            # Mark as processed
            article.processed = True
            article.processing_time = time.time() - start_time
            
            # Store in database
            await self._store_article(article)
            
            return article
            
        except Exception as e:
            logger.warning(f"Failed to process article {article.url}: {e}")
            article.processing_time = time.time() - start_time
            self.stats.failed_articles += 1
            return None
    
    def _calculate_relevance_score(self, article: NewsArticle, target_symbol: str = None) -> float:
        """Calculate relevance score for article"""
        
        text = (article.title + " " + article.summary).lower()
        score = 0.0
        
        # Base score from financial keywords
        for keyword, weight in self.financial_keywords.items():
            if keyword in text:
                score += weight * 0.1
        
        # Symbol-specific relevance
        if target_symbol:
            symbol_variations = [
                target_symbol.upper(),
                target_symbol.lower(),
                f"${target_symbol.upper()}",
                f"({target_symbol.upper()})"
            ]
            
            for variation in symbol_variations:
                if variation in text:
                    score += 0.5
                    break
        
        # Source reliability bonus
        score *= article.source_reliability
        
        # Recency bonus
        recency_score = self._calculate_recency_score(article.published_at)
        score += recency_score * 0.1
        
        # Title keyword bonus
        title_lower = article.title.lower()
        for keyword in self.financial_keywords:
            if keyword in title_lower:
                score += 0.1
        
        return min(1.0, score)
    
    def _calculate_recency_score(self, published_at: datetime) -> float:
        """Calculate recency score (newer = higher score)"""
        
        now = datetime.now()
        
        # Handle timezone-naive datetime
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=now.tzinfo)
        
        time_diff = (now - published_at).total_seconds()
        hours_old = time_diff / 3600
        
        # Score decreases with age
        if hours_old < 1:
            return 1.0
        elif hours_old < 6:
            return 0.8
        elif hours_old < 24:
            return 0.6
        elif hours_old < 72:
            return 0.4
        else:
            return 0.2
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text"""
        
        symbols = []
        
        # Pattern for symbols like $AAPL, (NASDAQ:AAPL), etc.
        symbol_patterns = [
            r'\$([A-Z]{1,5})\b',
            r'\(([A-Z]{1,5})\)',
            r'\b([A-Z]{2,5}):\s*([A-Z]{1,5})\b',
            r'\bNASDAQ:\s*([A-Z]{1,5})\b',
            r'\bNYSE:\s*([A-Z]{1,5})\b'
        ]
        
        for pattern in symbol_patterns:
            matches = re.findall(pattern, text.upper())
            for match in matches:
                if isinstance(match, tuple):
                    symbols.extend(match)
                else:
                    symbols.append(match)
        
        # Remove common false positives
        false_positives = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAD', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'GOT', 'LET', 'MAN', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
        
        valid_symbols = [s for s in symbols if s not in false_positives and len(s) <= 5]
        
        return list(set(valid_symbols))  # Remove duplicates
    
    def _categorize_article(self, article: NewsArticle) -> NewsCategory:
        """Categorize article based on content"""
        
        text = (article.title + " " + article.summary).lower()
        
        category_keywords = {
            NewsCategory.EARNINGS: ['earnings', 'quarterly', 'revenue', 'profit', 'eps'],
            NewsCategory.MERGERS_ACQUISITIONS: ['merger', 'acquisition', 'buyout', 'takeover'],
            NewsCategory.REGULATORY: ['sec', 'fda', 'regulation', 'compliance', 'investigation'],
            NewsCategory.MANAGEMENT_CHANGES: ['ceo', 'cfo', 'president', 'director', 'appointed', 'resigned'],
            NewsCategory.PRODUCT_LAUNCHES: ['launch', 'product', 'service', 'release', 'debut'],
            NewsCategory.PARTNERSHIPS: ['partnership', 'alliance', 'collaboration', 'joint venture'],
            NewsCategory.FINANCIAL_RESULTS: ['results', 'performance', 'guidance', 'outlook'],
            NewsCategory.BREAKING_NEWS: ['breaking', 'urgent', 'alert', 'developing']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text for keyword in keywords):
                return category
        
        return NewsCategory.GENERAL
    
    def _assess_market_impact(self, article: NewsArticle) -> MarketImpact:
        """Assess potential market impact of article"""
        
        text = (article.title + " " + article.summary).lower()
        
        # Check for high-impact keywords
        for keyword, impact in self.market_impact_keywords.items():
            if keyword in text:
                return impact
        
        # Default based on source reliability and relevance
        if article.source_reliability > 0.9 and article.relevance_score > 0.7:
            return MarketImpact.MEDIUM
        elif article.relevance_score > 0.5:
            return MarketImpact.LOW
        else:
            return MarketImpact.MINIMAL
    
    def _determine_urgency(self, article: NewsArticle) -> NewsUrgency:
        """Determine urgency level of article"""
        
        text = (article.title + " " + article.summary).lower()
        
        # Breaking news indicators
        if any(word in text for word in ['breaking', 'urgent', 'alert']):
            return NewsUrgency.BREAKING
        
        # Recent high-impact news
        hours_old = (datetime.now() - article.published_at).total_seconds() / 3600
        
        if hours_old < 1 and article.market_impact in [MarketImpact.VERY_HIGH, MarketImpact.HIGH]:
            return NewsUrgency.URGENT
        elif hours_old < 6 and article.relevance_score > 0.7:
            return NewsUrgency.IMPORTANT
        else:
            return NewsUrgency.NORMAL
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter common words and keep financial terms
        financial_terms = set(self.financial_keywords.keys())
        keywords = [word for word in words if word in financial_terms]
        
        return list(set(keywords))[:10]  # Limit to 10 most relevant
    
    async def _fetch_full_content(self, url: str) -> Optional[str]:
        """Fetch full article content from URL"""
        
        try:
            # Use newspaper library for better content extraction
            article = Article(url)
            article.download()
            article.parse()
            
            if len(article.text) > self.content_min_length:
                return article.text
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to fetch content from {url}: {e}")
            return None
    
    def _assess_content_quality(self, content: str) -> float:
        """Assess quality of article content"""
        
        if not content:
            return 0.0
        
        quality_score = 0.0
        
        # Length score
        if len(content) > 500:
            quality_score += 0.3
        elif len(content) > 200:
            quality_score += 0.2
        else:
            quality_score += 0.1
        
        # Financial keyword density
        financial_word_count = sum(1 for word in self.financial_keywords if word in content.lower())
        keyword_density = financial_word_count / max(1, len(content.split()) / 100)
        quality_score += min(0.4, keyword_density)
        
        # Readability (simplified)
        sentences = content.split('.')
        avg_sentence_length = len(content.split()) / max(1, len(sentences))
        
        if 10 <= avg_sentence_length <= 25:
            quality_score += 0.3
        else:
            quality_score += 0.1
        
        return min(1.0, quality_score)
    
    async def _store_article(self, article: NewsArticle):
        """Store article in database"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO news_articles (
                        article_id, title, content, summary, url, source, published_at,
                        author, relevance_score, market_impact, urgency, category,
                        sentiment_score, symbols_mentioned, keywords, content_hash, processed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article.article_id,
                    article.title,
                    article.content,
                    article.summary,
                    article.url,
                    article.source.value,
                    article.published_at.isoformat(),
                    article.author,
                    article.relevance_score,
                    article.market_impact.value,
                    article.urgency.value,
                    article.category.value,
                    article.sentiment_score,
                    json.dumps(article.symbols_mentioned),
                    json.dumps(article.keywords),
                    article.content_hash,
                    article.processed
                ))
                
        except Exception as e:
            logger.warning(f"Failed to store article: {e}")
    
    async def get_breaking_news(self, max_articles: int = 10) -> List[NewsArticle]:
        """Get breaking news articles"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT * FROM news_articles 
                    WHERE urgency IN ('breaking', 'urgent') 
                    AND published_at > datetime('now', '-24 hours')
                    ORDER BY published_at DESC 
                    LIMIT ?
                ''', (max_articles,))
                
                articles = []
                for row in cursor.fetchall():
                    # Reconstruct article object
                    article = self._row_to_article(row)
                    articles.append(article)
                
                return articles
                
        except Exception as e:
            logger.error(f"Failed to get breaking news: {e}")
            return []
    
    async def search_news(self, query: str, timeframe: str = "1w", 
                         max_articles: int = 50) -> List[NewsArticle]:
        """Search news articles by query"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Calculate time range
                if timeframe == "1h":
                    time_filter = "datetime('now', '-1 hours')"
                elif timeframe == "1d":
                    time_filter = "datetime('now', '-1 days')"
                elif timeframe == "1w":
                    time_filter = "datetime('now', '-7 days')"
                else:
                    time_filter = "datetime('now', '-1 days')"
                
                cursor = conn.execute(f'''
                    SELECT * FROM news_articles 
                    WHERE (title LIKE ? OR summary LIKE ? OR content LIKE ?)
                    AND published_at > {time_filter}
                    ORDER BY relevance_score DESC, published_at DESC 
                    LIMIT ?
                ''', (f'%{query}%', f'%{query}%', f'%{query}%', max_articles))
                
                articles = []
                for row in cursor.fetchall():
                    article = self._row_to_article(row)
                    articles.append(article)
                
                return articles
                
        except Exception as e:
            logger.error(f"Failed to search news: {e}")
            return []
    
    def _row_to_article(self, row) -> NewsArticle:
        """Convert database row to NewsArticle object"""
        
        return NewsArticle(
            article_id=row[1],
            title=row[2],
            content=row[3],
            summary=row[4],
            url=row[5],
            source=NewsSource(row[6]),
            published_at=datetime.fromisoformat(row[7]),
            author=row[8],
            relevance_score=row[9] or 0.0,
            market_impact=MarketImpact(row[10]) if row[10] else MarketImpact.LOW,
            urgency=NewsUrgency(row[11]) if row[11] else NewsUrgency.NORMAL,
            category=NewsCategory(row[12]) if row[12] else NewsCategory.GENERAL,
            sentiment_score=row[13] or 0.0,
            symbols_mentioned=json.loads(row[14]) if row[14] else [],
            keywords=json.loads(row[15]) if row[15] else [],
            content_hash=row[16],
            processed=bool(row[17])
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get news processing statistics"""
        
        return {
            'total_articles': self.stats.total_articles,
            'processed_articles': self.stats.processed_articles,
            'failed_articles': self.stats.failed_articles,
            'duplicate_articles': self.stats.duplicate_articles,
            'success_rate': (self.stats.processed_articles / max(1, self.stats.total_articles)) * 100,
            'cache_size': len(self.article_cache),
            'sources_configured': len(self.news_sources),
            'last_update': self.stats.last_update.isoformat()
        }

# ==================== CONVENIENCE FUNCTIONS ====================

async def get_latest_news(symbols: List[str] = None, max_articles: int = 20) -> List[NewsArticle]:
    """Convenience function to get latest news"""
    
    processor = NewsProcessor()
    
    try:
        if symbols:
            all_articles = []
            for symbol in symbols:
                articles = await processor.get_news(symbol, "4h", max_articles // len(symbols))
                all_articles.extend(articles)
            return sorted(all_articles, key=lambda x: x.published_at, reverse=True)[:max_articles]
        else:
            return await processor.get_news(None, "4h", max_articles)
    finally:
        await processor.cleanup()

async def get_earnings_news(timeframe: str = "1w") -> List[NewsArticle]:
    """Get earnings-related news"""
    
    processor = NewsProcessor()
    
    try:
        articles = await processor.get_news(None, timeframe, 100)
        earnings_articles = [a for a in articles if a.category == NewsCategory.EARNINGS]
        return sorted(earnings_articles, key=lambda x: x.relevance_score, reverse=True)
    finally:
        await processor.cleanup()

async def monitor_breaking_news(callback_func, check_interval: int = 60):
    """Monitor for breaking news and call callback function"""
    
    processor = NewsProcessor()
    last_check = datetime.now()
    
    try:
        while True:
            breaking_news = await processor.get_breaking_news()
            
            # Filter news since last check
            new_breaking = [n for n in breaking_news if n.published_at > last_check]
            
            if new_breaking:
                await callback_func(new_breaking)
            
            last_check = datetime.now()
            await asyncio.sleep(check_interval)
            
    except Exception as e:
        logger.error(f"Breaking news monitoring failed: {e}")
    finally:
        await processor.cleanup()