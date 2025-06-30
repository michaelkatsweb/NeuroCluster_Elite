#!/usr/bin/env python3
"""
File: social_sentiment.py
Path: NeuroCluster-Elite/src/analysis/social_sentiment.py
Description: Social media sentiment analysis system for market intelligence

This module implements comprehensive social media sentiment analysis capabilities,
processing data from Reddit, Twitter, Discord, Telegram, and other platforms to
gauge market sentiment and detect emerging trends.

Features:
- Multi-platform social media monitoring (Reddit, Twitter, Discord, Telegram)
- Real-time sentiment tracking with engagement weighting
- Influencer and whale tracker monitoring
- Viral content and trend detection
- Social volume and momentum analysis
- Crypto-specific social sentiment metrics
- Community sentiment scoring and consensus tracking
- Integration with main sentiment analyzer

Author: Your Name
Created: 2025-06-29
Version: 1.0.0
License: MIT
"""

import asyncio
import aiohttp
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

# Social media APIs
import praw  # Reddit API
import tweepy  # Twitter API
import requests
from urllib.parse import quote_plus

# NLP and sentiment
from textblob import TextBlob
import vaderSentiment.vaderSentiment as vader

# Import our modules
try:
    from src.core.neurocluster_elite import AssetType, MarketData
    from src.utils.config_manager import ConfigManager
    from src.utils.logger import get_enhanced_logger, LogCategory
    from src.utils.helpers import format_percentage, calculate_hash
except ImportError:
    # Fallback for testing
    pass

# Configure logging
logger = get_enhanced_logger(__name__, LogCategory.ANALYSIS)

# ==================== ENUMS AND DATA STRUCTURES ====================

class SocialPlatform(Enum):
    """Social media platforms"""
    REDDIT = "reddit"
    TWITTER = "twitter"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    STOCKTWITS = "stocktwits"
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    LINKEDIN = "linkedin"

class ContentType(Enum):
    """Types of social content"""
    POST = "post"
    COMMENT = "comment"
    REPLY = "reply"
    RETWEET = "retweet"
    SHARE = "share"
    THREAD = "thread"
    VIDEO = "video"
    IMAGE = "image"
    POLL = "poll"
    STORY = "story"

class EngagementType(Enum):
    """Types of engagement"""
    LIKE = "like"
    UPVOTE = "upvote"
    DOWNVOTE = "downvote"
    SHARE = "share"
    COMMENT = "comment"
    RETWEET = "retweet"
    FOLLOW = "follow"
    MENTION = "mention"
    TAG = "tag"

class InfluencerTier(Enum):
    """Influencer tiers based on following"""
    MEGA = "mega"        # >1M followers
    MACRO = "macro"      # 100K-1M followers
    MICRO = "micro"      # 10K-100K followers
    NANO = "nano"        # 1K-10K followers
    REGULAR = "regular"  # <1K followers

@dataclass
class SocialPost:
    """Social media post data structure"""
    post_id: str
    platform: SocialPlatform
    content: str
    author: str
    timestamp: datetime
    url: Optional[str] = None
    
    # Engagement metrics
    likes: int = 0
    shares: int = 0
    comments: int = 0
    views: int = 0
    engagement_score: float = 0.0
    
    # Content analysis
    sentiment_score: float = 0.0
    relevance_score: float = 0.0
    content_type: ContentType = ContentType.POST
    
    # Author information
    author_followers: int = 0
    author_tier: InfluencerTier = InfluencerTier.REGULAR
    author_verified: bool = False
    author_influence_score: float = 0.0
    
    # Symbol and topic tracking
    symbols_mentioned: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Processing metadata
    content_hash: str = ""
    processed: bool = False
    processing_time: float = 0.0

@dataclass
class SocialSentimentMetrics:
    """Social sentiment metrics for a symbol"""
    symbol: str
    platform: SocialPlatform
    timeframe: str
    timestamp: datetime
    
    # Volume metrics
    total_posts: int = 0
    total_engagement: int = 0
    unique_authors: int = 0
    
    # Sentiment metrics
    positive_posts: int = 0
    negative_posts: int = 0
    neutral_posts: int = 0
    average_sentiment: float = 0.0
    sentiment_momentum: float = 0.0
    
    # Influence metrics
    influencer_sentiment: float = 0.0
    whale_sentiment: float = 0.0
    community_consensus: float = 0.0
    
    # Engagement metrics
    viral_threshold_reached: bool = False
    trending_score: float = 0.0
    engagement_velocity: float = 0.0
    
    # Quality metrics
    data_quality: float = 1.0
    confidence: float = 0.0

@dataclass
class TrendingTopic:
    """Trending topic analysis"""
    topic: str
    platform: SocialPlatform
    mentions: int
    engagement: int
    sentiment: float
    momentum: float
    timeframe: str
    related_symbols: List[str] = field(default_factory=list)
    top_posts: List[SocialPost] = field(default_factory=list)

@dataclass
class InfluencerAnalysis:
    """Influencer sentiment analysis"""
    author: str
    platform: SocialPlatform
    tier: InfluencerTier
    followers: int
    posts_analyzed: int
    average_sentiment: float
    influence_score: float
    recent_posts: List[SocialPost] = field(default_factory=list)

# ==================== SOCIAL SENTIMENT PROCESSOR ====================

class SocialSentimentProcessor:
    """Advanced social media sentiment analysis system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize social sentiment processor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # API configurations
        self.reddit_client = None
        self.twitter_client = None
        
        # Initialize sentiment analyzer
        self.vader_analyzer = vader.SentimentIntensityAnalyzer()
        
        # Platform configurations
        self.platform_configs = self._initialize_platform_configs()
        
        # Processing settings
        self.max_posts_per_platform = self.config.get('max_posts_per_platform', 100)
        self.min_engagement_threshold = self.config.get('min_engagement_threshold', 5)
        self.relevance_threshold = self.config.get('relevance_threshold', 0.3)
        self.cache_ttl = self.config.get('cache_ttl', 900)  # 15 minutes
        
        # Storage and caching
        self.db_path = self.config.get('db_path', 'data/social_sentiment.db')
        self.sentiment_cache = {}
        self.processed_hashes = set()
        
        # Tracking data
        self.influencers = {}
        self.trending_topics = defaultdict(list)
        self.platform_stats = defaultdict(dict)
        
        # Keywords for crypto and finance
        self.crypto_keywords = self._load_crypto_keywords()
        self.finance_keywords = self._load_finance_keywords()
        
        # Initialize database
        self._initialize_database()
        
        # Initialize API clients
        self._initialize_api_clients()
        
        logger.info("📱 Social Sentiment Processor initialized")
    
    def _initialize_platform_configs(self) -> Dict[SocialPlatform, Dict[str, Any]]:
        """Initialize platform-specific configurations"""
        
        configs = {
            SocialPlatform.REDDIT: {
                'enabled': self.config.get('reddit_enabled', True),
                'subreddits': [
                    'wallstreetbets', 'stocks', 'investing', 'SecurityAnalysis',
                    'CryptoCurrency', 'Bitcoin', 'ethereum', 'altcoin',
                    'StockMarket', 'pennystocks', 'ValueInvesting'
                ],
                'sort_methods': ['hot', 'new', 'rising', 'top'],
                'time_filters': ['hour', 'day', 'week'],
                'post_limit': 100,
                'comment_limit': 50
            },
            
            SocialPlatform.TWITTER: {
                'enabled': self.config.get('twitter_enabled', True),
                'search_terms': ['$', 'stocks', 'crypto', 'bitcoin', 'ethereum', 'trading'],
                'result_type': 'recent',
                'lang': 'en',
                'count': 100,
                'include_entities': True
            },
            
            SocialPlatform.STOCKTWITS: {
                'enabled': self.config.get('stocktwits_enabled', True),
                'base_url': 'https://api.stocktwits.com/api/2',
                'endpoints': {
                    'streams': '/streams/symbol/{symbol}.json',
                    'trending': '/trending/symbols.json'
                }
            },
            
            SocialPlatform.DISCORD: {
                'enabled': self.config.get('discord_enabled', False),
                'servers': [],  # Would need specific server IDs
                'channels': []  # Would need specific channel IDs
            },
            
            SocialPlatform.TELEGRAM: {
                'enabled': self.config.get('telegram_enabled', False),
                'channels': []  # Would need specific channel usernames
            }
        }
        
        return configs
    
    def _load_crypto_keywords(self) -> Dict[str, float]:
        """Load cryptocurrency-specific keywords"""
        
        keywords = {
            # Major cryptocurrencies
            'bitcoin': 0.9, 'btc': 0.9, 'ethereum': 0.8, 'eth': 0.8,
            'dogecoin': 0.7, 'doge': 0.7, 'cardano': 0.6, 'ada': 0.6,
            'solana': 0.6, 'sol': 0.6, 'polkadot': 0.5, 'dot': 0.5,
            
            # Crypto terms
            'hodl': 0.6, 'moon': 0.7, 'lambo': 0.6, 'diamond hands': 0.8,
            'paper hands': -0.7, 'dump': -0.8, 'pump': 0.7, 'rekt': -0.9,
            'fud': -0.6, 'fomo': 0.5, 'ath': 0.6, 'dip': -0.3,
            
            # Trading terms
            'long': 0.5, 'short': -0.5, 'buy': 0.6, 'sell': -0.4,
            'accumulate': 0.7, 'distribute': -0.4, 'whale': 0.3,
            
            # Market sentiment
            'bullish': 0.8, 'bearish': -0.8, 'sideways': 0.0,
            'volatile': -0.2, 'stable': 0.3, 'moon shot': 0.9
        }
        
        return keywords
    
    def _load_finance_keywords(self) -> Dict[str, float]:
        """Load finance-specific keywords"""
        
        keywords = {
            # Market terms
            'rally': 0.7, 'crash': -0.9, 'correction': -0.5, 'bounce': 0.6,
            'breakout': 0.7, 'breakdown': -0.7, 'support': 0.3, 'resistance': -0.3,
            
            # Sentiment terms
            'confident': 0.6, 'worried': -0.6, 'optimistic': 0.7, 'pessimistic': -0.7,
            'excited': 0.8, 'concerned': -0.5, 'hopeful': 0.6, 'fearful': -0.8,
            
            # Performance terms
            'gains': 0.7, 'losses': -0.7, 'profit': 0.8, 'loss': -0.6,
            'winning': 0.7, 'losing': -0.6, 'beat': 0.6, 'miss': -0.6,
            
            # Action terms
            'buying': 0.6, 'selling': -0.4, 'holding': 0.2, 'waiting': 0.0,
            'accumulating': 0.7, 'dumping': -0.8, 'loading': 0.6
        }
        
        return keywords
    
    def _initialize_database(self):
        """Initialize social sentiment database"""
        
        try:
            # Create data directory
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create database tables
            with sqlite3.connect(self.db_path) as conn:
                # Social posts table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS social_posts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        post_id TEXT UNIQUE NOT NULL,
                        platform TEXT NOT NULL,
                        content TEXT NOT NULL,
                        author TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        url TEXT,
                        likes INTEGER DEFAULT 0,
                        shares INTEGER DEFAULT 0,
                        comments INTEGER DEFAULT 0,
                        views INTEGER DEFAULT 0,
                        engagement_score REAL DEFAULT 0.0,
                        sentiment_score REAL DEFAULT 0.0,
                        relevance_score REAL DEFAULT 0.0,
                        author_followers INTEGER DEFAULT 0,
                        author_verified BOOLEAN DEFAULT FALSE,
                        symbols_mentioned TEXT,
                        hashtags TEXT,
                        keywords TEXT,
                        content_hash TEXT UNIQUE,
                        processed BOOLEAN DEFAULT FALSE,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Sentiment metrics table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS sentiment_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        platform TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        total_posts INTEGER DEFAULT 0,
                        positive_posts INTEGER DEFAULT 0,
                        negative_posts INTEGER DEFAULT 0,
                        neutral_posts INTEGER DEFAULT 0,
                        average_sentiment REAL DEFAULT 0.0,
                        trending_score REAL DEFAULT 0.0,
                        confidence REAL DEFAULT 0.0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes
                conn.execute('CREATE INDEX IF NOT EXISTS idx_posts_timestamp ON social_posts(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_posts_platform ON social_posts(platform)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_posts_symbols ON social_posts(symbols_mentioned)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_metrics_symbol ON sentiment_metrics(symbol)')
                
                logger.info("📊 Social sentiment database initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize social sentiment database: {e}")
    
    def _initialize_api_clients(self):
        """Initialize social media API clients"""
        
        try:
            # Initialize Reddit client
            reddit_config = self.config.get('reddit', {})
            if reddit_config.get('client_id') and reddit_config.get('client_secret'):
                self.reddit_client = praw.Reddit(
                    client_id=reddit_config['client_id'],
                    client_secret=reddit_config['client_secret'],
                    user_agent=reddit_config.get('user_agent', 'NeuroCluster-Elite/1.0')
                )
                logger.info("🔴 Reddit client initialized")
            
            # Initialize Twitter client
            twitter_config = self.config.get('twitter', {})
            if twitter_config.get('bearer_token'):
                self.twitter_client = tweepy.Client(
                    bearer_token=twitter_config['bearer_token']
                )
                logger.info("🐦 Twitter client initialized")
            
        except Exception as e:
            logger.warning(f"Some API clients failed to initialize: {e}")
    
    async def get_posts(self, symbol: str, timeframe: str = "1h", 
                       platforms: List[SocialPlatform] = None) -> List[SocialPost]:
        """
        Get social media posts for a symbol
        
        Args:
            symbol: Trading symbol
            timeframe: Time range for posts
            platforms: List of platforms to search (default: all enabled)
            
        Returns:
            List of relevant social media posts
        """
        
        if platforms is None:
            platforms = [p for p, config in self.platform_configs.items() if config.get('enabled', False)]
        
        try:
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
            
            # Check cache
            cache_key = f"{symbol}_{timeframe}_{'-'.join(p.value for p in platforms)}"
            if cache_key in self.sentiment_cache:
                cached_data, timestamp = self.sentiment_cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    return cached_data
            
            # Collect posts from all platforms
            all_posts = []
            for platform in platforms:
                posts = await self._get_platform_posts(platform, symbol, start_time, end_time)
                all_posts.extend(posts)
            
            # Process and filter posts
            processed_posts = []
            for post in all_posts:
                processed_post = await self._process_post(post, symbol)
                if processed_post and processed_post.relevance_score > self.relevance_threshold:
                    processed_posts.append(processed_post)
            
            # Sort by engagement and relevance
            processed_posts.sort(
                key=lambda x: (x.engagement_score * 0.6 + x.relevance_score * 0.4),
                reverse=True
            )
            
            # Store in cache
            self.sentiment_cache[cache_key] = (processed_posts, time.time())
            
            logger.info(f"📱 Retrieved {len(processed_posts)} social posts for {symbol}")
            
            return processed_posts
            
        except Exception as e:
            logger.error(f"Failed to get social posts: {e}")
            return []
    
    async def _get_platform_posts(self, platform: SocialPlatform, symbol: str,
                                 start_time: datetime, end_time: datetime) -> List[SocialPost]:
        """Get posts from specific platform"""
        
        if platform == SocialPlatform.REDDIT:
            return await self._get_reddit_posts(symbol, start_time, end_time)
        elif platform == SocialPlatform.TWITTER:
            return await self._get_twitter_posts(symbol, start_time, end_time)
        elif platform == SocialPlatform.STOCKTWITS:
            return await self._get_stocktwits_posts(symbol, start_time, end_time)
        else:
            logger.warning(f"Platform {platform.value} not yet implemented")
            return []
    
    async def _get_reddit_posts(self, symbol: str, start_time: datetime, 
                               end_time: datetime) -> List[SocialPost]:
        """Get posts from Reddit"""
        
        if not self.reddit_client:
            return []
        
        posts = []
        config = self.platform_configs[SocialPlatform.REDDIT]
        
        try:
            # Search in relevant subreddits
            for subreddit_name in config['subreddits']:
                try:
                    subreddit = self.reddit_client.subreddit(subreddit_name)
                    
                    # Search for symbol mentions
                    search_queries = [symbol, f"${symbol}", f"({symbol})"]
                    
                    for query in search_queries:
                        for submission in subreddit.search(query, sort='new', time_filter='day', limit=20):
                            # Check if post is within time range
                            post_time = datetime.fromtimestamp(submission.created_utc)
                            if start_time <= post_time <= end_time:
                                
                                post = SocialPost(
                                    post_id=submission.id,
                                    platform=SocialPlatform.REDDIT,
                                    content=f"{submission.title} {submission.selftext}",
                                    author=str(submission.author) if submission.author else "unknown",
                                    timestamp=post_time,
                                    url=f"https://reddit.com{submission.permalink}",
                                    likes=submission.score,
                                    comments=submission.num_comments,
                                    content_type=ContentType.POST
                                )
                                
                                posts.append(post)
                                
                                # Limit posts per subreddit
                                if len(posts) >= 10:
                                    break
                        
                        if len(posts) >= 30:  # Limit total posts
                            break
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch from r/{subreddit_name}: {e}")
                    continue
            
            return posts
            
        except Exception as e:
            logger.warning(f"Reddit posts fetch failed: {e}")
            return []
    
    async def _get_twitter_posts(self, symbol: str, start_time: datetime, 
                                end_time: datetime) -> List[SocialPost]:
        """Get posts from Twitter"""
        
        if not self.twitter_client:
            return []
        
        posts = []
        
        try:
            # Search for symbol mentions
            search_queries = [f"${symbol}", f"#{symbol}", symbol]
            
            for query in search_queries:
                try:
                    tweets = tweepy.Paginator(
                        self.twitter_client.search_recent_tweets,
                        query=query,
                        tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations'],
                        max_results=100
                    ).flatten(limit=50)
                    
                    for tweet in tweets:
                        # Check if tweet is within time range
                        if start_time <= tweet.created_at <= end_time:
                            
                            metrics = tweet.public_metrics
                            engagement = (
                                metrics.get('like_count', 0) +
                                metrics.get('retweet_count', 0) * 2 +
                                metrics.get('reply_count', 0) * 1.5
                            )
                            
                            post = SocialPost(
                                post_id=tweet.id,
                                platform=SocialPlatform.TWITTER,
                                content=tweet.text,
                                author=tweet.author_id,
                                timestamp=tweet.created_at,
                                url=f"https://twitter.com/i/status/{tweet.id}",
                                likes=metrics.get('like_count', 0),
                                shares=metrics.get('retweet_count', 0),
                                comments=metrics.get('reply_count', 0),
                                engagement_score=engagement,
                                content_type=ContentType.POST
                            )
                            
                            posts.append(post)
                
                except Exception as e:
                    logger.warning(f"Failed to search Twitter for {query}: {e}")
                    continue
            
            return posts
            
        except Exception as e:
            logger.warning(f"Twitter posts fetch failed: {e}")
            return []
    
    async def _get_stocktwits_posts(self, symbol: str, start_time: datetime, 
                                   end_time: datetime) -> List[SocialPost]:
        """Get posts from StockTwits"""
        
        posts = []
        config = self.platform_configs[SocialPlatform.STOCKTWITS]
        
        try:
            url = f"{config['base_url']}/streams/symbol/{symbol}.json"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for message in data.get('messages', []):
                            try:
                                # Parse timestamp
                                created_at = datetime.strptime(
                                    message['created_at'], 
                                    '%Y-%m-%dT%H:%M:%SZ'
                                )
                                
                                # Check if post is within time range
                                if start_time <= created_at <= end_time:
                                    
                                    post = SocialPost(
                                        post_id=str(message['id']),
                                        platform=SocialPlatform.STOCKTWITS,
                                        content=message['body'],
                                        author=message['user']['username'],
                                        timestamp=created_at,
                                        url=f"https://stocktwits.com/message/{message['id']}",
                                        likes=message.get('likes', {}).get('total', 0),
                                        author_followers=message['user'].get('followers', 0),
                                        content_type=ContentType.POST
                                    )
                                    
                                    posts.append(post)
                                    
                            except Exception as e:
                                logger.warning(f"Failed to parse StockTwits message: {e}")
                                continue
            
            return posts
            
        except Exception as e:
            logger.warning(f"StockTwits posts fetch failed: {e}")
            return []
    
    async def _process_post(self, post: SocialPost, target_symbol: str = None) -> Optional[SocialPost]:
        """Process and analyze social media post"""
        
        start_time = time.time()
        
        try:
            # Generate content hash
            post.content_hash = self._generate_content_hash(post)
            
            # Skip if already processed
            if post.content_hash in self.processed_hashes:
                return None
            
            # Calculate sentiment score
            post.sentiment_score = self._calculate_post_sentiment(post.content)
            
            # Calculate relevance score
            post.relevance_score = self._calculate_post_relevance(post, target_symbol)
            
            # Extract symbols mentioned
            post.symbols_mentioned = self._extract_symbols_from_post(post.content)
            
            # Extract hashtags and mentions
            post.hashtags = self._extract_hashtags(post.content)
            post.mentions = self._extract_mentions(post.content)
            
            # Extract keywords
            post.keywords = self._extract_keywords_from_post(post.content)
            
            # Calculate engagement score
            post.engagement_score = self._calculate_engagement_score(post)
            
            # Determine author tier
            post.author_tier = self._determine_author_tier(post.author_followers)
            
            # Calculate author influence score
            post.author_influence_score = self._calculate_influence_score(post)
            
            # Mark as processed
            post.processed = True
            post.processing_time = time.time() - start_time
            
            # Store in database
            await self._store_post(post)
            
            # Add to processed hashes
            self.processed_hashes.add(post.content_hash)
            
            return post
            
        except Exception as e:
            logger.warning(f"Failed to process post {post.post_id}: {e}")
            return None
    
    def _generate_content_hash(self, post: SocialPost) -> str:
        """Generate hash for post content to detect duplicates"""
        
        # Normalize content
        content = re.sub(r'http\S+|www\S+|https\S+', '', post.content, flags=re.MULTILINE)
        content = re.sub(r'\s+', ' ', content).strip().lower()
        
        # Include author and platform to allow same content from different sources
        hash_input = f"{post.platform.value}_{post.author}_{content}"
        
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _calculate_post_sentiment(self, content: str) -> float:
        """Calculate sentiment score for post content"""
        
        try:
            # Clean content
            cleaned_content = self._clean_post_content(content)
            
            # VADER sentiment
            vader_scores = self.vader_analyzer.polarity_scores(cleaned_content)
            
            # TextBlob sentiment
            blob = TextBlob(cleaned_content)
            textblob_score = blob.sentiment.polarity
            
            # Crypto/finance specific sentiment
            crypto_score = self._calculate_crypto_sentiment(cleaned_content)
            finance_score = self._calculate_finance_sentiment(cleaned_content)
            
            # Combine scores
            combined_score = (
                vader_scores['compound'] * 0.3 +
                textblob_score * 0.2 +
                crypto_score * 0.3 +
                finance_score * 0.2
            )
            
            return max(-1.0, min(1.0, combined_score))
            
        except Exception as e:
            logger.warning(f"Sentiment calculation failed: {e}")
            return 0.0
    
    def _clean_post_content(self, content: str) -> str:
        """Clean post content for analysis"""
        
        # Remove URLs
        content = re.sub(r'http\S+|www\S+|https\S+', '', content, flags=re.MULTILINE)
        
        # Remove excessive emojis (keep some for sentiment)
        content = re.sub(r'[🚀🌙💎]{3,}', '🚀', content)  # Reduce emoji spam
        
        # Remove mentions and hashtags for sentiment (but keep the words)
        content = re.sub(r'[@#]', '', content)
        
        # Normalize whitespace
        content = ' '.join(content.split())
        
        return content
    
    def _calculate_crypto_sentiment(self, content: str) -> float:
        """Calculate crypto-specific sentiment"""
        
        content_lower = content.lower()
        sentiment_sum = 0.0
        word_count = 0
        
        for keyword, sentiment in self.crypto_keywords.items():
            if keyword in content_lower:
                sentiment_sum += sentiment
                word_count += 1
        
        if word_count == 0:
            return 0.0
        
        return sentiment_sum / word_count
    
    def _calculate_finance_sentiment(self, content: str) -> float:
        """Calculate finance-specific sentiment"""
        
        content_lower = content.lower()
        sentiment_sum = 0.0
        word_count = 0
        
        for keyword, sentiment in self.finance_keywords.items():
            if keyword in content_lower:
                sentiment_sum += sentiment
                word_count += 1
        
        if word_count == 0:
            return 0.0
        
        return sentiment_sum / word_count
    
    def _calculate_post_relevance(self, post: SocialPost, target_symbol: str = None) -> float:
        """Calculate relevance score for post"""
        
        content = post.content.lower()
        score = 0.0
        
        # Symbol-specific relevance
        if target_symbol:
            symbol_variations = [
                target_symbol.upper(),
                target_symbol.lower(),
                f"${target_symbol.upper()}",
                f"#{target_symbol.lower()}"
            ]
            
            for variation in symbol_variations:
                if variation.lower() in content:
                    score += 0.4
                    break
        
        # Financial keyword relevance
        for keyword in self.finance_keywords:
            if keyword in content:
                score += 0.1
        
        # Crypto keyword relevance
        for keyword in self.crypto_keywords:
            if keyword in content:
                score += 0.1
        
        # Platform-specific bonuses
        if post.platform == SocialPlatform.STOCKTWITS:
            score += 0.2  # StockTwits is finance-focused
        elif post.platform == SocialPlatform.REDDIT:
            score += 0.1  # Reddit has good financial discussions
        
        # Engagement bonus
        if post.engagement_score > 50:
            score += 0.1
        
        # Author credibility bonus
        if post.author_verified:
            score += 0.1
        
        return min(1.0, score)
    
    def _extract_symbols_from_post(self, content: str) -> List[str]:
        """Extract trading symbols from post content"""
        
        symbols = []
        
        # Symbol patterns
        patterns = [
            r'\$([A-Z]{1,5})\b',  # $AAPL
            r'#([A-Z]{1,5})\b',   # #AAPL
            r'\b([A-Z]{2,5})\s*(?:stock|shares|ticker)',  # AAPL stock
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content.upper())
            symbols.extend(matches)
        
        # Filter out common false positives
        false_positives = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN',
            'USD', 'CEO', 'IPO', 'SEC', 'FDA', 'API', 'APP', 'LOL', 'OMG'
        }
        
        valid_symbols = [s for s in symbols if s not in false_positives and len(s) <= 5]
        
        return list(set(valid_symbols))  # Remove duplicates
    
    def _extract_hashtags(self, content: str) -> List[str]:
        """Extract hashtags from post content"""
        
        hashtags = re.findall(r'#(\w+)', content)
        return [tag.lower() for tag in hashtags]
    
    def _extract_mentions(self, content: str) -> List[str]:
        """Extract mentions from post content"""
        
        mentions = re.findall(r'@(\w+)', content)
        return mentions
    
    def _extract_keywords_from_post(self, content: str) -> List[str]:
        """Extract relevant keywords from post content"""
        
        content_lower = content.lower()
        keywords = []
        
        # Find financial keywords
        for keyword in self.finance_keywords:
            if keyword in content_lower:
                keywords.append(keyword)
        
        # Find crypto keywords
        for keyword in self.crypto_keywords:
            if keyword in content_lower:
                keywords.append(keyword)
        
        return list(set(keywords))[:10]  # Limit to 10 keywords
    
    def _calculate_engagement_score(self, post: SocialPost) -> float:
        """Calculate engagement score for post"""
        
        # Base engagement metrics
        engagement = (
            post.likes * 1.0 +
            post.shares * 2.0 +
            post.comments * 1.5 +
            post.views * 0.01
        )
        
        # Normalize by platform
        if post.platform == SocialPlatform.TWITTER:
            # Twitter typically has higher engagement numbers
            engagement = engagement / 10
        elif post.platform == SocialPlatform.REDDIT:
            # Reddit scores can be negative, normalize differently
            engagement = max(0, post.likes) + post.comments * 2
        
        # Author influence multiplier
        if post.author_followers > 0:
            follower_multiplier = min(2.0, 1 + (post.author_followers / 100000))
            engagement *= follower_multiplier
        
        # Verified author bonus
        if post.author_verified:
            engagement *= 1.5
        
        return min(1000.0, engagement)  # Cap at 1000
    
    def _determine_author_tier(self, followers: int) -> InfluencerTier:
        """Determine author tier based on followers"""
        
        if followers >= 1000000:
            return InfluencerTier.MEGA
        elif followers >= 100000:
            return InfluencerTier.MACRO
        elif followers >= 10000:
            return InfluencerTier.MICRO
        elif followers >= 1000:
            return InfluencerTier.NANO
        else:
            return InfluencerTier.REGULAR
    
    def _calculate_influence_score(self, post: SocialPost) -> float:
        """Calculate author influence score"""
        
        score = 0.0
        
        # Follower-based score
        if post.author_followers > 0:
            score += min(0.5, post.author_followers / 1000000)  # Max 0.5 for followers
        
        # Engagement rate
        if post.author_followers > 0:
            engagement_rate = post.engagement_score / max(1, post.author_followers)
            score += min(0.3, engagement_rate * 1000)  # Max 0.3 for engagement rate
        
        # Verified bonus
        if post.author_verified:
            score += 0.2
        
        return min(1.0, score)
    
    async def _store_post(self, post: SocialPost):
        """Store post in database"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO social_posts (
                        post_id, platform, content, author, timestamp, url,
                        likes, shares, comments, views, engagement_score,
                        sentiment_score, relevance_score, author_followers,
                        author_verified, symbols_mentioned, hashtags, keywords,
                        content_hash, processed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    post.post_id,
                    post.platform.value,
                    post.content,
                    post.author,
                    post.timestamp.isoformat(),
                    post.url,
                    post.likes,
                    post.shares,
                    post.comments,
                    post.views,
                    post.engagement_score,
                    post.sentiment_score,
                    post.relevance_score,
                    post.author_followers,
                    post.author_verified,
                    json.dumps(post.symbols_mentioned),
                    json.dumps(post.hashtags),
                    json.dumps(post.keywords),
                    post.content_hash,
                    post.processed
                ))
                
        except Exception as e:
            logger.warning(f"Failed to store post: {e}")
    
    async def analyze_social_sentiment(self, symbol: str, timeframe: str = "1h") -> SocialSentimentMetrics:
        """Analyze overall social sentiment for a symbol"""
        
        try:
            # Get posts for analysis
            posts = await self.get_posts(symbol, timeframe)
            
            if not posts:
                return SocialSentimentMetrics(
                    symbol=symbol,
                    platform=SocialPlatform.REDDIT,  # Default
                    timeframe=timeframe,
                    timestamp=datetime.now(),
                    confidence=0.0
                )
            
            # Calculate metrics by platform
            platform_metrics = {}
            
            for platform in SocialPlatform:
                platform_posts = [p for p in posts if p.platform == platform]
                
                if platform_posts:
                    metrics = self._calculate_platform_metrics(symbol, platform, timeframe, platform_posts)
                    platform_metrics[platform] = metrics
            
            # Return primary platform metrics (or combined)
            if platform_metrics:
                primary_platform = max(platform_metrics.keys(), 
                                     key=lambda p: len([post for post in posts if post.platform == p]))
                return platform_metrics[primary_platform]
            
            # Fallback empty metrics
            return SocialSentimentMetrics(
                symbol=symbol,
                platform=SocialPlatform.REDDIT,
                timeframe=timeframe,
                timestamp=datetime.now(),
                confidence=0.0
            )
            
        except Exception as e:
            logger.error(f"Social sentiment analysis failed: {e}")
            return SocialSentimentMetrics(
                symbol=symbol,
                platform=SocialPlatform.REDDIT,
                timeframe=timeframe,
                timestamp=datetime.now(),
                confidence=0.0
            )
    
    def _calculate_platform_metrics(self, symbol: str, platform: SocialPlatform, 
                                   timeframe: str, posts: List[SocialPost]) -> SocialSentimentMetrics:
        """Calculate sentiment metrics for a platform"""
        
        if not posts:
            return SocialSentimentMetrics(
                symbol=symbol,
                platform=platform,
                timeframe=timeframe,
                timestamp=datetime.now(),
                confidence=0.0
            )
        
        # Basic counts
        total_posts = len(posts)
        positive_posts = len([p for p in posts if p.sentiment_score > 0.1])
        negative_posts = len([p for p in posts if p.sentiment_score < -0.1])
        neutral_posts = total_posts - positive_posts - negative_posts
        
        # Average sentiment
        avg_sentiment = np.mean([p.sentiment_score for p in posts])
        
        # Engagement metrics
        total_engagement = sum(p.engagement_score for p in posts)
        unique_authors = len(set(p.author for p in posts))
        
        # Influencer sentiment
        influencer_posts = [p for p in posts if p.author_tier in [InfluencerTier.MICRO, InfluencerTier.MACRO, InfluencerTier.MEGA]]
        influencer_sentiment = np.mean([p.sentiment_score for p in influencer_posts]) if influencer_posts else 0.0
        
        # Calculate confidence based on data quality
        confidence = min(1.0, total_posts / 20) * min(1.0, unique_authors / 10)
        
        # Calculate trending score
        trending_score = min(1.0, total_engagement / 1000)
        
        return SocialSentimentMetrics(
            symbol=symbol,
            platform=platform,
            timeframe=timeframe,
            timestamp=datetime.now(),
            total_posts=total_posts,
            total_engagement=int(total_engagement),
            unique_authors=unique_authors,
            positive_posts=positive_posts,
            negative_posts=negative_posts,
            neutral_posts=neutral_posts,
            average_sentiment=avg_sentiment,
            influencer_sentiment=influencer_sentiment,
            trending_score=trending_score,
            confidence=confidence,
            data_quality=min(1.0, total_posts / 50)
        )
    
    async def get_trending_topics(self, platform: SocialPlatform = None, 
                                 limit: int = 10) -> List[TrendingTopic]:
        """Get trending topics from social media"""
        
        # This would analyze hashtags, keywords, and mentions
        # For now, return placeholder
        return []
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get social sentiment processing statistics"""
        
        return {
            'platforms_enabled': len([p for p, config in self.platform_configs.items() if config.get('enabled', False)]),
            'cache_size': len(self.sentiment_cache),
            'processed_hashes': len(self.processed_hashes),
            'reddit_enabled': bool(self.reddit_client),
            'twitter_enabled': bool(self.twitter_client),
        }

# ==================== CONVENIENCE FUNCTIONS ====================

async def get_social_sentiment_overview(symbols: List[str], 
                                       timeframe: str = "4h") -> Dict[str, SocialSentimentMetrics]:
    """Get social sentiment overview for multiple symbols"""
    
    processor = SocialSentimentProcessor()
    
    sentiment_overview = {}
    for symbol in symbols:
        metrics = await processor.analyze_social_sentiment(symbol, timeframe)
        sentiment_overview[symbol] = metrics
    
    return sentiment_overview

async def monitor_social_mentions(symbol: str, callback_func, 
                                 check_interval: int = 300) -> None:
    """Monitor social media mentions for a symbol"""
    
    processor = SocialSentimentProcessor()
    last_check = datetime.now()
    
    try:
        while True:
            posts = await processor.get_posts(symbol, "1h")
            
            # Filter posts since last check
            new_posts = [p for p in posts if p.timestamp > last_check]
            
            if new_posts:
                await callback_func(symbol, new_posts)
            
            last_check = datetime.now()
            await asyncio.sleep(check_interval)
            
    except Exception as e:
        logger.error(f"Social monitoring failed: {e}")

def calculate_social_sentiment_signal(metrics: SocialSentimentMetrics) -> Dict[str, Any]:
    """Convert social sentiment metrics to trading signals"""
    
    signal = {
        'action': 'hold',
        'strength': 0.0,
        'confidence': metrics.confidence,
        'social_momentum': 0.0,
        'notes': []
    }
    
    # Calculate social momentum
    bullish_ratio = metrics.positive_posts / max(1, metrics.total_posts)
    bearish_ratio = metrics.negative_posts / max(1, metrics.total_posts)
    
    signal['social_momentum'] = bullish_ratio - bearish_ratio
    
    # Generate action based on sentiment and momentum
    if metrics.average_sentiment > 0.3 and bullish_ratio > 0.6 and metrics.confidence > 0.5:
        signal['action'] = 'buy'
        signal['strength'] = min(1.0, metrics.average_sentiment * metrics.confidence)
        signal['notes'].append('Strong bullish social sentiment')
    elif metrics.average_sentiment < -0.3 and bearish_ratio > 0.6 and metrics.confidence > 0.5:
        signal['action'] = 'sell'
        signal['strength'] = min(1.0, abs(metrics.average_sentiment) * metrics.confidence)
        signal['notes'].append('Strong bearish social sentiment')
    
    # Add trending information
    if metrics.trending_score > 0.7:
        signal['notes'].append('High social media trending score')
    
    if metrics.influencer_sentiment != 0 and abs(metrics.influencer_sentiment) > 0.5:
        direction = "positive" if metrics.influencer_sentiment > 0 else "negative"
        signal['notes'].append(f'Influencers showing {direction} sentiment')
    
    return signal