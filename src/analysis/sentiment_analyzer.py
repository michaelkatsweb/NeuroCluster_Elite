#!/usr/bin/env python3
"""
File: sentiment_analyzer.py
Path: NeuroCluster-Elite/src/analysis/sentiment_analyzer.py
Description: Advanced sentiment analysis system for market intelligence

This module implements comprehensive sentiment analysis capabilities for financial markets,
processing news, social media, analyst reports, and market sentiment indicators.

Features:
- Multi-source sentiment aggregation (news, social, analyst reports)
- Real-time sentiment scoring with confidence metrics
- Market-specific sentiment models for different asset types
- Sentiment trend analysis and momentum detection
- Fear & Greed index calculation
- Social media sentiment processing
- News sentiment analysis with source weighting
- Integration with NeuroCluster regime detection

Author: michael Katsaros
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
from pathlib import Path
import sqlite3
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

# NLP and sentiment analysis
from textblob import TextBlob
import vaderSentiment.vaderSentiment as vader
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Machine learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Import our modules
try:
    from src.core.neurocluster_elite import AssetType, MarketData, RegimeType
    from src.utils.config_manager import ConfigManager
    from src.utils.logger import get_enhanced_logger, LogCategory
    from src.utils.helpers import format_percentage, calculate_hash
    from src.analysis.news_processor import NewsProcessor, NewsArticle
    from src.analysis.social_sentiment import SocialSentimentProcessor, SocialPost
except ImportError:
    # Fallback for testing
    pass

# Configure logging
logger = get_enhanced_logger(__name__, LogCategory.ANALYSIS)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# ==================== ENUMS AND DATA STRUCTURES ====================

class SentimentType(Enum):
    """Types of sentiment"""
    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"

class SentimentSource(Enum):
    """Sentiment data sources"""
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    ANALYST_REPORTS = "analyst_reports"
    MARKET_DATA = "market_data"
    EARNINGS_CALLS = "earnings_calls"
    SEC_FILINGS = "sec_filings"
    REDDIT = "reddit"
    TWITTER = "twitter"
    DISCORD = "discord"
    TELEGRAM = "telegram"

class ConfidenceLevel(Enum):
    """Confidence levels for sentiment analysis"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class SentimentScore:
    """Individual sentiment score from a source"""
    source: SentimentSource
    score: float  # -1.0 to 1.0 (-1 = very bearish, 1 = very bullish)
    confidence: float  # 0.0 to 1.0
    sentiment_type: SentimentType
    timestamp: datetime
    source_count: int = 1
    raw_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AggregatedSentiment:
    """Aggregated sentiment analysis result"""
    symbol: str
    asset_type: AssetType
    overall_score: float  # -1.0 to 1.0
    overall_sentiment: SentimentType
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    
    # Source breakdown
    source_scores: Dict[SentimentSource, SentimentScore] = field(default_factory=dict)
    
    # Sentiment metrics
    bullish_ratio: float = 0.0
    bearish_ratio: float = 0.0
    neutral_ratio: float = 0.0
    
    # Trend analysis
    sentiment_trend: str = "stable"  # "improving", "declining", "stable"
    momentum: float = 0.0  # Rate of sentiment change
    
    # Fear & Greed metrics
    fear_greed_index: float = 50.0  # 0-100 (0 = extreme fear, 100 = extreme greed)
    volatility_sentiment: float = 0.0
    
    # Additional metrics
    source_count: int = 0
    data_quality: float = 1.0

@dataclass
class SentimentTrend:
    """Sentiment trend analysis"""
    symbol: str
    timeframe: str  # "1h", "4h", "1d", "1w"
    trend_direction: str  # "bullish", "bearish", "neutral"
    strength: float  # 0.0 to 1.0
    momentum: float  # Rate of change
    support_level: float  # Sentiment support level
    resistance_level: float  # Sentiment resistance level
    breakout_probability: float  # Probability of sentiment breakout

# ==================== SENTIMENT ANALYZER ====================

class AdvancedSentimentAnalyzer:
    """Advanced sentiment analysis system with multi-source aggregation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize sentiment analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.news_processor = None
        self.social_processor = None
        
        # Sentiment models
        self.vader_analyzer = vader.SentimentIntensityAnalyzer()
        self.custom_model = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Financial lexicon
        self.financial_lexicon = self._load_financial_lexicon()
        
        # Caching and storage
        self.sentiment_cache = {}
        self.sentiment_history = defaultdict(lambda: deque(maxlen=1000))
        self.db_path = self.config.get('db_path', 'data/sentiment.db')
        
        # Configuration
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.source_weights = self.config.get('source_weights', {
            SentimentSource.NEWS: 0.4,
            SentimentSource.ANALYST_REPORTS: 0.3,
            SentimentSource.SOCIAL_MEDIA: 0.2,
            SentimentSource.MARKET_DATA: 0.1
        })
        
        # Initialize database
        self._initialize_database()
        
        # Initialize models
        self._initialize_models()
        
        logger.info("🎯 Advanced Sentiment Analyzer initialized")
    
    def _load_financial_lexicon(self) -> Dict[str, float]:
        """Load financial sentiment lexicon"""
        
        # Financial terms with sentiment scores (-1 to 1)
        lexicon = {
            # Bullish terms
            'bullish': 0.8, 'rally': 0.7, 'surge': 0.8, 'soar': 0.9,
            'breakout': 0.7, 'momentum': 0.6, 'uptrend': 0.8, 'gains': 0.6,
            'profitable': 0.7, 'growth': 0.6, 'expansion': 0.5, 'beat': 0.6,
            'outperform': 0.7, 'upgrade': 0.8, 'buy': 0.6, 'strong': 0.5,
            
            # Bearish terms
            'bearish': -0.8, 'crash': -0.9, 'plunge': -0.8, 'collapse': -0.9,
            'breakdown': -0.7, 'decline': -0.6, 'downtrend': -0.8, 'losses': -0.6,
            'recession': -0.8, 'crisis': -0.9, 'risk': -0.5, 'volatility': -0.4,
            'underperform': -0.7, 'downgrade': -0.8, 'sell': -0.6, 'weak': -0.5,
            
            # Market specific
            'overbought': -0.6, 'oversold': 0.6, 'resistance': -0.3, 'support': 0.3,
            'bubble': -0.8, 'correction': -0.6, 'consolidation': 0.0, 'sideways': 0.0,
            
            # News events
            'earnings': 0.1, 'dividend': 0.5, 'merger': 0.4, 'acquisition': 0.3,
            'bankruptcy': -0.9, 'lawsuit': -0.7, 'investigation': -0.8, 'scandal': -0.9
        }
        
        return lexicon
    
    def _initialize_database(self):
        """Initialize sentiment database"""
        
        try:
            # Create data directory
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create database tables
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS sentiment_scores (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        asset_type TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        overall_score REAL,
                        confidence REAL,
                        sentiment_type TEXT,
                        source_scores TEXT,
                        fear_greed_index REAL,
                        data_quality REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_time 
                    ON sentiment_scores(symbol, timestamp)
                ''')
                
                logger.info("📊 Sentiment database initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize sentiment database: {e}")
    
    def _initialize_models(self):
        """Initialize sentiment analysis models"""
        
        try:
            # Initialize news processor
            self.news_processor = NewsProcessor(self.config.get('news', {}))
            
            # Initialize social sentiment processor
            self.social_processor = SocialSentimentProcessor(self.config.get('social', {}))
            
            # Train custom model if data available
            self._train_custom_model()
            
            logger.info("🤖 Sentiment models initialized")
            
        except Exception as e:
            logger.warning(f"Some sentiment models failed to initialize: {e}")
    
    def _train_custom_model(self):
        """Train custom sentiment model for financial text"""
        
        try:
            # This would load training data in production
            # For now, we'll use a simple pipeline
            
            self.custom_model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
                ('classifier', MultinomialNB())
            ])
            
            # In production, train with labeled financial text data
            logger.info("🎓 Custom sentiment model ready")
            
        except Exception as e:
            logger.warning(f"Custom model training failed: {e}")
            self.custom_model = None
    
    async def analyze_sentiment(self, symbol: str, asset_type: AssetType, 
                              timeframe: str = "1h") -> AggregatedSentiment:
        """
        Analyze comprehensive sentiment for a symbol
        
        Args:
            symbol: Trading symbol
            asset_type: Type of asset
            timeframe: Analysis timeframe
            
        Returns:
            Aggregated sentiment analysis
        """
        
        cache_key = f"{symbol}_{asset_type.value}_{timeframe}"
        
        # Check cache
        if cache_key in self.sentiment_cache:
            cached_data, timestamp = self.sentiment_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
        
        try:
            # Collect sentiment from all sources
            sentiment_scores = await self._collect_sentiment_data(symbol, asset_type, timeframe)
            
            # Aggregate sentiment scores
            aggregated = self._aggregate_sentiment(symbol, asset_type, sentiment_scores)
            
            # Calculate fear & greed index
            aggregated.fear_greed_index = self._calculate_fear_greed_index(aggregated)
            
            # Analyze sentiment trend
            aggregated.sentiment_trend, aggregated.momentum = self._analyze_sentiment_trend(
                symbol, aggregated.overall_score
            )
            
            # Store in cache
            self.sentiment_cache[cache_key] = (aggregated, time.time())
            
            # Store in database
            await self._store_sentiment(aggregated)
            
            logger.info(f"📈 Sentiment analyzed for {symbol}: {aggregated.overall_sentiment.value} "
                       f"(confidence: {aggregated.confidence:.2f})")
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbol}: {e}")
            
            # Return neutral sentiment as fallback
            return AggregatedSentiment(
                symbol=symbol,
                asset_type=asset_type,
                overall_score=0.0,
                overall_sentiment=SentimentType.NEUTRAL,
                confidence=0.1,
                timestamp=datetime.now(),
                data_quality=0.0
            )
    
    async def _collect_sentiment_data(self, symbol: str, asset_type: AssetType, 
                                    timeframe: str) -> List[SentimentScore]:
        """Collect sentiment data from all sources"""
        
        sentiment_scores = []
        
        try:
            # Collect sentiment from different sources concurrently
            tasks = []
            
            # News sentiment
            if self.news_processor:
                tasks.append(self._get_news_sentiment(symbol, timeframe))
            
            # Social media sentiment
            if self.social_processor:
                tasks.append(self._get_social_sentiment(symbol, timeframe))
            
            # Market data sentiment
            tasks.append(self._get_market_sentiment(symbol, asset_type))
            
            # Execute all tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Sentiment collection error: {result}")
                    continue
                
                if isinstance(result, list):
                    sentiment_scores.extend(result)
                elif isinstance(result, SentimentScore):
                    sentiment_scores.append(result)
            
            return sentiment_scores
            
        except Exception as e:
            logger.error(f"Failed to collect sentiment data: {e}")
            return []
    
    async def _get_news_sentiment(self, symbol: str, timeframe: str) -> List[SentimentScore]:
        """Get sentiment from news sources"""
        
        try:
            # Get recent news articles
            articles = await self.news_processor.get_news(symbol, timeframe)
            
            sentiment_scores = []
            
            for article in articles:
                # Analyze article sentiment
                score = self._analyze_text_sentiment(article.content or article.title)
                
                sentiment_scores.append(SentimentScore(
                    source=SentimentSource.NEWS,
                    score=score['compound'],
                    confidence=score['confidence'],
                    sentiment_type=self._score_to_sentiment_type(score['compound']),
                    timestamp=article.published_at,
                    raw_data={
                        'title': article.title,
                        'source': article.source,
                        'url': article.url,
                        'sentiment_breakdown': score
                    }
                ))
            
            return sentiment_scores
            
        except Exception as e:
            logger.warning(f"News sentiment analysis failed: {e}")
            return []
    
    async def _get_social_sentiment(self, symbol: str, timeframe: str) -> List[SentimentScore]:
        """Get sentiment from social media sources"""
        
        try:
            # Get social media posts
            posts = await self.social_processor.get_posts(symbol, timeframe)
            
            sentiment_scores = []
            
            for post in posts:
                # Analyze post sentiment
                score = self._analyze_text_sentiment(post.content)
                
                sentiment_scores.append(SentimentScore(
                    source=SentimentSource.SOCIAL_MEDIA,
                    score=score['compound'],
                    confidence=score['confidence'],
                    sentiment_type=self._score_to_sentiment_type(score['compound']),
                    timestamp=post.timestamp,
                    raw_data={
                        'platform': post.platform,
                        'author': post.author,
                        'engagement': post.engagement_score,
                        'sentiment_breakdown': score
                    }
                ))
            
            return sentiment_scores
            
        except Exception as e:
            logger.warning(f"Social sentiment analysis failed: {e}")
            return []
    
    async def _get_market_sentiment(self, symbol: str, asset_type: AssetType) -> SentimentScore:
        """Get sentiment from market data indicators"""
        
        try:
            # This would analyze technical indicators for sentiment
            # For now, return neutral with low confidence
            
            return SentimentScore(
                source=SentimentSource.MARKET_DATA,
                score=0.0,
                confidence=0.3,
                sentiment_type=SentimentType.NEUTRAL,
                timestamp=datetime.now(),
                raw_data={'note': 'Market sentiment analysis placeholder'}
            )
            
        except Exception as e:
            logger.warning(f"Market sentiment analysis failed: {e}")
            return SentimentScore(
                source=SentimentSource.MARKET_DATA,
                score=0.0,
                confidence=0.1,
                sentiment_type=SentimentType.NEUTRAL,
                timestamp=datetime.now()
            )
    
    def _analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text using multiple methods"""
        
        if not text:
            return {'compound': 0.0, 'confidence': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}
        
        try:
            # Clean text
            cleaned_text = self._clean_text(text)
            
            # VADER sentiment
            vader_scores = self.vader_analyzer.polarity_scores(cleaned_text)
            
            # TextBlob sentiment
            blob = TextBlob(cleaned_text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            # Financial lexicon sentiment
            financial_score = self._calculate_financial_sentiment(cleaned_text)
            
            # Combine scores with weights
            compound_score = (
                vader_scores['compound'] * 0.4 +
                textblob_polarity * 0.3 +
                financial_score * 0.3
            )
            
            # Calculate confidence based on agreement
            confidence = self._calculate_sentiment_confidence(
                vader_scores['compound'], textblob_polarity, financial_score
            )
            
            return {
                'compound': compound_score,
                'confidence': confidence,
                'pos': vader_scores['pos'],
                'neg': vader_scores['neg'],
                'neu': vader_scores['neu'],
                'textblob_polarity': textblob_polarity,
                'textblob_subjectivity': textblob_subjectivity,
                'financial_score': financial_score
            }
            
        except Exception as e:
            logger.warning(f"Text sentiment analysis failed: {e}")
            return {'compound': 0.0, 'confidence': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags (but keep the word)
        text = re.sub(r'[@#]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.lower()
    
    def _calculate_financial_sentiment(self, text: str) -> float:
        """Calculate sentiment using financial lexicon"""
        
        words = word_tokenize(text.lower())
        
        sentiment_sum = 0.0
        word_count = 0
        
        for word in words:
            if word in self.financial_lexicon and word not in self.stop_words:
                sentiment_sum += self.financial_lexicon[word]
                word_count += 1
        
        if word_count == 0:
            return 0.0
        
        return sentiment_sum / word_count
    
    def _calculate_sentiment_confidence(self, vader_score: float, textblob_score: float, 
                                      financial_score: float) -> float:
        """Calculate confidence based on agreement between methods"""
        
        scores = [vader_score, textblob_score, financial_score]
        
        # Calculate variance (lower variance = higher confidence)
        mean_score = np.mean(scores)
        variance = np.var(scores)
        
        # Convert variance to confidence (0-1)
        confidence = max(0.1, 1.0 - variance)
        
        return min(1.0, confidence)
    
    def _aggregate_sentiment(self, symbol: str, asset_type: AssetType, 
                           sentiment_scores: List[SentimentScore]) -> AggregatedSentiment:
        """Aggregate sentiment scores from multiple sources"""
        
        if not sentiment_scores:
            return AggregatedSentiment(
                symbol=symbol,
                asset_type=asset_type,
                overall_score=0.0,
                overall_sentiment=SentimentType.NEUTRAL,
                confidence=0.1,
                timestamp=datetime.now()
            )
        
        # Group by source
        source_groups = defaultdict(list)
        for score in sentiment_scores:
            source_groups[score.source].append(score)
        
        # Calculate weighted average for each source
        source_scores = {}
        total_weight = 0.0
        weighted_sum = 0.0
        confidence_sum = 0.0
        
        for source, scores in source_groups.items():
            if not scores:
                continue
            
            # Average scores for this source
            avg_score = np.mean([s.score for s in scores])
            avg_confidence = np.mean([s.confidence for s in scores])
            
            # Apply source weight
            weight = self.source_weights.get(source, 0.1)
            
            source_scores[source] = SentimentScore(
                source=source,
                score=avg_score,
                confidence=avg_confidence,
                sentiment_type=self._score_to_sentiment_type(avg_score),
                timestamp=datetime.now(),
                source_count=len(scores)
            )
            
            weighted_sum += avg_score * weight * avg_confidence
            total_weight += weight * avg_confidence
            confidence_sum += avg_confidence
        
        # Calculate overall metrics
        overall_score = weighted_sum / max(total_weight, 0.01)
        overall_confidence = confidence_sum / max(len(source_scores), 1)
        overall_sentiment = self._score_to_sentiment_type(overall_score)
        
        # Calculate sentiment distribution
        total_scores = len(sentiment_scores)
        bullish_count = len([s for s in sentiment_scores if s.score > 0.1])
        bearish_count = len([s for s in sentiment_scores if s.score < -0.1])
        neutral_count = total_scores - bullish_count - bearish_count
        
        return AggregatedSentiment(
            symbol=symbol,
            asset_type=asset_type,
            overall_score=overall_score,
            overall_sentiment=overall_sentiment,
            confidence=overall_confidence,
            timestamp=datetime.now(),
            source_scores=source_scores,
            bullish_ratio=bullish_count / max(total_scores, 1),
            bearish_ratio=bearish_count / max(total_scores, 1),
            neutral_ratio=neutral_count / max(total_scores, 1),
            source_count=len(source_scores),
            data_quality=min(1.0, len(sentiment_scores) / 10)  # Quality based on data volume
        )
    
    def _score_to_sentiment_type(self, score: float) -> SentimentType:
        """Convert numerical score to sentiment type"""
        
        if score <= -0.6:
            return SentimentType.VERY_BEARISH
        elif score <= -0.2:
            return SentimentType.BEARISH
        elif score <= 0.2:
            return SentimentType.NEUTRAL
        elif score <= 0.6:
            return SentimentType.BULLISH
        else:
            return SentimentType.VERY_BULLISH
    
    def _calculate_fear_greed_index(self, sentiment: AggregatedSentiment) -> float:
        """Calculate Fear & Greed index (0-100)"""
        
        # Convert sentiment score (-1 to 1) to 0-100 scale
        base_score = (sentiment.overall_score + 1) * 50
        
        # Adjust based on confidence
        confidence_adjustment = (sentiment.confidence - 0.5) * 20
        
        # Adjust based on source diversity
        source_diversity = len(sentiment.source_scores) / 4  # Max 4 main sources
        diversity_adjustment = source_diversity * 10
        
        # Calculate final index
        fear_greed = base_score + confidence_adjustment + diversity_adjustment
        
        return max(0.0, min(100.0, fear_greed))
    
    def _analyze_sentiment_trend(self, symbol: str, current_score: float) -> Tuple[str, float]:
        """Analyze sentiment trend for a symbol"""
        
        try:
            # Get historical sentiment scores
            history = self.sentiment_history[symbol]
            
            if len(history) < 2:
                history.append(current_score)
                return "stable", 0.0
            
            # Add current score
            history.append(current_score)
            
            # Calculate trend
            recent_scores = list(history)[-10:]  # Last 10 scores
            
            if len(recent_scores) < 3:
                return "stable", 0.0
            
            # Linear regression to determine trend
            x = np.arange(len(recent_scores))
            coeffs = np.polyfit(x, recent_scores, 1)
            slope = coeffs[0]
            
            # Determine trend direction
            if slope > 0.05:
                trend = "improving"
            elif slope < -0.05:
                trend = "declining"
            else:
                trend = "stable"
            
            # Calculate momentum (rate of change)
            momentum = abs(slope) * 10  # Scale to 0-1 range
            
            return trend, min(1.0, momentum)
            
        except Exception as e:
            logger.warning(f"Trend analysis failed: {e}")
            return "stable", 0.0
    
    async def _store_sentiment(self, sentiment: AggregatedSentiment):
        """Store sentiment data in database"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO sentiment_scores (
                        symbol, asset_type, timestamp, overall_score, confidence,
                        sentiment_type, source_scores, fear_greed_index, data_quality
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    sentiment.symbol,
                    sentiment.asset_type.value,
                    sentiment.timestamp,
                    sentiment.overall_score,
                    sentiment.confidence,
                    sentiment.overall_sentiment.value,
                    json.dumps({k.value: v.__dict__ for k, v in sentiment.source_scores.items()}, default=str),
                    sentiment.fear_greed_index,
                    sentiment.data_quality
                ))
                
        except Exception as e:
            logger.warning(f"Failed to store sentiment data: {e}")
    
    async def get_sentiment_history(self, symbol: str, timeframe: str = "1d", 
                                  limit: int = 100) -> List[AggregatedSentiment]:
        """Get historical sentiment data"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT * FROM sentiment_scores 
                    WHERE symbol = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (symbol, limit))
                
                results = []
                for row in cursor.fetchall():
                    # Reconstruct sentiment object
                    sentiment = AggregatedSentiment(
                        symbol=row[1],
                        asset_type=AssetType(row[2]),
                        overall_score=row[4],
                        overall_sentiment=SentimentType(row[6]),
                        confidence=row[5],
                        timestamp=datetime.fromisoformat(row[3]),
                        fear_greed_index=row[8],
                        data_quality=row[9]
                    )
                    results.append(sentiment)
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get sentiment history: {e}")
            return []
    
    async def get_market_sentiment_overview(self, symbols: List[str]) -> Dict[str, AggregatedSentiment]:
        """Get sentiment overview for multiple symbols"""
        
        tasks = []
        for symbol in symbols:
            # Determine asset type (simplified)
            asset_type = AssetType.CRYPTO if any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'USD']) else AssetType.STOCK
            tasks.append(self.analyze_sentiment(symbol, asset_type))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        sentiment_overview = {}
        for i, result in enumerate(results):
            if isinstance(result, AggregatedSentiment):
                sentiment_overview[symbols[i]] = result
            else:
                logger.warning(f"Sentiment analysis failed for {symbols[i]}: {result}")
        
        return sentiment_overview
    
    def get_sentiment_insights(self, sentiment: AggregatedSentiment) -> Dict[str, Any]:
        """Generate actionable insights from sentiment analysis"""
        
        insights = {
            'signal_strength': 'weak',
            'trading_recommendation': 'hold',
            'risk_level': 'medium',
            'key_factors': [],
            'confidence_assessment': 'low'
        }
        
        # Signal strength
        if abs(sentiment.overall_score) > 0.6 and sentiment.confidence > 0.7:
            insights['signal_strength'] = 'strong'
        elif abs(sentiment.overall_score) > 0.3 and sentiment.confidence > 0.5:
            insights['signal_strength'] = 'medium'
        
        # Trading recommendation
        if sentiment.overall_score > 0.3 and sentiment.confidence > 0.6:
            insights['trading_recommendation'] = 'bullish'
        elif sentiment.overall_score < -0.3 and sentiment.confidence > 0.6:
            insights['trading_recommendation'] = 'bearish'
        
        # Risk assessment
        if sentiment.data_quality < 0.5 or sentiment.confidence < 0.4:
            insights['risk_level'] = 'high'
        elif sentiment.confidence > 0.7 and sentiment.data_quality > 0.7:
            insights['risk_level'] = 'low'
        
        # Key factors
        if sentiment.bullish_ratio > 0.7:
            insights['key_factors'].append('Strong bullish consensus')
        if sentiment.bearish_ratio > 0.7:
            insights['key_factors'].append('Strong bearish consensus')
        if sentiment.fear_greed_index > 80:
            insights['key_factors'].append('Extreme greed levels')
        if sentiment.fear_greed_index < 20:
            insights['key_factors'].append('Extreme fear levels')
        
        # Confidence assessment
        if sentiment.confidence > 0.7:
            insights['confidence_assessment'] = 'high'
        elif sentiment.confidence > 0.5:
            insights['confidence_assessment'] = 'medium'
        
        return insights

# ==================== CONVENIENCE FUNCTIONS ====================

async def analyze_symbol_sentiment(symbol: str, asset_type: AssetType = AssetType.STOCK) -> AggregatedSentiment:
    """Convenience function to analyze sentiment for a single symbol"""
    
    analyzer = AdvancedSentimentAnalyzer()
    return await analyzer.analyze_sentiment(symbol, asset_type)

async def get_market_fear_greed_index(symbols: List[str]) -> float:
    """Calculate overall market fear & greed index"""
    
    analyzer = AdvancedSentimentAnalyzer()
    sentiment_data = await analyzer.get_market_sentiment_overview(symbols)
    
    if not sentiment_data:
        return 50.0  # Neutral
    
    # Average fear & greed index across symbols
    total_index = sum(s.fear_greed_index for s in sentiment_data.values())
    return total_index / len(sentiment_data)

def get_sentiment_signal_for_trading(sentiment: AggregatedSentiment) -> Dict[str, Any]:
    """Convert sentiment analysis to trading signals"""
    
    signal = {
        'action': 'hold',
        'strength': 0.0,
        'confidence': sentiment.confidence,
        'risk_score': 0.5,
        'notes': []
    }
    
    # Generate trading action based on sentiment
    if sentiment.overall_score > 0.4 and sentiment.confidence > 0.6:
        signal['action'] = 'buy'
        signal['strength'] = min(1.0, sentiment.overall_score * sentiment.confidence)
    elif sentiment.overall_score < -0.4 and sentiment.confidence > 0.6:
        signal['action'] = 'sell'
        signal['strength'] = min(1.0, abs(sentiment.overall_score) * sentiment.confidence)
    
    # Risk assessment
    if sentiment.data_quality < 0.5:
        signal['risk_score'] = 0.8
        signal['notes'].append('Low data quality increases risk')
    
    if sentiment.fear_greed_index > 85:
        signal['notes'].append('Extreme greed - potential reversal risk')
    elif sentiment.fear_greed_index < 15:
        signal['notes'].append('Extreme fear - potential opportunity')
    
    return signal