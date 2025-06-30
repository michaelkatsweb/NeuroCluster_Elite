#!/usr/bin/env python3
"""
File: __init__.py
Path: NeuroCluster-Elite/src/analysis/__init__.py
Description: Analysis package initialization

This module initializes the analysis components including technical indicators,
sentiment analysis, news processing, market scanning, and social sentiment tracking
for the NeuroCluster Elite trading platform.

Author: Your Name
Created: 2025-06-29
Version: 1.0.0
License: MIT
"""

# Import main analysis components
try:
    from .technical_indicators import (
        AdvancedTechnicalIndicators, IndicatorResult, IndicatorSignal,
        IndicatorType, SignalType, CompositeIndicator
    )
    from .sentiment_analyzer import (
        SentimentAnalyzer, SentimentResult, SentimentScore,
        SentimentType, EmotionScore, AnalysisConfig
    )
    from .news_processor import (
        NewsProcessor, NewsItem, NewsAnalysis, NewsSource,
        NewsCategory, NewsImpact, NewsFilter
    )
    from .social_sentiment import (
        SocialSentimentAnalyzer, SocialPost, SocialPlatform,
        EngagementMetrics, SocialTrend, InfluencerAnalysis
    )
    from .market_scanner import (
        MarketScanner, ScanResult, ScanCriteria, AlertLevel,
        TechnicalSetup, MarketCondition, OpportunityType
    )
    
    __all__ = [
        # Technical Analysis
        'AdvancedTechnicalIndicators',
        'IndicatorResult',
        'IndicatorSignal',
        'IndicatorType',
        'SignalType',
        'CompositeIndicator',
        
        # Sentiment Analysis
        'SentimentAnalyzer',
        'SentimentResult',
        'SentimentScore',
        'SentimentType',
        'EmotionScore',
        'AnalysisConfig',
        
        # News Processing
        'NewsProcessor',
        'NewsItem',
        'NewsAnalysis',
        'NewsSource',
        'NewsCategory',
        'NewsImpact',
        'NewsFilter',
        
        # Social Sentiment
        'SocialSentimentAnalyzer',
        'SocialPost',
        'SocialPlatform',
        'EngagementMetrics',
        'SocialTrend',
        'InfluencerAnalysis',
        
        # Market Scanner
        'MarketScanner',
        'ScanResult',
        'ScanCriteria',
        'AlertLevel',
        'TechnicalSetup',
        'MarketCondition',
        'OpportunityType'
    ]
    
except ImportError as e:
    import warnings
    warnings.warn(f"Some analysis components could not be imported: {e}")
    __all__ = []

# Analysis module constants
SUPPORTED_INDICATORS = [
    'sma', 'ema', 'wma', 'macd', 'adx', 'aroon', 'parabolic_sar', 'ichimoku',
    'rsi', 'stochastic', 'williams_r', 'roc', 'momentum', 'cci', 'ultimate_oscillator',
    'bollinger_bands', 'atr', 'keltner_channels', 'donchian_channels',
    'obv', 'ad_line', 'chaikin_oscillator', 'mfi', 'vwap', 'volume_profile'
]

SENTIMENT_SOURCES = ['news', 'social_media', 'analyst_reports', 'earnings_calls', 'sec_filings']
NEWS_PROVIDERS = ['yahoo_finance', 'alpha_vantage', 'newsapi', 'reddit', 'twitter', 'bloomberg', 'reuters']
SOCIAL_PLATFORMS = ['reddit', 'twitter', 'stocktwits', 'discord', 'telegram', 'youtube']

# Default analysis configurations
DEFAULT_INDICATOR_CONFIG = {
    'trend_indicators': {
        'sma_periods': [10, 20, 50, 100, 200],
        'ema_periods': [12, 26, 50],
        'macd_config': {'fast': 12, 'slow': 26, 'signal': 9}
    },
    'momentum_indicators': {
        'rsi_period': 14,
        'stochastic_config': {'k_period': 14, 'd_period': 3},
        'williams_r_period': 14
    },
    'volatility_indicators': {
        'bollinger_config': {'period': 20, 'std_dev': 2},
        'atr_period': 14,
        'keltner_config': {'period': 20, 'multiplier': 2}
    },
    'volume_indicators': {
        'obv_enabled': True,
        'vwap_enabled': True,
        'mfi_period': 14
    }
}

DEFAULT_SENTIMENT_CONFIG = {
    'news_analysis': {
        'lookback_hours': 24,
        'min_relevance_score': 0.5,
        'sentiment_threshold': 0.1,
        'max_articles_per_symbol': 50,
        'language_filter': ['en'],
        'exclude_duplicates': True
    },
    'social_sentiment': {
        'platforms': ['reddit', 'twitter', 'stocktwits'],
        'min_engagement': 10,
        'sentiment_smoothing': 0.7,
        'influencer_weight': 2.0,
        'spam_filter_enabled': True,
        'min_account_age_days': 30
    },
    'analysis_features': {
        'emotion_detection': True,
        'entity_extraction': True,
        'trend_analysis': True,
        'anomaly_detection': True,
        'impact_scoring': True
    }
}

DEFAULT_SCANNER_CONFIG = {
    'scan_frequency': 300,  # 5 minutes
    'max_results': 50,
    'min_volume': 100000,
    'min_price': 1.0,
    'max_price': 1000.0,
    'market_cap_min': 100000000,  # $100M
    'sectors': ['Technology', 'Healthcare', 'Financial', 'Energy', 'Consumer'],
    'technical_setups': [
        'breakout', 'breakdown', 'oversold_bounce', 'momentum_surge',
        'volume_spike', 'gap_up', 'gap_down', 'trend_reversal'
    ],
    'alert_conditions': {
        'price_change_threshold': 5.0,  # 5%
        'volume_surge_multiplier': 3.0,
        'rsi_oversold': 30,
        'rsi_overbought': 70
    }
}

DEFAULT_NEWS_CONFIG = {
    'providers': ['yahoo_finance', 'newsapi', 'alpha_vantage'],
    'categories': ['earnings', 'analyst_upgrades', 'mergers', 'regulatory', 'general'],
    'impact_levels': ['high', 'medium', 'low'],
    'sentiment_analysis': True,
    'entity_recognition': True,
    'duplicate_detection': True,
    'real_time_monitoring': True,
    'historical_lookback_days': 30
}

# Analysis performance thresholds
PERFORMANCE_THRESHOLDS = {
    'sentiment_analysis': {
        'max_processing_time_ms': 500,
        'min_confidence_score': 0.6,
        'cache_ttl_minutes': 15
    },
    'news_processing': {
        'max_processing_time_ms': 1000,
        'min_relevance_score': 0.5,
        'cache_ttl_minutes': 5
    },
    'market_scanning': {
        'max_scan_time_ms': 30000,
        'max_concurrent_scans': 5,
        'cache_ttl_minutes': 1
    },
    'technical_analysis': {
        'max_calculation_time_ms': 200,
        'min_data_points': 20,
        'cache_ttl_minutes': 5
    }
}

def get_available_indicators():
    """Get list of available technical indicators"""
    return SUPPORTED_INDICATORS.copy()

def get_supported_sentiment_sources():
    """Get list of supported sentiment sources"""
    return SENTIMENT_SOURCES.copy()

def get_supported_news_providers():
    """Get list of supported news providers"""
    return NEWS_PROVIDERS.copy()

def get_supported_social_platforms():
    """Get list of supported social platforms"""
    return SOCIAL_PLATFORMS.copy()

def get_analysis_info():
    """Get analysis module information"""
    return {
        'components': len(__all__),
        'indicators': len(SUPPORTED_INDICATORS),
        'sentiment_sources': len(SENTIMENT_SOURCES),
        'news_providers': len(NEWS_PROVIDERS),
        'social_platforms': len(SOCIAL_PLATFORMS),
        'default_configs': {
            'indicators': len(DEFAULT_INDICATOR_CONFIG),
            'sentiment': len(DEFAULT_SENTIMENT_CONFIG),
            'scanner': len(DEFAULT_SCANNER_CONFIG),
            'news': len(DEFAULT_NEWS_CONFIG)
        }
    }

def create_technical_analyzer(config: dict = None):
    """
    Create technical indicators analyzer with configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        AdvancedTechnicalIndicators instance
    """
    from .technical_indicators import AdvancedTechnicalIndicators
    
    analyzer_config = DEFAULT_INDICATOR_CONFIG.copy()
    if config:
        analyzer_config.update(config)
    
    return AdvancedTechnicalIndicators(analyzer_config)

def create_sentiment_analyzer(config: dict = None):
    """
    Create sentiment analyzer with configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SentimentAnalyzer instance
    """
    from .sentiment_analyzer import SentimentAnalyzer, AnalysisConfig
    
    analyzer_config = DEFAULT_SENTIMENT_CONFIG.copy()
    if config:
        analyzer_config.update(config)
    
    analysis_config = AnalysisConfig(**analyzer_config)
    return SentimentAnalyzer(analysis_config)

def create_news_processor(config: dict = None):
    """
    Create news processor with configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        NewsProcessor instance
    """
    from .news_processor import NewsProcessor
    
    processor_config = DEFAULT_NEWS_CONFIG.copy()
    if config:
        processor_config.update(config)
    
    return NewsProcessor(processor_config)

def create_social_sentiment_analyzer(config: dict = None):
    """
    Create social sentiment analyzer with configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SocialSentimentAnalyzer instance
    """
    from .social_sentiment import SocialSentimentAnalyzer
    
    analyzer_config = DEFAULT_SENTIMENT_CONFIG['social_sentiment'].copy()
    if config:
        analyzer_config.update(config)
    
    return SocialSentimentAnalyzer(analyzer_config)

def create_market_scanner(config: dict = None):
    """
    Create market scanner with configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MarketScanner instance
    """
    from .market_scanner import MarketScanner, ScanCriteria
    
    scanner_config = DEFAULT_SCANNER_CONFIG.copy()
    if config:
        scanner_config.update(config)
    
    scan_criteria = ScanCriteria(**scanner_config)
    return MarketScanner(scan_criteria)

def create_full_analysis_suite(config: dict = None):
    """
    Create complete analysis suite with all components
    
    Args:
        config: Master configuration dictionary
        
    Returns:
        Dictionary with all analysis components
    """
    
    base_config = config or {}
    
    suite = {
        'technical': create_technical_analyzer(base_config.get('technical')),
        'sentiment': create_sentiment_analyzer(base_config.get('sentiment')),
        'news': create_news_processor(base_config.get('news')),
        'social': create_social_sentiment_analyzer(base_config.get('social')),
        'scanner': create_market_scanner(base_config.get('scanner'))
    }
    
    return suite

def validate_analysis_config(config: dict) -> Tuple[bool, List[str]]:
    """
    Validate analysis configuration
    
    Args:
        config: Configuration to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    
    errors = []
    
    # Validate technical indicators config
    if 'technical' in config:
        tech_config = config['technical']
        if 'sma_periods' in tech_config:
            if not all(isinstance(p, int) and p > 0 for p in tech_config['sma_periods']):
                errors.append("SMA periods must be positive integers")
    
    # Validate sentiment config
    if 'sentiment' in config:
        sent_config = config['sentiment']
        if 'lookback_hours' in sent_config:
            if not isinstance(sent_config['lookback_hours'], int) or sent_config['lookback_hours'] <= 0:
                errors.append("Lookback hours must be a positive integer")
    
    # Validate scanner config
    if 'scanner' in config:
        scan_config = config['scanner']
        if 'min_price' in scan_config and 'max_price' in scan_config:
            if scan_config['min_price'] >= scan_config['max_price']:
                errors.append("Min price must be less than max price")
    
    return len(errors) == 0, errors

# Version information
__version__ = "1.0.0"
__author__ = "NeuroCluster Elite Team"
__license__ = "MIT"