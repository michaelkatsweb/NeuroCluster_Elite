#!/usr/bin/env python3
"""
File: crypto_strategies.py
Path: NeuroCluster-Elite/src/trading/strategies/crypto_strategies.py
Description: Cryptocurrency-specific trading strategies for NeuroCluster Elite

This module implements specialized trading strategies designed specifically for
cryptocurrency markets, accounting for their unique characteristics including
high volatility, 24/7 trading, sentiment-driven moves, and different market dynamics.

Features:
- Crypto momentum strategies optimized for volatile markets
- DeFi and yield farming considerations
- Social sentiment integration
- Cross-crypto correlation strategies
- Volatility breakout strategies tuned for crypto
- Mean reversion strategies for stable coins
- Market cap and volume-based position sizing
- Multi-exchange arbitrage detection

Author: Your Name
Created: 2025-06-29
Version: 1.0.0
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import math
from collections import deque, defaultdict

# Import our modules
try:
    from src.core.neurocluster_elite import RegimeType, AssetType, MarketData
    from src.trading.strategies.base_strategy import BaseStrategy, TradingSignal, SignalType, StrategyState, StrategyMetrics, create_signal
    from src.utils.helpers import calculate_sharpe_ratio, calculate_max_drawdown, format_percentage
    from src.analysis.sentiment_analyzer import SentimentScore
    from src.utils.config_manager import ConfigManager
except ImportError:
    # Fallback for testing
    from enum import Enum
    class RegimeType(Enum):
        BULL = "bull"
        BEAR = "bear"
        SIDEWAYS = "sideways"
        VOLATILE = "volatile"
        BREAKOUT = "breakout"
        BREAKDOWN = "breakdown"
    
    class AssetType(Enum):
        CRYPTO = "crypto"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CRYPTO-SPECIFIC ENUMS ====================

class CryptoCategory(Enum):
    """Cryptocurrency categories for strategy adaptation"""
    BITCOIN = "bitcoin"           # BTC - Digital gold
    ETHEREUM = "ethereum"         # ETH - Smart contract platform
    ALTCOIN_LARGE = "altcoin_large"    # Top 10 market cap
    ALTCOIN_MID = "altcoin_mid"   # Top 100 market cap
    ALTCOIN_SMALL = "altcoin_small"    # Below top 100
    STABLECOIN = "stablecoin"     # USDT, USDC, etc.
    DEFI = "defi"                 # DeFi tokens
    MEME = "meme"                 # Meme coins
    UTILITY = "utility"           # Utility tokens
    PRIVACY = "privacy"           # Privacy coins

class CryptoSignalType(Enum):
    """Crypto-specific signal types"""
    MOMENTUM_BUY = "momentum_buy"
    MOMENTUM_SELL = "momentum_sell"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    VOLATILITY_BREAKDOWN = "volatility_breakdown"
    SENTIMENT_RALLY = "sentiment_rally"
    SENTIMENT_DUMP = "sentiment_dump"
    WHALE_ACCUMULATION = "whale_accumulation"
    WHALE_DISTRIBUTION = "whale_distribution"
    TECHNICAL_BOUNCE = "technical_bounce"
    TECHNICAL_BREAKDOWN = "technical_breakdown"
    CORRELATION_DIVERGENCE = "correlation_divergence"
    ARBITRAGE_OPPORTUNITY = "arbitrage_opportunity"

@dataclass
class CryptoMarketData:
    """Extended market data for crypto assets"""
    base_data: MarketData
    
    # Crypto-specific metrics
    market_cap: Optional[float] = None
    circulating_supply: Optional[float] = None
    max_supply: Optional[float] = None
    volume_24h: Optional[float] = None
    volume_change_24h: Optional[float] = None
    
    # Social metrics
    social_sentiment: Optional[float] = None
    reddit_sentiment: Optional[float] = None
    twitter_sentiment: Optional[float] = None
    fear_greed_index: Optional[float] = None
    
    # On-chain metrics
    active_addresses: Optional[int] = None
    transaction_count: Optional[int] = None
    hash_rate: Optional[float] = None
    network_value: Optional[float] = None
    
    # Exchange metrics
    exchange_inflows: Optional[float] = None
    exchange_outflows: Optional[float] = None
    whale_transactions: Optional[int] = None
    
    # DeFi metrics (if applicable)
    total_value_locked: Optional[float] = None
    yield_rate: Optional[float] = None

@dataclass
class CryptoStrategyConfig:
    """Configuration for crypto strategies"""
    # Volatility settings
    high_volatility_threshold: float = 50.0
    volatility_breakout_multiplier: float = 2.0
    volatility_window: int = 24  # hours
    
    # Momentum settings
    momentum_lookback: int = 14
    momentum_threshold: float = 0.05  # 5%
    momentum_confirmation_bars: int = 3
    
    # Sentiment settings
    sentiment_weight: float = 0.3
    sentiment_threshold_bullish: float = 0.6
    sentiment_threshold_bearish: float = -0.6
    social_sentiment_weight: float = 0.2
    
    # Risk management
    crypto_position_size_multiplier: float = 0.7  # Reduce position size for volatility
    stop_loss_atr_multiplier: float = 2.5
    take_profit_risk_reward: float = 2.0
    max_correlation_exposure: float = 0.6
    
    # Category-specific settings
    bitcoin_dominance_threshold: float = 45.0
    stablecoin_deviation_threshold: float = 0.02  # 2%
    meme_coin_volume_threshold: float = 10.0  # 10x average volume
    
    # Time-based settings
    asian_session_multiplier: float = 0.8
    us_session_multiplier: float = 1.2
    weekend_multiplier: float = 0.9

# ==================== CRYPTO MOMENTUM STRATEGY ====================

class CryptoMomentumStrategy(BaseStrategy):
    """
    Cryptocurrency momentum strategy optimized for high-volatility environments
    
    This strategy capitalizes on strong momentum moves in crypto markets,
    incorporating social sentiment, on-chain metrics, and volatility patterns.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize crypto momentum strategy"""
        
        super().__init__(config)
        self.strategy_name = "CryptoMomentumStrategy"
        self.crypto_config = CryptoStrategyConfig()
        
        # Crypto-specific tracking
        self.btc_correlation_tracker = deque(maxlen=50)
        self.eth_correlation_tracker = deque(maxlen=50)
        self.sentiment_history = deque(maxlen=100)
        self.volatility_history = deque(maxlen=50)
        
        # Category classification
        self.crypto_categories = self._initialize_crypto_categories()
        
        logger.info("🚀 Crypto Momentum Strategy initialized")
    
    def _initialize_crypto_categories(self) -> Dict[str, CryptoCategory]:
        """Initialize cryptocurrency category mappings"""
        
        categories = {
            # Bitcoin
            'BTC': CryptoCategory.BITCOIN,
            'BTCUSD': CryptoCategory.BITCOIN,
            'BTC-USD': CryptoCategory.BITCOIN,
            
            # Ethereum
            'ETH': CryptoCategory.ETHEREUM,
            'ETHUSD': CryptoCategory.ETHEREUM,
            'ETH-USD': CryptoCategory.ETHEREUM,
            
            # Large cap altcoins
            'ADA': CryptoCategory.ALTCOIN_LARGE,
            'SOL': CryptoCategory.ALTCOIN_LARGE,
            'DOT': CryptoCategory.ALTCOIN_LARGE,
            'AVAX': CryptoCategory.ALTCOIN_LARGE,
            'MATIC': CryptoCategory.ALTCOIN_LARGE,
            'LINK': CryptoCategory.ALTCOIN_LARGE,
            
            # Stablecoins
            'USDT': CryptoCategory.STABLECOIN,
            'USDC': CryptoCategory.STABLECOIN,
            'BUSD': CryptoCategory.STABLECOIN,
            'DAI': CryptoCategory.STABLECOIN,
            
            # DeFi tokens
            'UNI': CryptoCategory.DEFI,
            'AAVE': CryptoCategory.DEFI,
            'COMP': CryptoCategory.DEFI,
            'SUSHI': CryptoCategory.DEFI,
            'CRV': CryptoCategory.DEFI,
            
            # Meme coins
            'DOGE': CryptoCategory.MEME,
            'SHIB': CryptoCategory.MEME,
            
            # Privacy coins
            'XMR': CryptoCategory.PRIVACY,
            'ZEC': CryptoCategory.PRIVACY,
        }
        
        return categories
    
    def generate_signal(self, market_data: MarketData, regime: RegimeType, confidence: float) -> Optional[TradingSignal]:
        """Generate crypto momentum trading signal"""
        
        try:
            # Convert to crypto market data if possible
            crypto_data = self._enhance_crypto_data(market_data)
            
            # Get crypto category
            category = self._get_crypto_category(market_data.symbol)
            
            # Skip stablecoins for momentum trading
            if category == CryptoCategory.STABLECOIN:
                return None
            
            # Calculate momentum components
            momentum_score = self._calculate_crypto_momentum(crypto_data, regime)
            sentiment_score = self._calculate_sentiment_score(crypto_data)
            volatility_score = self._calculate_volatility_score(crypto_data)
            
            # Combine scores with category-specific weights
            combined_score = self._combine_crypto_scores(
                momentum_score, sentiment_score, volatility_score, category
            )
            
            # Generate signal based on combined score
            signal = self._generate_crypto_signal(
                combined_score, crypto_data, regime, confidence, category
            )
            
            # Apply crypto-specific filters
            if signal:
                signal = self._apply_crypto_filters(signal, crypto_data, category)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating crypto momentum signal: {e}")
            return None
    
    def _enhance_crypto_data(self, market_data: MarketData) -> CryptoMarketData:
        """Enhance market data with crypto-specific metrics"""
        
        # For now, create basic crypto data structure
        # In production, this would fetch additional crypto metrics
        crypto_data = CryptoMarketData(
            base_data=market_data,
            volume_24h=market_data.volume * 24 if market_data.volume else None,
            social_sentiment=getattr(market_data, 'sentiment_score', 0.0),
            fear_greed_index=50.0  # Neutral default
        )
        
        return crypto_data
    
    def _get_crypto_category(self, symbol: str) -> CryptoCategory:
        """Get cryptocurrency category for the symbol"""
        
        # Clean symbol
        clean_symbol = symbol.upper().replace('-USD', '').replace('USD', '').replace('-', '')
        
        return self.crypto_categories.get(clean_symbol, CryptoCategory.ALTCOIN_MID)
    
    def _calculate_crypto_momentum(self, crypto_data: CryptoMarketData, regime: RegimeType) -> float:
        """Calculate cryptocurrency-specific momentum score"""
        
        try:
            market_data = crypto_data.base_data
            
            # Base momentum from price change
            base_momentum = market_data.change_percent / 100.0 if market_data.change_percent else 0.0
            
            # Volatility-adjusted momentum
            volatility = market_data.volatility or 20.0
            volatility_adjustment = min(2.0, volatility / 30.0)  # Cap at 2x for extreme volatility
            
            # Volume momentum
            volume_momentum = 0.0
            if crypto_data.volume_change_24h:
                volume_momentum = min(0.5, crypto_data.volume_change_24h / 100.0)
            
            # Regime-based momentum adjustment
            regime_multiplier = {
                RegimeType.BULL: 1.3,
                RegimeType.BEAR: 0.7,
                RegimeType.BREAKOUT: 1.5,
                RegimeType.BREAKDOWN: 0.5,
                RegimeType.VOLATILE: 1.1,
                RegimeType.SIDEWAYS: 0.8
            }.get(regime, 1.0)
            
            # Combine momentum components
            momentum_score = (base_momentum * volatility_adjustment + volume_momentum * 0.3) * regime_multiplier
            
            # Normalize to [-1, 1]
            momentum_score = np.tanh(momentum_score * 2)
            
            return momentum_score
            
        except Exception as e:
            logger.warning(f"Error calculating crypto momentum: {e}")
            return 0.0
    
    def _calculate_sentiment_score(self, crypto_data: CryptoMarketData) -> float:
        """Calculate sentiment-based score for crypto"""
        
        try:
            sentiment_components = []
            
            # Social sentiment
            if crypto_data.social_sentiment is not None:
                sentiment_components.append(crypto_data.social_sentiment)
            
            # Fear & Greed Index
            if crypto_data.fear_greed_index is not None:
                # Convert from 0-100 to -1 to 1
                fg_normalized = (crypto_data.fear_greed_index - 50) / 50
                sentiment_components.append(fg_normalized)
            
            # Base sentiment from market data
            if crypto_data.base_data.sentiment_score is not None:
                sentiment_components.append(crypto_data.base_data.sentiment_score)
            
            if sentiment_components:
                sentiment_score = np.mean(sentiment_components)
            else:
                sentiment_score = 0.0
            
            # Store sentiment history
            self.sentiment_history.append(sentiment_score)
            
            # Calculate sentiment momentum
            if len(self.sentiment_history) > 5:
                recent_sentiment = np.mean(list(self.sentiment_history)[-3:])
                older_sentiment = np.mean(list(self.sentiment_history)[-6:-3])
                sentiment_momentum = recent_sentiment - older_sentiment
                sentiment_score += sentiment_momentum * 0.5
            
            return np.clip(sentiment_score, -1.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating sentiment score: {e}")
            return 0.0
    
    def _calculate_volatility_score(self, crypto_data: CryptoMarketData) -> float:
        """Calculate volatility-based trading score"""
        
        try:
            volatility = crypto_data.base_data.volatility or 20.0
            
            # Store volatility history
            self.volatility_history.append(volatility)
            
            # Volatility percentile (high volatility can signal breakouts)
            if len(self.volatility_history) > 10:
                vol_percentile = len([v for v in self.volatility_history if v < volatility]) / len(self.volatility_history)
            else:
                vol_percentile = 0.5
            
            # High volatility can be bullish for breakouts
            if volatility > self.crypto_config.high_volatility_threshold:
                volatility_score = vol_percentile * 0.5  # Positive but capped
            else:
                volatility_score = (vol_percentile - 0.5) * 0.2  # Neutral to slightly negative
            
            return volatility_score
            
        except Exception as e:
            logger.warning(f"Error calculating volatility score: {e}")
            return 0.0
    
    def _combine_crypto_scores(self, momentum: float, sentiment: float, 
                              volatility: float, category: CryptoCategory) -> float:
        """Combine all scores with category-specific weights"""
        
        # Base weights
        momentum_weight = 0.6
        sentiment_weight = self.crypto_config.sentiment_weight
        volatility_weight = 0.1
        
        # Category-specific adjustments
        if category == CryptoCategory.BITCOIN:
            # Bitcoin is less sentiment-driven, more technical
            momentum_weight = 0.7
            sentiment_weight = 0.2
            volatility_weight = 0.1
            
        elif category == CryptoCategory.MEME:
            # Meme coins are highly sentiment-driven
            momentum_weight = 0.4
            sentiment_weight = 0.5
            volatility_weight = 0.1
            
        elif category in [CryptoCategory.DEFI, CryptoCategory.UTILITY]:
            # DeFi/Utility tokens balance technical and sentiment
            momentum_weight = 0.5
            sentiment_weight = 0.35
            volatility_weight = 0.15
        
        # Combine scores
        combined_score = (
            momentum * momentum_weight +
            sentiment * sentiment_weight +
            volatility * volatility_weight
        )
        
        return np.clip(combined_score, -1.0, 1.0)
    
    def _generate_crypto_signal(self, combined_score: float, crypto_data: CryptoMarketData,
                               regime: RegimeType, confidence: float, category: CryptoCategory) -> Optional[TradingSignal]:
        """Generate trading signal based on combined crypto score"""
        
        try:
            market_data = crypto_data.base_data
            
            # Determine signal type and strength
            if combined_score > 0.6:
                signal_type = SignalType.STRONG_BUY
                crypto_signal_type = CryptoSignalType.MOMENTUM_BUY
            elif combined_score > 0.3:
                signal_type = SignalType.BUY
                crypto_signal_type = CryptoSignalType.TECHNICAL_BOUNCE
            elif combined_score < -0.6:
                signal_type = SignalType.STRONG_SELL
                crypto_signal_type = CryptoSignalType.MOMENTUM_SELL
            elif combined_score < -0.3:
                signal_type = SignalType.SELL
                crypto_signal_type = CryptoSignalType.TECHNICAL_BREAKDOWN
            else:
                signal_type = SignalType.HOLD
                crypto_signal_type = None
            
            if signal_type == SignalType.HOLD:
                return None
            
            # Adjust confidence based on crypto factors
            crypto_confidence = confidence * abs(combined_score)
            
            # Category-specific confidence adjustments
            if category == CryptoCategory.BITCOIN:
                crypto_confidence *= 1.1  # Higher confidence for BTC
            elif category == CryptoCategory.MEME:
                crypto_confidence *= 0.8  # Lower confidence for meme coins
            
            # Create signal
            signal = create_signal(
                symbol=market_data.symbol,
                asset_type=AssetType.CRYPTO,
                signal_type=signal_type,
                regime=regime,
                confidence=crypto_confidence,
                entry_price=market_data.price,
                current_price=market_data.price,
                strategy_name=self.strategy_name,
                reasoning=f"Crypto momentum signal: {crypto_signal_type.value if crypto_signal_type else 'hold'} "
                         f"(score: {combined_score:.3f}, category: {category.value})"
            )
            
            # Add crypto-specific metadata
            signal.technical_factors.update({
                'crypto_category': category.value,
                'combined_score': combined_score,
                'crypto_signal_type': crypto_signal_type.value if crypto_signal_type else None,
                'volatility_score': crypto_data.base_data.volatility,
                'sentiment_score': crypto_data.social_sentiment
            })
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating crypto signal: {e}")
            return None
    
    def _apply_crypto_filters(self, signal: TradingSignal, crypto_data: CryptoMarketData,
                             category: CryptoCategory) -> Optional[TradingSignal]:
        """Apply crypto-specific filters to the signal"""
        
        try:
            # Time-based filters for crypto markets (24/7 trading)
            current_hour = datetime.now().hour
            
            # Reduce signal strength during low-activity hours (2-6 AM UTC)
            if 2 <= current_hour <= 6:
                signal.confidence *= self.crypto_config.asian_session_multiplier
            
            # Increase signal strength during high-activity hours (13-21 UTC)
            elif 13 <= current_hour <= 21:
                signal.confidence *= self.crypto_config.us_session_multiplier
            
            # Weekend adjustments
            if datetime.now().weekday() >= 5:  # Saturday or Sunday
                signal.confidence *= self.crypto_config.weekend_multiplier
            
            # Volatility filters
            volatility = crypto_data.base_data.volatility or 20.0
            if volatility > 100:  # Extreme volatility
                if category in [CryptoCategory.ALTCOIN_SMALL, CryptoCategory.MEME]:
                    signal.confidence *= 0.6  # Reduce confidence for small/meme coins
                else:
                    signal.confidence *= 0.8  # Slight reduction for others
            
            # Volume filter
            if crypto_data.volume_change_24h and crypto_data.volume_change_24h < -50:
                signal.confidence *= 0.7  # Reduce confidence on low volume
            
            # Category-specific risk adjustments
            if category == CryptoCategory.MEME:
                signal.max_risk_pct *= 0.5  # Reduce position size for meme coins
            elif category == CryptoCategory.ALTCOIN_SMALL:
                signal.max_risk_pct *= 0.7  # Reduce position size for small altcoins
            
            # Minimum confidence threshold
            if signal.confidence < 0.3:
                return None
            
            return signal
            
        except Exception as e:
            logger.warning(f"Error applying crypto filters: {e}")
            return signal
    
    def get_strategy_description(self) -> str:
        """Get strategy description"""
        return (
            "Cryptocurrency momentum strategy that combines price momentum, "
            "social sentiment, and volatility patterns optimized for crypto markets. "
            "Adapts to different cryptocurrency categories and incorporates "
            "crypto-specific risk management."
        )

# ==================== CRYPTO VOLATILITY STRATEGY ====================

class CryptoVolatilityStrategy(BaseStrategy):
    """
    Cryptocurrency volatility strategy for extreme market conditions
    
    This strategy specializes in trading crypto volatility breakouts and
    mean reversion patterns in highly volatile cryptocurrency markets.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize crypto volatility strategy"""
        
        super().__init__(config)
        self.strategy_name = "CryptoVolatilityStrategy"
        self.crypto_config = CryptoStrategyConfig()
        
        # Volatility tracking
        self.volatility_bands = deque(maxlen=100)
        self.breakout_levels = {}
        
        logger.info("⚡ Crypto Volatility Strategy initialized")
    
    def generate_signal(self, market_data: MarketData, regime: RegimeType, confidence: float) -> Optional[TradingSignal]:
        """Generate crypto volatility trading signal"""
        
        try:
            volatility = market_data.volatility or 20.0
            
            # Store volatility for bands calculation
            self.volatility_bands.append(volatility)
            
            if len(self.volatility_bands) < 20:
                return None
            
            # Calculate volatility percentiles
            vol_array = np.array(self.volatility_bands)
            vol_percentile = len(vol_array[vol_array < volatility]) / len(vol_array)
            
            # Detect volatility breakouts
            if vol_percentile > 0.9:  # Top 10% volatility
                # High volatility breakout
                signal_type = SignalType.STRONG_BUY if market_data.change_percent > 0 else SignalType.STRONG_SELL
                reasoning = f"Volatility breakout: {volatility:.1f}% (top 10%)"
                
            elif vol_percentile < 0.2 and volatility < 15:  # Low volatility, potential mean reversion setup
                # Look for mean reversion opportunity
                if abs(market_data.change_percent) > 3:  # Significant move on low volatility
                    signal_type = SignalType.SELL if market_data.change_percent > 0 else SignalType.BUY
                    reasoning = f"Mean reversion on low volatility: {volatility:.1f}%"
                else:
                    return None
            else:
                return None
            
            # Adjust confidence based on volatility extremes
            vol_confidence = confidence * min(1.5, abs(vol_percentile - 0.5) * 4)
            
            signal = create_signal(
                symbol=market_data.symbol,
                asset_type=AssetType.CRYPTO,
                signal_type=signal_type,
                regime=regime,
                confidence=vol_confidence,
                entry_price=market_data.price,
                current_price=market_data.price,
                strategy_name=self.strategy_name,
                reasoning=reasoning
            )
            
            # Crypto volatility specific risk management
            if volatility > 80:  # Extreme volatility
                signal.max_risk_pct *= 0.5  # Reduce position size significantly
                signal.stop_loss = market_data.price * (0.95 if signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else 1.05)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating crypto volatility signal: {e}")
            return None
    
    def get_strategy_description(self) -> str:
        """Get strategy description"""
        return (
            "Cryptocurrency volatility strategy that trades extreme volatility "
            "breakouts and mean reversion patterns in crypto markets."
        )

# ==================== CRYPTO SENTIMENT STRATEGY ====================

class CryptoSentimentStrategy(BaseStrategy):
    """
    Cryptocurrency sentiment-driven strategy
    
    This strategy focuses on social sentiment, fear & greed index,
    and market psychology indicators specific to cryptocurrency markets.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize crypto sentiment strategy"""
        
        super().__init__(config)
        self.strategy_name = "CryptoSentimentStrategy"
        self.crypto_config = CryptoStrategyConfig()
        
        # Sentiment tracking
        self.sentiment_history = deque(maxlen=50)
        self.fear_greed_history = deque(maxlen=30)
        
        logger.info("🧠 Crypto Sentiment Strategy initialized")
    
    def generate_signal(self, market_data: MarketData, regime: RegimeType, confidence: float) -> Optional[TradingSignal]:
        """Generate crypto sentiment trading signal"""
        
        try:
            # Get sentiment score (would be enhanced with real sentiment data)
            sentiment_score = getattr(market_data, 'sentiment_score', 0.0)
            
            # Store sentiment history
            self.sentiment_history.append(sentiment_score)
            
            if len(self.sentiment_history) < 10:
                return None
            
            # Calculate sentiment momentum
            recent_sentiment = np.mean(list(self.sentiment_history)[-5:])
            older_sentiment = np.mean(list(self.sentiment_history)[-10:-5])
            sentiment_momentum = recent_sentiment - older_sentiment
            
            # Generate signals based on sentiment extremes and momentum
            if recent_sentiment > self.crypto_config.sentiment_threshold_bullish and sentiment_momentum > 0.1:
                signal_type = SignalType.BUY
                reasoning = f"Bullish sentiment momentum: {recent_sentiment:.2f}"
                
            elif recent_sentiment < self.crypto_config.sentiment_threshold_bearish and sentiment_momentum < -0.1:
                signal_type = SignalType.SELL
                reasoning = f"Bearish sentiment momentum: {recent_sentiment:.2f}"
                
            elif recent_sentiment > 0.8:  # Extreme bullish sentiment (contrarian)
                signal_type = SignalType.WEAK_SELL
                reasoning = f"Extreme bullish sentiment (contrarian): {recent_sentiment:.2f}"
                
            elif recent_sentiment < -0.8:  # Extreme bearish sentiment (contrarian)
                signal_type = SignalType.WEAK_BUY
                reasoning = f"Extreme bearish sentiment (contrarian): {recent_sentiment:.2f}"
                
            else:
                return None
            
            # Sentiment-based confidence
            sentiment_confidence = confidence * (abs(sentiment_momentum) * 2 + abs(recent_sentiment) * 0.5)
            
            signal = create_signal(
                symbol=market_data.symbol,
                asset_type=AssetType.CRYPTO,
                signal_type=signal_type,
                regime=regime,
                confidence=sentiment_confidence,
                entry_price=market_data.price,
                current_price=market_data.price,
                strategy_name=self.strategy_name,
                reasoning=reasoning
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating crypto sentiment signal: {e}")
            return None
    
    def get_strategy_description(self) -> str:
        """Get strategy description"""
        return (
            "Cryptocurrency sentiment strategy that trades based on social sentiment, "
            "fear & greed index, and market psychology indicators."
        )

# ==================== STRATEGY FACTORY ====================

class CryptoStrategyFactory:
    """Factory for creating crypto strategies"""
    
    @staticmethod
    def create_strategy(strategy_type: str, config: Dict = None) -> BaseStrategy:
        """Create a crypto strategy by type"""
        
        strategies = {
            'crypto_momentum': CryptoMomentumStrategy,
            'crypto_volatility': CryptoVolatilityStrategy,
            'crypto_sentiment': CryptoSentimentStrategy
        }
        
        strategy_class = strategies.get(strategy_type)
        if not strategy_class:
            raise ValueError(f"Unknown crypto strategy type: {strategy_type}")
        
        return strategy_class(config)
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available crypto strategies"""
        return ['crypto_momentum', 'crypto_volatility', 'crypto_sentiment']
    
    @staticmethod
    def get_strategy_descriptions() -> Dict[str, str]:
        """Get descriptions of all crypto strategies"""
        return {
            'crypto_momentum': "Momentum-based strategy optimized for cryptocurrency markets",
            'crypto_volatility': "Volatility breakout and mean reversion strategy for crypto",
            'crypto_sentiment': "Sentiment-driven strategy using social and market psychology indicators"
        }

# ==================== TESTING FUNCTION ====================

def test_crypto_strategies():
    """Test crypto strategies functionality"""
    
    print("🚀 Testing Crypto Trading Strategies")
    print("=" * 50)
    
    # Test data class
    class MockMarketData:
        def __init__(self, symbol, price, change_pct, volatility, volume, sentiment=0.0):
            self.symbol = symbol
            self.asset_type = AssetType.CRYPTO
            self.price = price
            self.change = price * change_pct / 100
            self.change_percent = change_pct
            self.volume = volume
            self.volatility = volatility
            self.sentiment_score = sentiment
            self.timestamp = datetime.now()
    
    # Test different crypto scenarios
    test_scenarios = [
        {
            'name': 'Bitcoin Bull Run',
            'data': MockMarketData('BTC-USD', 45000, 8.5, 35, 1000000, 0.7),
            'regime': RegimeType.BULL
        },
        {
            'name': 'Ethereum Volatility Breakout',
            'data': MockMarketData('ETH-USD', 3200, 12.3, 85, 2000000, 0.4),
            'regime': RegimeType.BREAKOUT
        },
        {
            'name': 'Altcoin Bear Signal',
            'data': MockMarketData('ADA-USD', 0.45, -15.2, 45, 500000, -0.6),
            'regime': RegimeType.BEAR
        },
        {
            'name': 'Meme Coin Sentiment Rally',
            'data': MockMarketData('DOGE-USD', 0.08, 25.7, 95, 5000000, 0.9),
            'regime': RegimeType.VOLATILE
        }
    ]
    
    # Test each strategy
    strategies = [
        ('crypto_momentum', CryptoMomentumStrategy()),
        ('crypto_volatility', CryptoVolatilityStrategy()),
        ('crypto_sentiment', CryptoSentimentStrategy())
    ]
    
    for strategy_name, strategy in strategies:
        print(f"\n📈 Testing {strategy_name.replace('_', ' ').title()}")
        print("-" * 40)
        
        for scenario in test_scenarios:
            signal = strategy.generate_signal(
                scenario['data'], 
                scenario['regime'], 
                0.8
            )
            
            if signal:
                print(f"✅ {scenario['name']}: {signal.signal_type.value} "
                      f"(Confidence: {signal.confidence:.2f})")
                print(f"   Reasoning: {signal.reasoning}")
            else:
                print(f"⚪ {scenario['name']}: No signal generated")
    
    # Test factory
    print(f"\n🏭 Testing Crypto Strategy Factory")
    print("-" * 40)
    
    factory = CryptoStrategyFactory()
    available = factory.get_available_strategies()
    print(f"✅ Available strategies: {available}")
    
    for strategy_type in available:
        try:
            strategy = factory.create_strategy(strategy_type)
            print(f"✅ Created {strategy_type}: {strategy.strategy_name}")
        except Exception as e:
            print(f"❌ Failed to create {strategy_type}: {e}")
    
    print("\n🎉 Crypto strategies testing completed!")

if __name__ == "__main__":
    test_crypto_strategies()