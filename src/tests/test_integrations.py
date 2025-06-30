#!/usr/bin/env python3
"""
File: test_integrations.py
Path: NeuroCluster-Elite/src/tests/test_integrations.py
Description: Comprehensive tests for external integrations

This module contains comprehensive tests for all external integrations including
broker connections, exchange APIs, notification systems, and third-party services.

Features tested:
- Broker integrations (Interactive Brokers, TD Ameritrade, Alpaca)
- Cryptocurrency exchange integrations (Binance, Coinbase, Kraken)
- Notification systems (Email, Discord, Telegram, Mobile Push)
- Paper trading simulation
- API rate limiting and error handling
- Authentication and security
- Real-time data feeds
- Order execution and management

Author: Michael Katsaros
Created: 2025-06-30
Version: 1.0.0
License: MIT
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import json
import time
import aiohttp
import websockets
from typing import Dict, List, Any, Optional
import ssl
import certifi

# Import modules to test
try:
    # Broker integrations
    from src.integrations.brokers.interactive_brokers import InteractiveBrokersAPI
    from src.integrations.brokers.td_ameritrade import TDAmeritrade
    from src.integrations.brokers.alpaca import AlpacaTrading
    from src.integrations.brokers.paper_trading import PaperTradingBroker
    
    # Exchange integrations
    from src.integrations.exchanges.binance import BinanceExchange
    from src.integrations.exchanges.coinbase import CoinbaseExchange
    from src.integrations.exchanges.kraken import KrakenExchange
    
    # Notification systems
    from src.integrations.notifications.email_alerts import EmailAlertSystem
    from src.integrations.notifications.discord_bot import DiscordNotifier
    from src.integrations.notifications.telegram_bot import TelegramNotifier
    from src.integrations.notifications.mobile_push import MobilePushNotifier
    from src.integrations.notifications.alert_system import UnifiedAlertSystem
    
    # Core types
    from src.trading.trading_engine import OrderType, OrderStatus, PositionSide
    from src.core.neurocluster_elite import MarketData, AssetType
    
except ImportError as e:
    pytest.skip(f"Could not import integration modules: {e}", allow_module_level=True)

# Test configuration
TEST_CONFIG = {
    'paper_trading': True,
    'api_timeout': 10,
    'max_retries': 3,
    'rate_limit_calls': 100,
    'rate_limit_period': 60
}

# Mock API responses
MOCK_STOCK_QUOTE = {
    'symbol': 'AAPL',
    'last': 150.25,
    'bid': 150.20,
    'ask': 150.30,
    'volume': 45231000,
    'change': 2.75,
    'changePercent': 1.86
}

MOCK_CRYPTO_TICKER = {
    'symbol': 'BTCUSDT',
    'price': '45250.75',
    'priceChange': '1250.25',
    'priceChangePercent': '2.84',
    'volume': '28500.12345678'
}

MOCK_ORDER_RESPONSE = {
    'orderId': '12345',
    'symbol': 'AAPL',
    'side': 'BUY',
    'quantity': '100',
    'price': '150.25',
    'status': 'FILLED',
    'timestamp': datetime.now().isoformat()
}

# ==================== FIXTURES ====================

@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp session for API calls"""
    session = Mock()
    
    async def mock_get(*args, **kwargs):
        response = Mock()
        response.status = 200
        response.json = AsyncMock(return_value=MOCK_STOCK_QUOTE)
        response.text = AsyncMock(return_value=json.dumps(MOCK_STOCK_QUOTE))
        return response
    
    async def mock_post(*args, **kwargs):
        response = Mock()
        response.status = 200
        response.json = AsyncMock(return_value=MOCK_ORDER_RESPONSE)
        response.text = AsyncMock(return_value=json.dumps(MOCK_ORDER_RESPONSE))
        return response
    
    session.get = AsyncMock(side_effect=mock_get)
    session.post = AsyncMock(side_effect=mock_post)
    session.put = AsyncMock(side_effect=mock_post)
    session.delete = AsyncMock(side_effect=mock_post)
    
    return session

@pytest.fixture
def mock_websocket():
    """Mock websocket connection"""
    ws = Mock()
    
    async def mock_recv():
        return json.dumps({
            'stream': 'btcusdt@ticker',
            'data': MOCK_CRYPTO_TICKER
        })
    
    async def mock_send(message):
        pass
    
    ws.recv = AsyncMock(side_effect=mock_recv)
    ws.send = AsyncMock(side_effect=mock_send)
    ws.close = AsyncMock()
    
    return ws

@pytest.fixture
def paper_trading_broker():
    """Create paper trading broker for testing"""
    return PaperTradingBroker(config=TEST_CONFIG)

@pytest.fixture
def email_alert_system():
    """Create email alert system for testing"""
    config = {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'test@example.com',
        'password': 'test_password',
        'from_email': 'test@example.com'
    }
    return EmailAlertSystem(config=config)

@pytest.fixture
def unified_alert_system():
    """Create unified alert system for testing"""
    return UnifiedAlertSystem(config=TEST_CONFIG)

# ==================== PAPER TRADING TESTS ====================

class TestPaperTradingBroker:
    """Test paper trading broker functionality"""
    
    def test_initialization(self, paper_trading_broker):
        """Test paper trading broker initialization"""
        assert paper_trading_broker.initial_capital == 100000.0
        assert paper_trading_broker.cash_balance == 100000.0
        assert len(paper_trading_broker.positions) == 0
        assert len(paper_trading_broker.orders) == 0
        assert len(paper_trading_broker.trade_history) == 0
    
    @pytest.mark.asyncio
    async def test_market_order_execution(self, paper_trading_broker):
        """Test market order execution"""
        order_request = {
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 100,
            'order_type': 'MARKET',
            'price': None
        }
        
        # Mock market data
        with patch.object(paper_trading_broker, '_get_current_price') as mock_price:
            mock_price.return_value = 150.25
            
            order = await paper_trading_broker.place_order(**order_request)
            
            assert order['status'] == 'FILLED'
            assert order['symbol'] == 'AAPL'
            assert order['quantity'] == 100
            assert order['fill_price'] == 150.25
            
            # Check position created
            assert 'AAPL' in paper_trading_broker.positions
            position = paper_trading_broker.positions['AAPL']
            assert position['quantity'] == 100
            assert position['avg_cost'] == 150.25
    
    @pytest.mark.asyncio
    async def test_limit_order_execution(self, paper_trading_broker):
        """Test limit order execution"""
        order_request = {
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 100,
            'order_type': 'LIMIT',
            'price': 149.50
        }
        
        # Mock current price above limit
        with patch.object(paper_trading_broker, '_get_current_price') as mock_price:
            mock_price.return_value = 150.25
            
            order = await paper_trading_broker.place_order(**order_request)
            
            # Should remain pending
            assert order['status'] == 'PENDING'
            
            # Simulate price dropping to limit
            mock_price.return_value = 149.25
            
            # Process pending orders
            await paper_trading_broker._process_pending_orders()
            
            # Order should now be filled
            filled_order = paper_trading_broker.orders[order['order_id']]
            assert filled_order['status'] == 'FILLED'
    
    @pytest.mark.asyncio
    async def test_stop_loss_execution(self, paper_trading_broker):
        """Test stop loss order execution"""
        # First, create a position
        await paper_trading_broker.place_order(
            symbol='AAPL',
            side='BUY',
            quantity=100,
            order_type='MARKET',
            price=None
        )
        
        # Place stop loss order
        stop_order = await paper_trading_broker.place_order(
            symbol='AAPL',
            side='SELL',
            quantity=100,
            order_type='STOP_LOSS',
            price=142.50
        )
        
        assert stop_order['status'] == 'PENDING'
        
        # Simulate price dropping to stop loss
        with patch.object(paper_trading_broker, '_get_current_price') as mock_price:
            mock_price.return_value = 142.00
            
            await paper_trading_broker._process_pending_orders()
            
            # Stop loss should trigger
            filled_order = paper_trading_broker.orders[stop_order['order_id']]
            assert filled_order['status'] == 'FILLED'
            
            # Position should be closed
            assert paper_trading_broker.positions['AAPL']['quantity'] == 0
    
    def test_portfolio_valuation(self, paper_trading_broker):
        """Test portfolio valuation calculations"""
        # Add mock positions
        paper_trading_broker.positions = {
            'AAPL': {
                'quantity': 100,
                'avg_cost': 150.00,
                'current_price': 155.00
            },
            'GOOGL': {
                'quantity': 50,
                'avg_cost': 2800.00,
                'current_price': 2850.00
            }
        }
        
        portfolio_value = paper_trading_broker.get_portfolio_value()
        
        # AAPL: 100 * 155 = 15,500
        # GOOGL: 50 * 2,850 = 142,500
        # Total positions: 158,000
        # Plus remaining cash
        expected_total = 158000 + paper_trading_broker.cash_balance
        
        assert abs(portfolio_value - expected_total) < 1.0
    
    def test_performance_metrics(self, paper_trading_broker):
        """Test performance metrics calculation"""
        # Add some trade history
        paper_trading_broker.trade_history = [
            {
                'symbol': 'AAPL',
                'quantity': 100,
                'entry_price': 150.00,
                'exit_price': 155.00,
                'pnl': 500.00,
                'timestamp': datetime.now()
            },
            {
                'symbol': 'GOOGL',
                'quantity': 50,
                'entry_price': 2800.00,
                'exit_price': 2750.00,
                'pnl': -2500.00,
                'timestamp': datetime.now()
            }
        ]
        
        metrics = paper_trading_broker.get_performance_metrics()
        
        assert 'total_pnl' in metrics
        assert 'win_rate' in metrics
        assert 'num_trades' in metrics
        
        assert metrics['total_pnl'] == -2000.0  # 500 - 2500
        assert metrics['win_rate'] == 0.5  # 1 out of 2 winning trades
        assert metrics['num_trades'] == 2

# ==================== BROKER INTEGRATION TESTS ====================

class TestBrokerIntegrations:
    """Test broker API integrations"""
    
    @pytest.mark.asyncio
    async def test_alpaca_integration(self, mock_aiohttp_session):
        """Test Alpaca API integration"""
        config = {
            'api_key': 'test_key',
            'secret_key': 'test_secret',
            'base_url': 'https://paper-api.alpaca.markets',
            'paper_trading': True
        }
        
        alpaca = AlpacaTrading(config=config)
        alpaca.session = mock_aiohttp_session
        
        # Test account info
        account_info = await alpaca.get_account_info()
        assert account_info is not None
        
        # Test market data
        quote = await alpaca.get_quote('AAPL')
        assert quote['symbol'] == 'AAPL'
        assert 'last' in quote
        
        # Test order placement
        order = await alpaca.place_order(
            symbol='AAPL',
            qty=100,
            side='buy',
            type='market'
        )
        assert order is not None
    
    @pytest.mark.asyncio
    async def test_td_ameritrade_integration(self, mock_aiohttp_session):
        """Test TD Ameritrade API integration"""
        config = {
            'client_id': 'test_client_id',
            'redirect_uri': 'https://localhost',
            'refresh_token': 'test_refresh_token'
        }
        
        td = TDAmeritrade(config=config)
        td.session = mock_aiohttp_session
        
        # Test authentication
        with patch.object(td, '_refresh_access_token') as mock_auth:
            mock_auth.return_value = 'test_access_token'
            
            await td.authenticate()
            assert td.access_token == 'test_access_token'
        
        # Test market data
        quote = await td.get_quote('AAPL')
        assert quote is not None
        
        # Test order placement (mock)
        order = await td.place_order(
            symbol='AAPL',
            instruction='BUY',
            quantity=100,
            type='MARKET'
        )
        assert order is not None
    
    @pytest.mark.asyncio
    async def test_interactive_brokers_integration(self):
        """Test Interactive Brokers API integration"""
        config = {
            'host': 'localhost',
            'port': 7497,
            'client_id': 1
        }
        
        ib = InteractiveBrokersAPI(config=config)
        
        # Mock IB connection
        with patch.object(ib, 'connect') as mock_connect:
            with patch.object(ib, 'isConnected') as mock_connected:
                mock_connect.return_value = True
                mock_connected.return_value = True
                
                # Test connection
                connected = await ib.connect_async()
                assert connected
                
                # Test contract creation
                contract = ib.create_stock_contract('AAPL')
                assert contract.symbol == 'AAPL'
                assert contract.secType == 'STK'
    
    def test_broker_error_handling(self, mock_aiohttp_session):
        """Test broker API error handling"""
        # Mock API error
        async def mock_error_response(*args, **kwargs):
            response = Mock()
            response.status = 400
            response.json = AsyncMock(return_value={'error': 'Invalid request'})
            response.text = AsyncMock(return_value='Bad Request')
            return response
        
        mock_aiohttp_session.get = AsyncMock(side_effect=mock_error_response)
        
        config = {'api_key': 'test_key', 'secret_key': 'test_secret'}
        alpaca = AlpacaTrading(config=config)
        alpaca.session = mock_aiohttp_session
        
        # Should handle error gracefully
        with pytest.raises(Exception):
            asyncio.run(alpaca.get_quote('INVALID'))

# ==================== EXCHANGE INTEGRATION TESTS ====================

class TestExchangeIntegrations:
    """Test cryptocurrency exchange integrations"""
    
    @pytest.mark.asyncio
    async def test_binance_integration(self, mock_aiohttp_session):
        """Test Binance API integration"""
        config = {
            'api_key': 'test_api_key',
            'secret_key': 'test_secret_key',
            'testnet': True
        }
        
        binance = BinanceExchange(config=config)
        binance.session = mock_aiohttp_session
        
        # Test ticker data
        ticker = await binance.get_ticker('BTCUSDT')
        assert ticker['symbol'] == 'BTCUSDT'
        
        # Test order book
        orderbook = await binance.get_orderbook('BTCUSDT', limit=100)
        assert orderbook is not None
        
        # Test order placement (testnet)
        order = await binance.place_order(
            symbol='BTCUSDT',
            side='BUY',
            type='MARKET',
            quantity=0.001
        )
        assert order is not None
    
    @pytest.mark.asyncio
    async def test_coinbase_integration(self, mock_aiohttp_session):
        """Test Coinbase Pro API integration"""
        config = {
            'api_key': 'test_api_key',
            'secret_key': 'test_secret_key',
            'passphrase': 'test_passphrase',
            'sandbox': True
        }
        
        coinbase = CoinbaseExchange(config=config)
        coinbase.session = mock_aiohttp_session
        
        # Test ticker data
        ticker = await coinbase.get_ticker('BTC-USD')
        assert ticker is not None
        
        # Test account balances
        balances = await coinbase.get_accounts()
        assert balances is not None
        
        # Test order placement (sandbox)
        order = await coinbase.place_order(
            product_id='BTC-USD',
            side='buy',
            type='market',
            funds='100.00'
        )
        assert order is not None
    
    @pytest.mark.asyncio
    async def test_kraken_integration(self, mock_aiohttp_session):
        """Test Kraken API integration"""
        config = {
            'api_key': 'test_api_key',
            'private_key': 'test_private_key'
        }
        
        kraken = KrakenExchange(config=config)
        kraken.session = mock_aiohttp_session
        
        # Test ticker data
        ticker = await kraken.get_ticker('XBTUSD')
        assert ticker is not None
        
        # Test asset info
        assets = await kraken.get_asset_info()
        assert assets is not None
        
        # Test balance (would require authentication)
        with patch.object(kraken, '_sign_request') as mock_sign:
            mock_sign.return_value = 'test_signature'
            
            balance = await kraken.get_balance()
            assert balance is not None
    
    @pytest.mark.asyncio
    async def test_websocket_integration(self, mock_websocket):
        """Test exchange WebSocket integration"""
        binance = BinanceExchange(config={'testnet': True})
        
        with patch('websockets.connect') as mock_connect:
            mock_connect.return_value.__aenter__.return_value = mock_websocket
            
            # Test WebSocket connection
            async def message_handler(message):
                data = json.loads(message)
                assert 'stream' in data
                assert 'data' in data
            
            # Start WebSocket (would run indefinitely in real usage)
            try:
                await asyncio.wait_for(
                    binance.start_websocket(['btcusdt@ticker'], message_handler),
                    timeout=0.1
                )
            except asyncio.TimeoutError:
                pass  # Expected for this test
            
            # Verify connection was attempted
            mock_connect.assert_called_once()

# ==================== NOTIFICATION SYSTEM TESTS ====================

class TestNotificationSystems:
    """Test notification system integrations"""
    
    @pytest.mark.asyncio
    async def test_email_alerts(self, email_alert_system):
        """Test email alert system"""
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = Mock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            # Test email sending
            success = await email_alert_system.send_alert(
                subject='Test Alert',
                message='This is a test alert',
                recipients=['test@example.com']
            )
            
            assert success
            mock_server.send_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_discord_notifications(self, mock_aiohttp_session):
        """Test Discord notification system"""
        config = {
            'webhook_url': 'https://discord.com/api/webhooks/test',
            'username': 'NeuroCluster Elite',
            'avatar_url': 'https://example.com/avatar.png'
        }
        
        discord = DiscordNotifier(config=config)
        discord.session = mock_aiohttp_session
        
        # Test Discord message
        success = await discord.send_notification(
            title='Test Alert',
            message='This is a test notification',
            color=0x00ff00  # Green
        )
        
        assert success
        mock_aiohttp_session.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_telegram_notifications(self, mock_aiohttp_session):
        """Test Telegram notification system"""
        config = {
            'bot_token': 'test_bot_token',
            'chat_id': 'test_chat_id'
        }
        
        telegram = TelegramNotifier(config=config)
        telegram.session = mock_aiohttp_session
        
        # Test Telegram message
        success = await telegram.send_notification(
            message='Test notification from NeuroCluster Elite'
        )
        
        assert success
        mock_aiohttp_session.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mobile_push_notifications(self, mock_aiohttp_session):
        """Test mobile push notification system"""
        config = {
            'firebase_server_key': 'test_server_key',
            'device_tokens': ['test_device_token']
        }
        
        mobile_push = MobilePushNotifier(config=config)
        mobile_push.session = mock_aiohttp_session
        
        # Test push notification
        success = await mobile_push.send_notification(
            title='NeuroCluster Elite Alert',
            body='New trading signal generated',
            data={'symbol': 'AAPL', 'signal': 'BUY'}
        )
        
        assert success
        mock_aiohttp_session.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_unified_alert_system(self, unified_alert_system):
        """Test unified alert system"""
        # Mock all notification services
        with patch.object(unified_alert_system, 'email_notifier') as mock_email:
            with patch.object(unified_alert_system, 'discord_notifier') as mock_discord:
                with patch.object(unified_alert_system, 'telegram_notifier') as mock_telegram:
                    
                    mock_email.send_alert = AsyncMock(return_value=True)
                    mock_discord.send_notification = AsyncMock(return_value=True)
                    mock_telegram.send_notification = AsyncMock(return_value=True)
                    
                    # Test unified alert
                    alert_data = {
                        'title': 'Trading Signal Alert',
                        'message': 'New BUY signal for AAPL',
                        'priority': 'HIGH',
                        'channels': ['email', 'discord', 'telegram']
                    }
                    
                    success = await unified_alert_system.send_unified_alert(**alert_data)
                    
                    assert success
                    mock_email.send_alert.assert_called_once()
                    mock_discord.send_notification.assert_called_once()
                    mock_telegram.send_notification.assert_called_once()
    
    def test_alert_rate_limiting(self, unified_alert_system):
        """Test alert rate limiting"""
        # Send multiple alerts rapidly
        alert_data = {
            'title': 'Spam Alert',
            'message': 'This is spam',
            'channels': ['email']
        }
        
        # First alert should go through
        result1 = unified_alert_system._should_send_alert(alert_data)
        assert result1
        
        # Immediate duplicate should be rate limited
        result2 = unified_alert_system._should_send_alert(alert_data)
        assert not result2
    
    def test_alert_templating(self, unified_alert_system):
        """Test alert message templating"""
        template_data = {
            'symbol': 'AAPL',
            'signal_type': 'BUY',
            'price': 150.25,
            'confidence': 0.85
        }
        
        message = unified_alert_system._format_alert_message(
            template='trading_signal',
            data=template_data
        )
        
        assert 'AAPL' in message
        assert 'BUY' in message
        assert '150.25' in message
        assert message is not None

# ==================== API RATE LIMITING TESTS ====================

class TestAPIRateLimiting:
    """Test API rate limiting functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter(self):
        """Test API rate limiting"""
        from src.integrations.brokers.alpaca import RateLimiter
        
        # Create rate limiter: 5 calls per second
        rate_limiter = RateLimiter(calls=5, period=1.0)
        
        start_time = time.time()
        
        # Make 10 calls rapidly
        for i in range(10):
            await rate_limiter.acquire()
        
        elapsed_time = time.time() - start_time
        
        # Should take at least 1 second due to rate limiting
        assert elapsed_time >= 1.0
    
    @pytest.mark.asyncio
    async def test_adaptive_rate_limiting(self):
        """Test adaptive rate limiting based on API responses"""
        from src.integrations.exchanges.binance import AdaptiveRateLimiter
        
        rate_limiter = AdaptiveRateLimiter(initial_rate=10)
        
        # Simulate successful requests
        for _ in range(5):
            await rate_limiter.acquire()
            rate_limiter.record_success()
        
        # Rate should increase slightly
        assert rate_limiter.current_rate >= 10
        
        # Simulate rate limit error
        rate_limiter.record_rate_limit_error()
        
        # Rate should decrease significantly
        assert rate_limiter.current_rate < 10
    
    def test_token_bucket_algorithm(self):
        """Test token bucket rate limiting algorithm"""
        from src.integrations.brokers.alpaca import TokenBucket
        
        bucket = TokenBucket(capacity=10, refill_rate=2)  # 2 tokens per second
        
        # Should have full capacity initially
        assert bucket.tokens == 10
        
        # Consume all tokens
        for _ in range(10):
            assert bucket.consume(1)
        
        # Should be empty now
        assert not bucket.consume(1)
        
        # Wait and check refill
        time.sleep(1.1)  # Allow refill
        bucket._refill()
        assert bucket.tokens >= 2

# ==================== SECURITY TESTS ====================

class TestIntegrationSecurity:
    """Test security aspects of integrations"""
    
    def test_api_key_encryption(self):
        """Test API key encryption/decryption"""
        from src.integrations.brokers.alpaca import APIKeyManager
        
        key_manager = APIKeyManager()
        
        original_key = 'test_api_key_12345'
        
        # Encrypt key
        encrypted = key_manager.encrypt_key(original_key)
        assert encrypted != original_key
        
        # Decrypt key
        decrypted = key_manager.decrypt_key(encrypted)
        assert decrypted == original_key
    
    def test_request_signing(self):
        """Test request signing for secure APIs"""
        from src.integrations.exchanges.binance import RequestSigner
        
        signer = RequestSigner('test_secret_key')
        
        params = {
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'type': 'MARKET',
            'quantity': '0.001',
            'timestamp': int(time.time() * 1000)
        }
        
        signature = signer.sign_request(params)
        
        assert signature is not None
        assert len(signature) > 0
        assert signature != 'test_secret_key'  # Should be hashed
    
    def test_ssl_certificate_validation(self):
        """Test SSL certificate validation for API connections"""
        import ssl
        
        # Test SSL context creation
        context = ssl.create_default_context()
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        
        # Should have proper CA certificates
        assert context.ca_certs is not None or context.get_ca_certs()
    
    def test_environment_variable_security(self):
        """Test secure handling of environment variables"""
        import os
        from src.integrations.brokers.alpaca import SecureConfig
        
        # Mock environment variables
        test_env = {
            'ALPACA_API_KEY': 'test_api_key',
            'ALPACA_SECRET_KEY': 'test_secret_key'
        }
        
        with patch.dict(os.environ, test_env):
            config = SecureConfig.from_environment()
            
            assert config.api_key == 'test_api_key'
            assert config.secret_key == 'test_secret_key'
            
            # Keys should not be logged or exposed
            config_str = str(config)
            assert 'test_api_key' not in config_str
            assert 'test_secret_key' not in config_str

# ==================== PERFORMANCE TESTS ====================

class TestIntegrationPerformance:
    """Test performance of integrations"""
    
    @pytest.mark.asyncio
    async def test_concurrent_api_calls(self, mock_aiohttp_session):
        """Test concurrent API call performance"""
        alpaca = AlpacaTrading(config={'api_key': 'test', 'secret_key': 'test'})
        alpaca.session = mock_aiohttp_session
        
        # Test concurrent quote requests
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        start_time = time.time()
        
        tasks = [alpaca.get_quote(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n⚡ Concurrent API Performance:")
        print(f"   Symbols: {len(symbols)}")
        print(f"   Total time: {elapsed_time:.3f}s")
        print(f"   Average per call: {elapsed_time/len(symbols):.3f}s")
        
        # Should be reasonably fast
        assert elapsed_time < 2.0
        assert len(results) == len(symbols)
        
        # Count successful results
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) > 0
    
    @pytest.mark.asyncio
    async def test_websocket_message_throughput(self, mock_websocket):
        """Test WebSocket message processing throughput"""
        binance = BinanceExchange(config={'testnet': True})
        
        messages_received = []
        
        async def high_frequency_handler(message):
            messages_received.append(message)
        
        with patch('websockets.connect') as mock_connect:
            mock_connect.return_value.__aenter__.return_value = mock_websocket
            
            # Simulate high-frequency messages
            mock_websocket.recv = AsyncMock(side_effect=[
                json.dumps({'stream': 'btcusdt@ticker', 'data': MOCK_CRYPTO_TICKER})
                for _ in range(100)
            ] + [websockets.exceptions.ConnectionClosed(None, None)])
            
            try:
                await binance.start_websocket(
                    ['btcusdt@ticker'], 
                    high_frequency_handler
                )
            except websockets.exceptions.ConnectionClosed:
                pass
            
            # Should process messages efficiently
            assert len(messages_received) > 0
    
    def test_memory_usage_with_connections(self):
        """Test memory usage with multiple connections"""
        import tracemalloc
        
        tracemalloc.start()
        
        # Create multiple integration instances
        integrations = []
        for i in range(10):
            alpaca = AlpacaTrading(config={'api_key': f'test_{i}', 'secret_key': f'test_{i}'})
            binance = BinanceExchange(config={'api_key': f'test_{i}', 'secret_key': f'test_{i}'})
            integrations.extend([alpaca, binance])
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"\n💾 Integration Memory Usage:")
        print(f"   Instances: {len(integrations)}")
        print(f"   Current: {current / 1024 / 1024:.1f} MB")
        print(f"   Peak: {peak / 1024 / 1024:.1f} MB")
        
        # Memory should be reasonable
        assert peak < 100 * 1024 * 1024  # Under 100MB

# ==================== ERROR RECOVERY TESTS ====================

class TestErrorRecovery:
    """Test error recovery and resilience"""
    
    @pytest.mark.asyncio
    async def test_connection_retry_logic(self, mock_aiohttp_session):
        """Test connection retry logic"""
        alpaca = AlpacaTrading(config={'api_key': 'test', 'secret_key': 'test'})
        
        # Mock intermittent failures
        call_count = 0
        
        async def failing_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count < 3:  # Fail first 2 attempts
                raise aiohttp.ClientError("Connection failed")
            else:  # Succeed on 3rd attempt
                response = Mock()
                response.status = 200
                response.json = AsyncMock(return_value=MOCK_STOCK_QUOTE)
                return response
        
        alpaca.session = Mock()
        alpaca.session.get = AsyncMock(side_effect=failing_request)
        
        # Should eventually succeed after retries
        quote = await alpaca.get_quote('AAPL')
        
        assert quote is not None
        assert call_count == 3  # Should have retried 2 times
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_pattern(self):
        """Test circuit breaker pattern for failing services"""
        from src.integrations.brokers.alpaca import CircuitBreaker
        
        circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=5.0,
            expected_exception=Exception
        )
        
        # Simulate multiple failures
        for _ in range(4):
            try:
                with circuit_breaker:
                    raise Exception("Service unavailable")
            except Exception:
                pass
        
        # Circuit should be open now
        assert circuit_breaker.state == 'OPEN'
        
        # Further calls should fail fast
        with pytest.raises(Exception):
            with circuit_breaker:
                pass  # Should fail immediately
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, unified_alert_system):
        """Test graceful degradation when services fail"""
        # Mock some services failing
        with patch.object(unified_alert_system, 'email_notifier') as mock_email:
            with patch.object(unified_alert_system, 'discord_notifier') as mock_discord:
                
                # Email fails, Discord succeeds
                mock_email.send_alert = AsyncMock(side_effect=Exception("Email service down"))
                mock_discord.send_notification = AsyncMock(return_value=True)
                
                alert_data = {
                    'title': 'Test Alert',
                    'message': 'Test message',
                    'channels': ['email', 'discord']
                }
                
                # Should still succeed partially
                success = await unified_alert_system.send_unified_alert(**alert_data)
                
                # Should indicate partial success
                assert success  # At least one channel succeeded

# ==================== INTEGRATION TESTING ====================

class TestEndToEndIntegration:
    """Test end-to-end integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_complete_trading_workflow(self, paper_trading_broker, unified_alert_system):
        """Test complete trading workflow"""
        # 1. Receive market data
        market_data = MarketData(
            symbol='AAPL',
            asset_type=AssetType.STOCK,
            price=150.25,
            timestamp=datetime.now()
        )
        
        # 2. Generate trading signal (mock)
        signal = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'quantity': 100,
            'price': 150.25,
            'confidence': 0.85
        }
        
        # 3. Place order
        order = await paper_trading_broker.place_order(
            symbol=signal['symbol'],
            side=signal['action'],
            quantity=signal['quantity'],
            order_type='MARKET'
        )
        
        assert order['status'] == 'FILLED'
        
        # 4. Send alert notification
        alert_success = await unified_alert_system.send_unified_alert(
            title=f"Order Executed: {signal['symbol']}",
            message=f"Successfully {signal['action']} {signal['quantity']} shares of {signal['symbol']}",
            channels=['email']
        )
        
        assert alert_success
        
        # 5. Check portfolio status
        portfolio_value = paper_trading_broker.get_portfolio_value()
        assert portfolio_value > 0
        
        print(f"\n🔄 Complete Trading Workflow Test:")
        print(f"   Signal: {signal['action']} {signal['quantity']} {signal['symbol']}")
        print(f"   Order Status: {order['status']}")
        print(f"   Portfolio Value: ${portfolio_value:,.2f}")
        print(f"   Alert Sent: {alert_success}")
    
    @pytest.mark.asyncio
    async def test_multi_asset_trading(self, paper_trading_broker):
        """Test multi-asset trading scenario"""
        # Trade multiple asset types
        orders = []
        
        # Stock order
        stock_order = await paper_trading_broker.place_order(
            symbol='AAPL',
            side='BUY',
            quantity=100,
            order_type='MARKET'
        )
        orders.append(stock_order)
        
        # ETF order  
        etf_order = await paper_trading_broker.place_order(
            symbol='SPY',
            side='BUY',
            quantity=50,
            order_type='MARKET'
        )
        orders.append(etf_order)
        
        # Verify all orders executed
        for order in orders:
            assert order['status'] == 'FILLED'
        
        # Check portfolio diversity
        positions = paper_trading_broker.get_positions()
        assert len(positions) >= 2
        
        print(f"\n📊 Multi-Asset Trading Test:")
        print(f"   Orders Executed: {len(orders)}")
        print(f"   Active Positions: {len(positions)}")
        for symbol, position in positions.items():
            print(f"   {symbol}: {position['quantity']} shares")

# ==================== MAIN TEST RUNNER ====================

if __name__ == "__main__":
    print("🧪 Running Integration Tests")
    print("=" * 50)
    
    # Run tests with verbose output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings",
        "-x"  # Stop on first failure
    ])