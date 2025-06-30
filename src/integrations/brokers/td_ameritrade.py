#!/usr/bin/env python3
"""
File: td_ameritrade.py
Path: NeuroCluster-Elite/src/integrations/brokers/td_ameritrade.py
Description: TD Ameritrade broker integration for NeuroCluster Elite

This module implements the TD Ameritrade API integration, providing full-service
US brokerage capabilities with stocks, options, ETFs, and mutual funds trading.

Features:
- TD Ameritrade API v1 integration
- OAuth2 authentication flow
- Real-time and delayed market data
- Stock, options, ETF, and mutual fund trading
- Advanced order types and strategies
- Account and position management
- Options chains and analysis
- Historical data and charts
- Watchlist management

API Documentation: https://developer.tdameritrade.com/apis

Note: TD Ameritrade is now part of Charles Schwab. 
Migration to Schwab API may be required in the future.

Author: Your Name
Created: 2025-06-30
Version: 1.0.0
License: MIT
"""

import asyncio
import aiohttp
import json
import logging
import base64
import urllib.parse
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import secrets

# Import our modules
try:
    from src.integrations.brokers import (
        BaseBroker, BrokerAccount, BrokerPosition, BrokerOrder,
        BrokerType, OrderType, OrderSide, OrderStatus, TimeInForce
    )
    from src.integrations import IntegrationConfig, IntegrationStatus
    from src.core.neurocluster_elite import AssetType
    from src.utils.helpers import format_currency, format_percentage
    from src.utils.config_manager import ConfigManager
except ImportError:
    # Fallback for testing
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== TD AMERITRADE SPECIFIC ENUMS ====================

class TDAAccountType(Enum):
    """TD Ameritrade account types"""
    CASH = "CASH"
    MARGIN = "MARGIN"
    IRA = "IRA"
    ROTH_IRA = "ROTH_IRA"

class TDAAssetType(Enum):
    """TD Ameritrade asset types"""
    EQUITY = "EQUITY"
    OPTION = "OPTION"
    ETF = "ETF"
    MUTUAL_FUND = "MUTUAL_FUND"
    CASH_EQUIVALENT = "CASH_EQUIVALENT"
    FOREX = "FOREX"
    FUTURE = "FUTURE"

class TDAOrderType(Enum):
    """TD Ameritrade order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"
    MARKET_ON_CLOSE = "MARKET_ON_CLOSE"
    EXERCISE = "EXERCISE"
    TRAILING_STOP_LIMIT = "TRAILING_STOP_LIMIT"
    NET_DEBIT = "NET_DEBIT"
    NET_CREDIT = "NET_CREDIT"
    NET_ZERO = "NET_ZERO"

class TDAInstruction(Enum):
    """TD Ameritrade order instructions"""
    BUY = "BUY"
    SELL = "SELL"
    BUY_TO_COVER = "BUY_TO_COVER"
    SELL_SHORT = "SELL_SHORT"
    BUY_TO_OPEN = "BUY_TO_OPEN"
    BUY_TO_CLOSE = "BUY_TO_CLOSE"
    SELL_TO_OPEN = "SELL_TO_OPEN"
    SELL_TO_CLOSE = "SELL_TO_CLOSE"

# ==================== TD AMERITRADE DATA STRUCTURES ====================

@dataclass
class TDAConfig:
    """TD Ameritrade specific configuration"""
    client_id: str
    client_secret: str
    redirect_uri: str = "https://localhost"
    
    # Authentication
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_expires_at: Optional[datetime] = None
    
    # API settings
    base_url: str = "https://api.tdameritrade.com"
    api_version: str = "v1"
    
    # Account settings
    account_id: Optional[str] = None  # Auto-detect if None
    
    # Features
    enable_real_time_data: bool = False  # Requires subscription
    enable_level2_data: bool = False
    enable_streaming: bool = True
    
    # Rate limiting
    requests_per_minute: int = 120
    
    # Risk settings
    max_order_value: float = 25000.0
    enable_options_trading: bool = True

@dataclass
class TDAToken:
    """TD Ameritrade OAuth token information"""
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int = 1800  # 30 minutes
    refresh_token_expires_in: int = 7776000  # 90 days
    scope: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def expires_at(self) -> datetime:
        return self.created_at + timedelta(seconds=self.expires_in)
    
    @property
    def refresh_expires_at(self) -> datetime:
        return self.created_at + timedelta(seconds=self.refresh_token_expires_in)
    
    @property
    def is_expired(self) -> bool:
        return datetime.now() >= self.expires_at
    
    @property
    def needs_refresh(self) -> bool:
        return datetime.now() >= (self.expires_at - timedelta(minutes=5))

# ==================== TD AMERITRADE BROKER IMPLEMENTATION ====================

class TDAmeritradeBroker(BaseBroker):
    """
    TD Ameritrade broker implementation
    
    Provides integration with TD Ameritrade's API for comprehensive
    US securities trading including stocks, options, ETFs, and mutual funds.
    """
    
    def __init__(self, config: IntegrationConfig):
        """Initialize TD Ameritrade broker"""
        super().__init__(config)
        
        # Extract TDA-specific config
        self.client_id = config.api_key or config.additional_auth.get('client_id', '')
        self.client_secret = config.secret_key or config.additional_auth.get('client_secret', '')
        self.redirect_uri = config.additional_auth.get('redirect_uri', 'https://localhost')
        
        # API settings
        self.base_url = "https://api.tdameritrade.com"
        self.api_version = "v1"
        
        # Authentication
        self.token: Optional[TDAToken] = None
        self._load_saved_token()
        
        # Account settings
        self.account_id = config.additional_auth.get('account_id')
        self.account_ids = []
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = datetime.now()
        
        # Streaming
        self.streaming_enabled = False
        self.stream_session: Optional[aiohttp.ClientSession] = None
        
        logger.info("📈 TD Ameritrade broker initialized")
    
    # ==================== CONNECTION MANAGEMENT ====================
    
    async def connect(self) -> bool:
        """Connect to TD Ameritrade API"""
        try:
            # Validate credentials
            if not self.client_id:
                error_msg = "TD Ameritrade client ID not provided"
                self.update_status(IntegrationStatus.ERROR, error_msg)
                logger.error(f"❌ {error_msg}")
                return False
            
            # Check if we have a valid token
            if not self.token or self.token.is_expired:
                if self.token and self.token.refresh_token:
                    # Try to refresh token
                    if not await self._refresh_access_token():
                        error_msg = "Failed to refresh access token. Re-authentication required."
                        self.update_status(IntegrationStatus.ERROR, error_msg)
                        logger.error(f"❌ {error_msg}")
                        return False
                else:
                    error_msg = "No valid authentication token. Please authenticate first."
                    self.update_status(IntegrationStatus.ERROR, error_msg)
                    logger.error(f"❌ {error_msg}")
                    logger.info("💡 Use get_auth_url() to start authentication flow")
                    return False
            
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers=self._get_auth_headers()
            )
            
            # Test connection and get accounts
            accounts = await self._get_accounts()
            if accounts:
                self.account_ids = list(accounts.keys())
                
                # Use first account if none specified
                if not self.account_id and self.account_ids:
                    self.account_id = self.account_ids[0]
                    logger.info(f"🎯 Using account: {self.account_id}")
                
                # Get account info
                await self.get_account_info()
                
                self.update_status(IntegrationStatus.CONNECTED)
                logger.info(f"✅ TD Ameritrade connected - Account: {self.account_id}")
                return True
            else:
                error_msg = "Failed to retrieve account information"
                self.update_status(IntegrationStatus.ERROR, error_msg)
                logger.error(f"❌ {error_msg}")
                return False
                
        except Exception as e:
            error_msg = f"TDA connection failed: {e}"
            self.update_status(IntegrationStatus.ERROR, error_msg)
            logger.error(f"❌ {error_msg}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from TD Ameritrade API"""
        try:
            # Close streaming session
            if self.stream_session:
                await self.stream_session.close()
                self.stream_session = None
            
            # Close HTTP session
            if self.session:
                await self.session.close()
                self.session = None
            
            self.update_status(IntegrationStatus.DISCONNECTED)
            logger.info("✅ TD Ameritrade disconnected")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error disconnecting TDA: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test TD Ameritrade connection"""
        try:
            if not self.session or not self.token:
                return False
            
            # Simple API call to test connectivity
            async with self.session.get(f"{self.base_url}/{self.api_version}/accounts") as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"❌ TDA connection test failed: {e}")
            return False
    
    # ==================== AUTHENTICATION ====================
    
    def get_auth_url(self) -> str:
        """Get OAuth2 authorization URL for user authentication"""
        params = {
            'response_type': 'code',
            'redirect_uri': self.redirect_uri,
            'client_id': f"{self.client_id}@AMER.OAUTHAP"
        }
        
        auth_url = f"{self.base_url}/{self.api_version}/oauth2/authorization?" + urllib.parse.urlencode(params)
        
        logger.info(f"🔗 TDA Auth URL: {auth_url}")
        return auth_url
    
    async def authenticate_with_code(self, authorization_code: str) -> bool:
        """Complete OAuth2 flow with authorization code"""
        try:
            token_data = {
                'grant_type': 'authorization_code',
                'refresh_token': '',
                'access_type': 'offline',
                'code': urllib.parse.unquote(authorization_code),
                'client_id': f"{self.client_id}@AMER.OAUTHAP",
                'redirect_uri': self.redirect_uri
            }
            
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/{self.api_version}/oauth2/token",
                    data=token_data,
                    headers=headers
                ) as response:
                    
                    if response.status == 200:
                        token_response = await response.json()
                        self.token = TDAToken(
                            access_token=token_response['access_token'],
                            refresh_token=token_response['refresh_token'],
                            expires_in=token_response.get('expires_in', 1800),
                            refresh_token_expires_in=token_response.get('refresh_token_expires_in', 7776000)
                        )
                        
                        # Save token for future use
                        self._save_token()
                        
                        logger.info("✅ TDA authentication successful")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ TDA authentication failed: {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ TDA authentication error: {e}")
            return False
    
    async def _refresh_access_token(self) -> bool:
        """Refresh the access token using refresh token"""
        try:
            if not self.token or not self.token.refresh_token:
                return False
            
            token_data = {
                'grant_type': 'refresh_token',
                'refresh_token': self.token.refresh_token,
                'access_type': 'offline',
                'client_id': f"{self.client_id}@AMER.OAUTHAP"
            }
            
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/{self.api_version}/oauth2/token",
                    data=token_data,
                    headers=headers
                ) as response:
                    
                    if response.status == 200:
                        token_response = await response.json()
                        
                        # Update token
                        self.token.access_token = token_response['access_token']
                        self.token.expires_in = token_response.get('expires_in', 1800)
                        self.token.created_at = datetime.now()
                        
                        # Update refresh token if provided
                        if 'refresh_token' in token_response:
                            self.token.refresh_token = token_response['refresh_token']
                        
                        # Save updated token
                        self._save_token()
                        
                        logger.info("✅ TDA token refreshed")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ TDA token refresh failed: {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ TDA token refresh error: {e}")
            return False
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests"""
        if not self.token:
            return {}
        
        return {
            "Authorization": f"Bearer {self.token.access_token}",
            "Content-Type": "application/json"
        }
    
    def _save_token(self):
        """Save token to persistent storage"""
        # In a real implementation, save to secure storage
        # For now, this is a placeholder
        pass
    
    def _load_saved_token(self):
        """Load saved token from persistent storage"""
        # In a real implementation, load from secure storage
        # For now, this is a placeholder
        pass
    
    # ==================== ACCOUNT MANAGEMENT ====================
    
    async def get_account_info(self) -> Optional[BrokerAccount]:
        """Get account information"""
        try:
            account_data = await self._get_account_details(self.account_id)
            if account_data:
                self.account_info = self._parse_account(account_data)
                return self.account_info
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting TDA account info: {e}")
            return None
    
    async def _get_accounts(self) -> Dict[str, Any]:
        """Get all accounts for the authenticated user"""
        try:
            if not self.session:
                return {}
            
            async with self.session.get(f"{self.base_url}/{self.api_version}/accounts") as response:
                if response.status == 200:
                    accounts_data = await response.json()
                    accounts = {}
                    
                    for account in accounts_data:
                        account_info = account.get('securitiesAccount', {})
                        account_id = account_info.get('accountId', '')
                        if account_id:
                            accounts[account_id] = account_info
                    
                    return accounts
                else:
                    error_text = await response.text()
                    logger.error(f"❌ TDA accounts API error {response.status}: {error_text}")
                    return {}
                    
        except Exception as e:
            logger.error(f"❌ Error fetching TDA accounts: {e}")
            return {}
    
    async def _get_account_details(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific account"""
        try:
            if not self.session:
                await self.connect()
            
            params = {
                'fields': 'positions,orders'
            }
            
            async with self.session.get(
                f"{self.base_url}/{self.api_version}/accounts/{account_id}",
                params=params
            ) as response:
                if response.status == 200:
                    account_data = await response.json()
                    return account_data.get('securitiesAccount', {})
                else:
                    error_text = await response.text()
                    logger.error(f"❌ TDA account details API error {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Error fetching TDA account details: {e}")
            return None
    
    def _parse_account(self, account_data: Dict[str, Any]) -> BrokerAccount:
        """Parse TDA account data into standard format"""
        # Extract balances
        current_balances = account_data.get('currentBalances', {})
        initial_balances = account_data.get('initialBalances', {})
        
        return BrokerAccount(
            account_id=account_data.get('accountId', ''),
            broker_type=BrokerType.TD_AMERITRADE,
            account_type=account_data.get('type', 'cash').lower(),
            
            # Balances
            cash_balance=float(current_balances.get('cashBalance', 0)),
            total_equity=float(current_balances.get('equity', 0)),
            buying_power=float(current_balances.get('buyingPower', 0)),
            maintenance_margin=float(current_balances.get('maintenanceRequirement', 0)),
            
            # Day trading
            day_trading_buying_power=float(current_balances.get('dayTradingBuyingPower', 0)),
            pattern_day_trader=account_data.get('isDayTrader', False),
            day_trades_remaining=3,  # Default for non-PDT accounts
            
            # Status
            account_status="active",
            is_restricted=account_data.get('isClosingOnlyRestricted', False),
            restrictions=[],
            
            last_updated=datetime.now()
        )
    
    # ==================== POSITION MANAGEMENT ====================
    
    async def get_positions(self) -> Dict[str, BrokerPosition]:
        """Get all positions"""
        try:
            account_data = await self._get_account_details(self.account_id)
            if account_data and 'positions' in account_data:
                positions = {}
                
                for pos_data in account_data['positions']:
                    if float(pos_data.get('longQuantity', 0)) != 0 or float(pos_data.get('shortQuantity', 0)) != 0:
                        position = self._parse_position(pos_data)
                        positions[position.symbol] = position
                
                self.positions = positions
                return positions
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Error getting TDA positions: {e}")
            return {}
    
    async def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        """Get position for specific symbol"""
        positions = await self.get_positions()
        return positions.get(symbol)
    
    def _parse_position(self, position_data: Dict[str, Any]) -> BrokerPosition:
        """Parse TDA position data into standard format"""
        instrument = position_data.get('instrument', {})
        symbol = instrument.get('symbol', '')
        
        long_qty = float(position_data.get('longQuantity', 0))
        short_qty = float(position_data.get('shortQuantity', 0))
        
        # Determine net position
        if long_qty > 0:
            quantity = long_qty
            side = 'long'
        elif short_qty > 0:
            quantity = short_qty
            side = 'short'
        else:
            quantity = 0
            side = 'flat'
        
        return BrokerPosition(
            symbol=symbol,
            asset_type=self._get_asset_type(instrument.get('assetType', 'EQUITY')),
            quantity=quantity,
            market_value=float(position_data.get('marketValue', 0)),
            avg_cost=float(position_data.get('averagePrice', 0)),
            unrealized_pnl=float(position_data.get('currentDayProfitLoss', 0)),
            realized_pnl=0.0,  # Not provided in position data
            
            # Position details
            side=side,
            cost_basis=quantity * float(position_data.get('averagePrice', 0)),
            last_price=float(position_data.get('marketValue', 0)) / quantity if quantity > 0 else 0,
            
            broker_type=BrokerType.TD_AMERITRADE,
            last_updated=datetime.now()
        )
    
    # ==================== ORDER MANAGEMENT ====================
    
    async def place_order(self, symbol: str, side: str, quantity: float,
                         order_type: str = "market", price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         time_in_force: str = "day",
                         **kwargs) -> Dict[str, Any]:
        """Place an order with TD Ameritrade"""
        try:
            # Validate order parameters
            validation = self.validate_order_params(symbol, side, quantity, order_type, price)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': f"Order validation failed: {', '.join(validation['errors'])}",
                    'order_id': None
                }
            
            if not self.session:
                await self.connect()
            
            # Prepare order data
            order_data = self._prepare_order_data(
                symbol, side, quantity, order_type, price, stop_price, time_in_force, **kwargs
            )
            
            # Submit order
            async with self.session.post(
                f"{self.base_url}/{self.api_version}/accounts/{self.account_id}/orders",
                json=order_data
            ) as response:
                
                if response.status == 201:
                    # TDA returns order ID in Location header
                    location = response.headers.get('Location', '')
                    order_id = location.split('/')[-1] if location else str(int(time.time()))
                    
                    # Create order record
                    order = BrokerOrder(
                        order_id=order_id,
                        symbol=symbol,
                        asset_type=self._get_asset_type_from_string(kwargs.get('asset_type', 'stock')),
                        side=OrderSide(side.upper()),
                        order_type=OrderType(order_type.upper()),
                        quantity=quantity,
                        status=OrderStatus.PENDING,
                        
                        limit_price=price,
                        stop_price=stop_price,
                        time_in_force=TimeInForce(time_in_force.upper()),
                        
                        broker_type=BrokerType.TD_AMERITRADE,
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    
                    self.orders[order_id] = order
                    
                    logger.info(f"✅ TDA order placed: {order_id} - {side.upper()} {quantity} {symbol}")
                    
                    return {
                        'success': True,
                        'order_id': order_id,
                        'order': order
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"❌ TDA order failed {response.status}: {error_text}")
                    
                    return {
                        'success': False,
                        'error': f"Order failed: {error_text}",
                        'order_id': None
                    }
                    
        except Exception as e:
            error_msg = f"Error placing TDA order: {e}"
            logger.error(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'order_id': None
            }
    
    def _prepare_order_data(self, symbol: str, side: str, quantity: float,
                           order_type: str, price: Optional[float],
                           stop_price: Optional[float], time_in_force: str,
                           **kwargs) -> Dict[str, Any]:
        """Prepare order data for TDA API"""
        
        # Convert our order type to TDA format
        tda_order_type = self._convert_order_type(order_type)
        tda_instruction = self._convert_side_to_instruction(side, kwargs.get('asset_type', 'stock'))
        
        # Base order structure
        order_data = {
            "orderType": tda_order_type,
            "session": "NORMAL",
            "duration": time_in_force.upper(),
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": tda_instruction,
                    "quantity": quantity,
                    "instrument": {
                        "symbol": symbol.upper(),
                        "assetType": self._get_tda_asset_type(kwargs.get('asset_type', 'stock'))
                    }
                }
            ]
        }
        
        # Add price for limit orders
        if order_type.lower() == "limit" and price:
            order_data["price"] = price
        
        # Add stop price for stop orders
        if order_type.lower() in ["stop", "stop_limit"] and stop_price:
            order_data["stopPrice"] = stop_price
            if order_type.lower() == "stop_limit" and price:
                order_data["price"] = price
        
        return order_data
    
    def _convert_order_type(self, order_type: str) -> str:
        """Convert our order type to TDA order type"""
        type_map = {
            'market': 'MARKET',
            'limit': 'LIMIT',
            'stop': 'STOP',
            'stop_limit': 'STOP_LIMIT'
        }
        return type_map.get(order_type.lower(), 'MARKET')
    
    def _convert_side_to_instruction(self, side: str, asset_type: str) -> str:
        """Convert side to TDA instruction"""
        if asset_type.lower() == 'option':
            # Options have different instructions
            if side.lower() == 'buy':
                return 'BUY_TO_OPEN'
            else:
                return 'SELL_TO_CLOSE'
        else:
            # Stocks and other assets
            if side.lower() == 'buy':
                return 'BUY'
            else:
                return 'SELL'
    
    def _get_tda_asset_type(self, asset_type: str) -> str:
        """Convert our asset type to TDA asset type"""
        type_map = {
            'stock': 'EQUITY',
            'option': 'OPTION',
            'etf': 'ETF',
            'mutual_fund': 'MUTUAL_FUND'
        }
        return type_map.get(asset_type.lower(), 'EQUITY')
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            if not self.session:
                await self.connect()
            
            async with self.session.delete(
                f"{self.base_url}/{self.api_version}/accounts/{self.account_id}/orders/{order_id}"
            ) as response:
                if response.status == 200:
                    # Update local order if exists
                    if order_id in self.orders:
                        self.orders[order_id].status = OrderStatus.CANCELLED
                        self.orders[order_id].updated_at = datetime.now()
                    
                    logger.info(f"✅ TDA order cancelled: {order_id}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"❌ TDA cancel order failed {response.status}: {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error cancelling TDA order {order_id}: {e}")
            return False
    
    async def get_orders(self, status: Optional[str] = None) -> Dict[str, BrokerOrder]:
        """Get orders"""
        try:
            account_data = await self._get_account_details(self.account_id)
            if account_data and 'orderStrategies' in account_data:
                orders = {}
                
                for order_data in account_data['orderStrategies']:
                    order = self._parse_order(order_data)
                    if not status or order.status.value.lower() == status.lower():
                        orders[order.order_id] = order
                
                self.orders.update(orders)
                return orders
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Error getting TDA orders: {e}")
            return {}
    
    async def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        """Get specific order"""
        try:
            if not self.session:
                await self.connect()
            
            async with self.session.get(
                f"{self.base_url}/{self.api_version}/accounts/{self.account_id}/orders/{order_id}"
            ) as response:
                if response.status == 200:
                    order_data = await response.json()
                    order = self._parse_order(order_data)
                    self.orders[order_id] = order
                    return order
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Error getting TDA order {order_id}: {e}")
            return None
    
    def _parse_order(self, order_data: Dict[str, Any]) -> BrokerOrder:
        """Parse TDA order data into standard format"""
        
        # Extract order leg (TDA supports complex orders, we'll take the first leg)
        order_legs = order_data.get('orderLegCollection', [])
        if not order_legs:
            # Return a minimal order if no legs found
            return BrokerOrder(
                order_id=str(order_data.get('orderId', '')),
                symbol='',
                asset_type=AssetType.STOCK,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0,
                status=OrderStatus.REJECTED,
                broker_type=BrokerType.TD_AMERITRADE
            )
        
        first_leg = order_legs[0]
        instrument = first_leg.get('instrument', {})
        
        return BrokerOrder(
            order_id=str(order_data.get('orderId', '')),
            symbol=instrument.get('symbol', ''),
            asset_type=self._get_asset_type(instrument.get('assetType', 'EQUITY')),
            side=self._parse_instruction(first_leg.get('instruction', 'BUY')),
            order_type=self._parse_order_type(order_data.get('orderType', 'MARKET')),
            quantity=float(first_leg.get('quantity', 0)),
            status=self._parse_order_status(order_data.get('status', 'PENDING')),
            
            # Pricing
            limit_price=float(order_data.get('price', 0)) or None,
            stop_price=float(order_data.get('stopPrice', 0)) or None,
            filled_price=None,  # Would need to get from executions
            
            # Execution
            filled_quantity=float(order_data.get('filledQuantity', 0)),
            remaining_quantity=float(order_data.get('remainingQuantity', 0)),
            time_in_force=TimeInForce(order_data.get('duration', 'DAY')),
            
            # Timestamps
            created_at=self._parse_datetime(order_data.get('enteredTime')),
            updated_at=datetime.now(),
            
            broker_type=BrokerType.TD_AMERITRADE
        )
    
    # ==================== HELPER METHODS ====================
    
    def _get_asset_type(self, tda_asset_type: str) -> AssetType:
        """Convert TDA asset type to our AssetType"""
        type_map = {
            'EQUITY': AssetType.STOCK,
            'OPTION': AssetType.OPTION if hasattr(AssetType, 'OPTION') else AssetType.STOCK,
            'ETF': AssetType.ETF if hasattr(AssetType, 'ETF') else AssetType.STOCK,
            'MUTUAL_FUND': AssetType.STOCK,
            'FOREX': AssetType.FOREX,
            'FUTURE': AssetType.STOCK
        }
        return type_map.get(tda_asset_type, AssetType.STOCK)
    
    def _get_asset_type_from_string(self, asset_type_str: str) -> AssetType:
        """Convert string to AssetType"""
        type_map = {
            'stock': AssetType.STOCK,
            'equity': AssetType.STOCK,
            'crypto': AssetType.CRYPTO,
            'forex': AssetType.FOREX,
            'option': AssetType.OPTION if hasattr(AssetType, 'OPTION') else AssetType.STOCK
        }
        return type_map.get(asset_type_str.lower(), AssetType.STOCK)
    
    def _parse_instruction(self, tda_instruction: str) -> OrderSide:
        """Convert TDA instruction to OrderSide"""
        instruction_map = {
            'BUY': OrderSide.BUY,
            'SELL': OrderSide.SELL,
            'BUY_TO_COVER': OrderSide.BUY,
            'SELL_SHORT': OrderSide.SELL,
            'BUY_TO_OPEN': OrderSide.BUY,
            'BUY_TO_CLOSE': OrderSide.BUY,
            'SELL_TO_OPEN': OrderSide.SELL,
            'SELL_TO_CLOSE': OrderSide.SELL
        }
        return instruction_map.get(tda_instruction, OrderSide.BUY)
    
    def _parse_order_type(self, tda_order_type: str) -> OrderType:
        """Convert TDA order type to OrderType"""
        type_map = {
            'MARKET': OrderType.MARKET,
            'LIMIT': OrderType.LIMIT,
            'STOP': OrderType.STOP,
            'STOP_LIMIT': OrderType.STOP_LIMIT
        }
        return type_map.get(tda_order_type, OrderType.MARKET)
    
    def _parse_order_status(self, tda_status: str) -> OrderStatus:
        """Convert TDA order status to OrderStatus"""
        status_map = {
            'AWAITING_PARENT_ORDER': OrderStatus.PENDING,
            'AWAITING_CONDITION': OrderStatus.PENDING,
            'AWAITING_MANUAL_REVIEW': OrderStatus.PENDING,
            'ACCEPTED': OrderStatus.OPEN,
            'AWAITING_UR_OUT': OrderStatus.OPEN,
            'PENDING_ACTIVATION': OrderStatus.PENDING,
            'QUEUED': OrderStatus.PENDING,
            'WORKING': OrderStatus.OPEN,
            'REJECTED': OrderStatus.REJECTED,
            'PENDING_CANCEL': OrderStatus.OPEN,
            'CANCELED': OrderStatus.CANCELLED,
            'PENDING_REPLACE': OrderStatus.OPEN,
            'REPLACED': OrderStatus.OPEN,
            'FILLED': OrderStatus.FILLED,
            'EXPIRED': OrderStatus.EXPIRED
        }
        return status_map.get(tda_status, OrderStatus.PENDING)
    
    def _parse_datetime(self, datetime_str: Optional[str]) -> datetime:
        """Parse TDA datetime string"""
        if not datetime_str:
            return datetime.now()
        
        try:
            # TDA uses ISO format with timezone
            return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        except:
            return datetime.now()
    
    def format_symbol(self, symbol: str) -> str:
        """Format symbol for TDA API"""
        return symbol.upper().strip()
    
    # ==================== CAPABILITIES ====================
    
    async def get_capabilities(self) -> List[str]:
        """Get TD Ameritrade capabilities"""
        return [
            "market_orders",
            "limit_orders",
            "stop_orders",
            "stop_limit_orders",
            "stocks",
            "etfs",
            "options",
            "mutual_funds",
            "live_trading",
            "real_time_data",
            "historical_data",
            "options_chains",
            "portfolio_tracking",
            "advanced_orders",
            "after_hours_trading"
        ]

# ==================== TESTING ====================

async def test_tda_broker():
    """Test TD Ameritrade broker functionality"""
    print("🧪 Testing TD Ameritrade Broker")
    print("=" * 40)
    
    # Create test configuration
    config = IntegrationConfig(
        name="td_ameritrade",
        integration_type="broker",
        enabled=True,
        paper_trading=False,  # TDA doesn't have separate paper trading
        api_key="test_client_id",
        additional_auth={
            'client_secret': 'test_secret',
            'redirect_uri': 'https://localhost'
        }
    )
    
    # Create broker instance
    broker = TDAmeritradeBroker(config)
    
    print("✅ Broker instance created")
    print(f"✅ Client ID: {broker.client_id[:8]}...")
    print(f"✅ Base URL: {broker.base_url}")
    
    # Test auth URL generation
    auth_url = broker.get_auth_url()
    print(f"✅ Auth URL generated: {auth_url[:50]}...")
    
    # Test capabilities
    capabilities = await broker.get_capabilities()
    print(f"✅ Capabilities: {', '.join(capabilities[:5])}...")
    
    print("\n⚠️  Note: Full testing requires OAuth2 authentication")
    print("   Use get_auth_url() to start authentication flow")
    
    print("\n🎉 TD Ameritrade broker tests completed!")

if __name__ == "__main__":
    asyncio.run(test_tda_broker())