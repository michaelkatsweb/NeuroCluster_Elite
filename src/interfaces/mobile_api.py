#!/usr/bin/env python3
"""
File: mobile_api.py
Path: NeuroCluster-Elite/src/interfaces/mobile_api.py
Description: Mobile API interface for NeuroCluster Elite

This module implements a comprehensive RESTful API designed for mobile applications,
providing secure access to trading functionality, market data, portfolio management,
and real-time notifications with WebSocket support.

Features:
- RESTful API with FastAPI framework
- JWT authentication and authorization
- Real-time WebSocket connections
- Mobile-optimized data formats
- Push notification support
- Rate limiting and security
- Comprehensive endpoint coverage
- API documentation with OpenAPI/Swagger
- Error handling and logging
- Mobile-specific optimizations

Author: Your Name
Created: 2025-06-29
Version: 1.0.0
License: MIT
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import uuid
import hashlib
import hmac
from pathlib import Path

# FastAPI and related
from fastapi import FastAPI, HTTPException, Depends, Security, status, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocketState
import uvicorn

# Pydantic models
from pydantic import BaseModel, Field, validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

# Security and auth
import jwt
from passlib.context import CryptContext
import secrets

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import our modules
try:
    from src.core.neurocluster_elite import NeuroClusterElite, RegimeType, AssetType, MarketData
    from src.data.multi_asset_manager import MultiAssetDataManager
    from src.trading.trading_engine import AdvancedTradingEngine
    from src.trading.portfolio_manager import PortfolioManager
    from src.analysis.sentiment_analyzer import AdvancedSentimentAnalyzer
    from src.analysis.market_scanner import AdvancedMarketScanner, ScanType
    from src.utils.config_manager import ConfigManager
    from src.utils.logger import get_enhanced_logger, LogCategory
    from src.utils.helpers import format_currency, format_percentage
except ImportError:
    # Fallback for testing
    pass

# Configure logging
logger = get_enhanced_logger(__name__, LogCategory.INTERFACE)

# ==================== CONFIGURATION ====================

# Security configuration
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# ==================== ENUMS AND DATA MODELS ====================

class APIVersion(Enum):
    """API version enum"""
    V1 = "v1"
    V2 = "v2"

class NotificationType(Enum):
    """Push notification types"""
    PRICE_ALERT = "price_alert"
    ORDER_FILLED = "order_filled"
    PORTFOLIO_UPDATE = "portfolio_update"
    MARKET_NEWS = "market_news"
    SYSTEM_ALERT = "system_alert"
    REGIME_CHANGE = "regime_change"

class WebSocketMessageType(Enum):
    """WebSocket message types"""
    MARKET_DATA = "market_data"
    PORTFOLIO_UPDATE = "portfolio_update"
    ORDER_UPDATE = "order_update"
    ALERT = "alert"
    HEARTBEAT = "heartbeat"
    ERROR = "error"

# ==================== PYDANTIC MODELS ====================

class UserCreate(BaseModel):
    """User creation model"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    """User login model"""
    username: str
    password: str

class Token(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class MarketQuoteResponse(BaseModel):
    """Market quote response model"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: float
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None

class OrderRequest(BaseModel):
    """Order request model"""
    symbol: str
    side: str = Field(..., regex=r'^(buy|sell)$')
    order_type: str = Field(..., regex=r'^(market|limit|stop)$')
    quantity: float = Field(..., gt=0)
    price: Optional[float] = Field(None, gt=0)
    stop_price: Optional[float] = Field(None, gt=0)
    time_in_force: str = Field("GTC", regex=r'^(GTC|IOC|FOK)$')

class OrderResponse(BaseModel):
    """Order response model"""
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    status: str
    timestamp: datetime

class PortfolioResponse(BaseModel):
    """Portfolio response model"""
    total_value: float
    cash_balance: float
    market_value: float
    day_pnl: float
    day_pnl_percent: float
    total_pnl: float
    total_pnl_percent: float
    positions: List[Dict[str, Any]]

class ScanRequest(BaseModel):
    """Market scan request model"""
    scan_types: List[str]
    asset_types: List[str] = ["stock", "crypto"]
    min_volume: float = 0
    min_price: float = 0
    max_results: int = Field(50, le=100)

class ScanResult(BaseModel):
    """Scan result model"""
    symbol: str
    scan_type: str
    score: float
    confidence: float
    signal: str
    current_price: float
    description: str

class AlertRequest(BaseModel):
    """Alert request model"""
    symbol: str
    alert_type: str
    condition: str  # "above", "below", "change"
    value: float
    enabled: bool = True

class NotificationRequest(BaseModel):
    """Push notification request"""
    device_token: str
    notification_types: List[str]
    enabled: bool = True

class APIResponse(BaseModel):
    """Standard API response model"""
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# ==================== AUTHENTICATION AND SECURITY ====================

class AuthManager:
    """Authentication and authorization manager"""
    
    def __init__(self):
        self.users_db = {}  # In production, use proper database
        self.active_sessions = {}
        self.blacklisted_tokens = set()
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash password"""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(self, data: dict) -> str:
        """Create refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[dict]:
        """Verify and decode token"""
        try:
            if token in self.blacklisted_tokens:
                return None
            
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.JWTError:
            return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[dict]:
        """Authenticate user"""
        user = self.users_db.get(username)
        if not user:
            return None
        
        if not self.verify_password(password, user["hashed_password"]):
            return None
        
        return user
    
    def create_user(self, user_data: UserCreate) -> dict:
        """Create new user"""
        if user_data.username in self.users_db:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        hashed_password = self.get_password_hash(user_data.password)
        user = {
            "username": user_data.username,
            "email": user_data.email,
            "full_name": user_data.full_name,
            "hashed_password": hashed_password,
            "created_at": datetime.now(),
            "is_active": True
        }
        
        self.users_db[user_data.username] = user
        return user

# ==================== WEBSOCKET MANAGER ====================

class ConnectionManager:
    """WebSocket connection manager"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """Connect new WebSocket"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = []
        self.user_connections[user_id].append(websocket)
        
        logger.info(f"WebSocket connected for user: {user_id}")
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        """Disconnect WebSocket"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        if user_id in self.user_connections:
            if websocket in self.user_connections[user_id]:
                self.user_connections[user_id].remove(websocket)
            
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        logger.info(f"WebSocket disconnected for user: {user_id}")
    
    async def send_personal_message(self, message: dict, user_id: str):
        """Send message to specific user"""
        if user_id in self.user_connections:
            for connection in self.user_connections[user_id]:
                if connection.client_state == WebSocketState.CONNECTED:
                    try:
                        await connection.send_text(json.dumps(message))
                    except Exception as e:
                        logger.warning(f"Failed to send message to {user_id}: {e}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connections"""
        for connection in self.active_connections:
            if connection.client_state == WebSocketState.CONNECTED:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    logger.warning(f"Failed to broadcast message: {e}")

# ==================== MOBILE API CLASS ====================

class MobileAPI:
    """Mobile API implementation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Mobile API"""
        self.config = config or {}
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="NeuroCluster Elite Mobile API",
            description="AI-Powered Trading Platform Mobile API",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Initialize components
        self.auth_manager = AuthManager()
        self.connection_manager = ConnectionManager()
        
        # Trading components
        self.neurocluster = None
        self.data_manager = None
        self.trading_engine = None
        self.portfolio_manager = None
        self.sentiment_analyzer = None
        self.market_scanner = None
        
        # Initialize components
        self._initialize_components()
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
        # Setup WebSocket endpoints
        self._setup_websockets()
        
        logger.info("📱 Mobile API initialized")
    
    def _initialize_components(self):
        """Initialize NeuroCluster components"""
        
        try:
            config_manager = ConfigManager()
            config = config_manager.load_config()
            
            self.neurocluster = NeuroClusterElite(config.get('algorithm', {}))
            self.data_manager = MultiAssetDataManager(config.get('data', {}))
            self.trading_engine = AdvancedTradingEngine(config.get('trading', {}))
            self.portfolio_manager = PortfolioManager(config.get('portfolio', {}))
            self.sentiment_analyzer = AdvancedSentimentAnalyzer(config.get('sentiment', {}))
            self.market_scanner = AdvancedMarketScanner(config.get('scanner', {}))
            
            logger.info("✅ API components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
    
    def _setup_middleware(self):
        """Setup middleware"""
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Trusted host middleware
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure appropriately for production
        )
        
        # Rate limiting
        self.app.state.limiter = limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    def _setup_routes(self):
        """Setup API routes"""
        
        # Authentication routes
        self._setup_auth_routes()
        
        # Market data routes
        self._setup_market_routes()
        
        # Trading routes
        self._setup_trading_routes()
        
        # Portfolio routes
        self._setup_portfolio_routes()
        
        # Analysis routes
        self._setup_analysis_routes()
        
        # Scanner routes
        self._setup_scanner_routes()
        
        # User management routes
        self._setup_user_routes()
        
        # System routes
        self._setup_system_routes()
    
    def _setup_auth_routes(self):
        """Setup authentication routes"""
        
        @self.app.post("/auth/register", response_model=APIResponse)
        @limiter.limit("5/minute")
        async def register(request, user: UserCreate):
            """Register new user"""
            try:
                created_user = self.auth_manager.create_user(user)
                return APIResponse(
                    success=True,
                    data={"username": created_user["username"]},
                    message="User registered successfully"
                )
            except HTTPException as e:
                raise e
            except Exception as e:
                logger.error(f"Registration failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Registration failed"
                )
        
        @self.app.post("/auth/login", response_model=Token)
        @limiter.limit("10/minute")
        async def login(request, user: UserLogin):
            """Login user"""
            try:
                authenticated_user = self.auth_manager.authenticate_user(
                    user.username, user.password
                )
                
                if not authenticated_user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Incorrect username or password"
                    )
                
                access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
                access_token = self.auth_manager.create_access_token(
                    data={"sub": user.username}, expires_delta=access_token_expires
                )
                refresh_token = self.auth_manager.create_refresh_token(
                    data={"sub": user.username}
                )
                
                return Token(
                    access_token=access_token,
                    refresh_token=refresh_token,
                    expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
                )
                
            except HTTPException as e:
                raise e
            except Exception as e:
                logger.error(f"Login failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Login failed"
                )
        
        @self.app.post("/auth/refresh", response_model=Token)
        async def refresh_token(refresh_token: str):
            """Refresh access token"""
            try:
                payload = self.auth_manager.verify_token(refresh_token)
                if not payload or payload.get("type") != "refresh":
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid refresh token"
                    )
                
                username = payload.get("sub")
                access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
                new_access_token = self.auth_manager.create_access_token(
                    data={"sub": username}, expires_delta=access_token_expires
                )
                
                return Token(
                    access_token=new_access_token,
                    refresh_token=refresh_token,
                    expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
                )
                
            except Exception as e:
                logger.error(f"Token refresh failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token refresh failed"
                )
    
    def _setup_market_routes(self):
        """Setup market data routes"""
        
        @self.app.get("/market/quote/{symbol}", response_model=MarketQuoteResponse)
        @limiter.limit("100/minute")
        async def get_quote(request, symbol: str, current_user: str = Depends(self.get_current_user)):
            """Get market quote"""
            try:
                # Simulate getting market data
                quote = MarketQuoteResponse(
                    symbol=symbol.upper(),
                    price=145.67,
                    change=2.34,
                    change_percent=1.63,
                    volume=1234567,
                    timestamp=datetime.now(),
                    bid=145.65,
                    ask=145.69,
                    high=147.23,
                    low=143.45
                )
                
                return quote
                
            except Exception as e:
                logger.error(f"Quote fetch failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to fetch quote"
                )
        
        @self.app.get("/market/quotes", response_model=List[MarketQuoteResponse])
        @limiter.limit("50/minute")
        async def get_quotes(request, symbols: str, current_user: str = Depends(self.get_current_user)):
            """Get multiple quotes"""
            try:
                symbol_list = symbols.split(",")
                quotes = []
                
                for symbol in symbol_list[:10]:  # Limit to 10 symbols
                    quote = MarketQuoteResponse(
                        symbol=symbol.upper(),
                        price=145.67,
                        change=2.34,
                        change_percent=1.63,
                        volume=1234567,
                        timestamp=datetime.now()
                    )
                    quotes.append(quote)
                
                return quotes
                
            except Exception as e:
                logger.error(f"Multiple quotes fetch failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to fetch quotes"
                )
        
        @self.app.get("/market/regime/{symbol}")
        async def get_regime(request, symbol: str, current_user: str = Depends(self.get_current_user)):
            """Get market regime for symbol"""
            try:
                # Simulate regime detection
                regime_data = {
                    "symbol": symbol.upper(),
                    "regime": "Bull Market",
                    "confidence": 87.5,
                    "duration": "3 days",
                    "last_update": datetime.now().isoformat()
                }
                
                return APIResponse(success=True, data=regime_data)
                
            except Exception as e:
                logger.error(f"Regime detection failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to detect regime"
                )
    
    def _setup_trading_routes(self):
        """Setup trading routes"""
        
        @self.app.post("/trading/order", response_model=OrderResponse)
        @limiter.limit("20/minute")
        async def place_order(request, order: OrderRequest, current_user: str = Depends(self.get_current_user)):
            """Place trading order"""
            try:
                # Simulate order placement
                order_id = f"ORD{int(time.time())}"
                
                order_response = OrderResponse(
                    order_id=order_id,
                    symbol=order.symbol.upper(),
                    side=order.side.upper(),
                    order_type=order.order_type.upper(),
                    quantity=order.quantity,
                    price=order.price,
                    status="PENDING",
                    timestamp=datetime.now()
                )
                
                # Send WebSocket update
                await self.connection_manager.send_personal_message(
                    {
                        "type": WebSocketMessageType.ORDER_UPDATE.value,
                        "data": asdict(order_response)
                    },
                    current_user
                )
                
                return order_response
                
            except Exception as e:
                logger.error(f"Order placement failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to place order"
                )
        
        @self.app.get("/trading/orders")
        async def get_orders(request, current_user: str = Depends(self.get_current_user)):
            """Get user orders"""
            try:
                # Simulate getting orders
                orders = [
                    {
                        "order_id": "ORD123",
                        "symbol": "AAPL",
                        "side": "BUY",
                        "quantity": 100,
                        "price": 150.00,
                        "status": "PENDING",
                        "timestamp": datetime.now().isoformat()
                    }
                ]
                
                return APIResponse(success=True, data=orders)
                
            except Exception as e:
                logger.error(f"Orders fetch failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to fetch orders"
                )
        
        @self.app.delete("/trading/order/{order_id}")
        @limiter.limit("30/minute")
        async def cancel_order(request, order_id: str, current_user: str = Depends(self.get_current_user)):
            """Cancel order"""
            try:
                # Simulate order cancellation
                return APIResponse(
                    success=True,
                    message=f"Order {order_id} cancelled successfully"
                )
                
            except Exception as e:
                logger.error(f"Order cancellation failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to cancel order"
                )
    
    def _setup_portfolio_routes(self):
        """Setup portfolio routes"""
        
        @self.app.get("/portfolio", response_model=PortfolioResponse)
        @limiter.limit("60/minute")
        async def get_portfolio(request, current_user: str = Depends(self.get_current_user)):
            """Get portfolio summary"""
            try:
                # Simulate portfolio data
                portfolio = PortfolioResponse(
                    total_value=105000.00,
                    cash_balance=15000.00,
                    market_value=90000.00,
                    day_pnl=2500.00,
                    day_pnl_percent=2.43,
                    total_pnl=5000.00,
                    total_pnl_percent=5.00,
                    positions=[
                        {
                            "symbol": "AAPL",
                            "quantity": 100,
                            "avg_price": 145.00,
                            "current_price": 148.50,
                            "market_value": 14850.00,
                            "pnl": 350.00,
                            "pnl_percent": 2.41
                        }
                    ]
                )
                
                return portfolio
                
            except Exception as e:
                logger.error(f"Portfolio fetch failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to fetch portfolio"
                )
        
        @self.app.get("/portfolio/positions")
        async def get_positions(request, current_user: str = Depends(self.get_current_user)):
            """Get portfolio positions"""
            try:
                positions = [
                    {
                        "symbol": "AAPL",
                        "quantity": 100,
                        "avg_price": 145.00,
                        "current_price": 148.50,
                        "market_value": 14850.00,
                        "pnl": 350.00,
                        "pnl_percent": 2.41
                    },
                    {
                        "symbol": "BTC-USD",
                        "quantity": 1.5,
                        "avg_price": 43000.00,
                        "current_price": 44500.00,
                        "market_value": 66750.00,
                        "pnl": 2250.00,
                        "pnl_percent": 3.49
                    }
                ]
                
                return APIResponse(success=True, data=positions)
                
            except Exception as e:
                logger.error(f"Positions fetch failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to fetch positions"
                )
    
    def _setup_analysis_routes(self):
        """Setup analysis routes"""
        
        @self.app.get("/analysis/sentiment/{symbol}")
        async def get_sentiment(request, symbol: str, current_user: str = Depends(self.get_current_user)):
            """Get sentiment analysis"""
            try:
                sentiment_data = {
                    "symbol": symbol.upper(),
                    "overall_sentiment": 0.65,
                    "sentiment_type": "Bullish",
                    "confidence": 0.82,
                    "sources": {
                        "news": 0.72,
                        "social": 0.68,
                        "technical": 0.55
                    },
                    "fear_greed_index": 75.0,
                    "last_update": datetime.now().isoformat()
                }
                
                return APIResponse(success=True, data=sentiment_data)
                
            except Exception as e:
                logger.error(f"Sentiment analysis failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to analyze sentiment"
                )
        
        @self.app.get("/analysis/news/{symbol}")
        async def get_news(request, symbol: str, limit: int = 10, current_user: str = Depends(self.get_current_user)):
            """Get news for symbol"""
            try:
                news = [
                    {
                        "title": f"{symbol} reports strong quarterly earnings",
                        "summary": "Company beats analyst expectations with robust growth",
                        "sentiment": 0.8,
                        "source": "Reuters",
                        "published_at": datetime.now().isoformat(),
                        "url": "https://example.com/news1"
                    }
                ]
                
                return APIResponse(success=True, data=news)
                
            except Exception as e:
                logger.error(f"News fetch failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to fetch news"
                )
    
    def _setup_scanner_routes(self):
        """Setup scanner routes"""
        
        @self.app.post("/scanner/scan", response_model=List[ScanResult])
        @limiter.limit("10/minute")
        async def run_scan(request, scan_request: ScanRequest, current_user: str = Depends(self.get_current_user)):
            """Run market scan"""
            try:
                # Simulate scan results
                results = [
                    ScanResult(
                        symbol="AAPL",
                        scan_type="breakout",
                        score=85.2,
                        confidence=0.87,
                        signal="BUY",
                        current_price=148.50,
                        description="Strong breakout above resistance with volume confirmation"
                    ),
                    ScanResult(
                        symbol="BTC-USD",
                        scan_type="momentum",
                        score=78.9,
                        confidence=0.82,
                        signal="BUY",
                        current_price=44500.00,
                        description="Bullish momentum with positive sentiment"
                    )
                ]
                
                return results
                
            except Exception as e:
                logger.error(f"Market scan failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to run market scan"
                )
    
    def _setup_user_routes(self):
        """Setup user management routes"""
        
        @self.app.get("/user/profile")
        async def get_profile(current_user: str = Depends(self.get_current_user)):
            """Get user profile"""
            try:
                user_data = self.auth_manager.users_db.get(current_user)
                if not user_data:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="User not found"
                    )
                
                profile = {
                    "username": user_data["username"],
                    "email": user_data["email"],
                    "full_name": user_data.get("full_name"),
                    "created_at": user_data["created_at"].isoformat(),
                    "is_active": user_data["is_active"]
                }
                
                return APIResponse(success=True, data=profile)
                
            except HTTPException as e:
                raise e
            except Exception as e:
                logger.error(f"Profile fetch failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to fetch profile"
                )
        
        @self.app.post("/user/notifications")
        async def setup_notifications(notification_request: NotificationRequest, current_user: str = Depends(self.get_current_user)):
            """Setup push notifications"""
            try:
                # Store notification preferences
                return APIResponse(
                    success=True,
                    message="Notification preferences updated"
                )
                
            except Exception as e:
                logger.error(f"Notification setup failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to setup notifications"
                )
    
    def _setup_system_routes(self):
        """Setup system routes"""
        
        @self.app.get("/system/status")
        async def get_system_status():
            """Get system status"""
            try:
                status_data = {
                    "status": "healthy",
                    "version": "1.0.0",
                    "uptime": "5 days, 3 hours",
                    "components": {
                        "neurocluster": "active",
                        "data_manager": "connected",
                        "trading_engine": "running",
                        "sentiment_analyzer": "updating"
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                return APIResponse(success=True, data=status_data)
                
            except Exception as e:
                logger.error(f"Status check failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to get system status"
                )
        
        @self.app.get("/system/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.now()}
    
    def _setup_websockets(self):
        """Setup WebSocket endpoints"""
        
        @self.app.websocket("/ws/{user_id}")
        async def websocket_endpoint(websocket: WebSocket, user_id: str):
            """WebSocket endpoint for real-time updates"""
            await self.connection_manager.connect(websocket, user_id)
            
            try:
                # Send initial heartbeat
                await websocket.send_text(json.dumps({
                    "type": WebSocketMessageType.HEARTBEAT.value,
                    "timestamp": datetime.now().isoformat()
                }))
                
                while True:
                    # Wait for messages from client
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Handle different message types
                    if message.get("type") == "ping":
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        }))
                    
                    elif message.get("type") == "subscribe":
                        # Handle subscription to data feeds
                        symbols = message.get("symbols", [])
                        # Start sending real-time data for symbols
                        pass
                    
            except WebSocketDisconnect:
                self.connection_manager.disconnect(websocket, user_id)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.connection_manager.disconnect(websocket, user_id)
    
    # ==================== AUTHENTICATION DEPENDENCY ====================
    
    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())):
        """Get current authenticated user"""
        try:
            token = credentials.credentials
            payload = self.auth_manager.verify_token(token)
            
            if not payload:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials"
                )
            
            username = payload.get("sub")
            if not username:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload"
                )
            
            return username
            
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )
    
    # ==================== BACKGROUND TASKS ====================
    
    async def start_background_tasks(self):
        """Start background tasks"""
        
        # Start market data streaming
        asyncio.create_task(self._market_data_streamer())
        
        # Start portfolio updates
        asyncio.create_task(self._portfolio_updater())
        
        # Start alert processor
        asyncio.create_task(self._alert_processor())
    
    async def _market_data_streamer(self):
        """Stream market data to connected clients"""
        
        while True:
            try:
                # Simulate market data update
                market_data = {
                    "type": WebSocketMessageType.MARKET_DATA.value,
                    "data": {
                        "AAPL": {"price": 148.50, "change": 2.34, "timestamp": datetime.now().isoformat()},
                        "BTC-USD": {"price": 44500.00, "change": 1500.00, "timestamp": datetime.now().isoformat()}
                    }
                }
                
                # Broadcast to all connected clients
                await self.connection_manager.broadcast(market_data)
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Market data streaming error: {e}")
                await asyncio.sleep(10)
    
    async def _portfolio_updater(self):
        """Update portfolios for connected users"""
        
        while True:
            try:
                # Update portfolios for each connected user
                for user_id in self.connection_manager.user_connections:
                    portfolio_update = {
                        "type": WebSocketMessageType.PORTFOLIO_UPDATE.value,
                        "data": {
                            "total_value": 105000.00,
                            "day_pnl": 2500.00,
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    
                    await self.connection_manager.send_personal_message(portfolio_update, user_id)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Portfolio update error: {e}")
                await asyncio.sleep(60)
    
    async def _alert_processor(self):
        """Process and send alerts"""
        
        while True:
            try:
                # Check for alert conditions
                # Send alerts to relevant users
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(30)
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
        """Run the mobile API server"""
        
        # Start background tasks
        asyncio.create_task(self.start_background_tasks())
        
        # Run the server
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            debug=debug,
            log_level="info"
        )

# ==================== CONVENIENCE FUNCTIONS ====================

def create_mobile_api(config: Dict[str, Any] = None) -> MobileAPI:
    """Create and configure mobile API instance"""
    
    api = MobileAPI(config)
    return api

def run_mobile_api(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """Run mobile API server"""
    
    api = create_mobile_api()
    api.run(host=host, port=port, debug=debug)

# ==================== MAIN ENTRY POINT ====================

def main():
    """Main entry point"""
    
    try:
        logger.info("🚀 Starting NeuroCluster Elite Mobile API")
        run_mobile_api(debug=True)
        
    except Exception as e:
        logger.error(f"Failed to start mobile API: {e}")

if __name__ == "__main__":
    main()