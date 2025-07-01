# 📚 NeuroCluster Elite API Reference

Complete API documentation for the NeuroCluster Elite Trading Platform. This REST API provides programmatic access to all trading, analysis, and portfolio management features with 99.59% algorithm efficiency.

## 📋 Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Base URLs](#base-urls)
- [Rate Limiting](#rate-limiting)
- [Error Handling](#error-handling)
- [Health & Status](#health--status)
- [Portfolio Management](#portfolio-management)
- [Trading Operations](#trading-operations)
- [Market Analysis](#market-analysis)
- [Algorithm & Signals](#algorithm--signals)
- [Risk Management](#risk-management)
- [User Management](#user-management)
- [System Administration](#system-administration)
- [WebSocket Streams](#websocket-streams)
- [SDKs & Libraries](#sdks--libraries)
- [Examples](#examples)

---

## 🌐 Overview

The NeuroCluster Elite API is a RESTful service built with FastAPI that provides real-time access to advanced algorithmic trading capabilities. The API follows OpenAPI 3.0 specifications and supports both synchronous and asynchronous operations.

### Key Features

- **Real-time Trading**: Execute trades with millisecond precision
- **Advanced Analytics**: 50+ technical indicators and market analysis
- **Portfolio Management**: Complete portfolio tracking and optimization
- **Risk Controls**: Multi-layered risk management system
- **WebSocket Streams**: Real-time data feeds and notifications
- **High Performance**: 99.59% algorithm efficiency, 0.045ms processing time

### API Characteristics

| Feature | Specification |
|---------|---------------|
| **Base Protocol** | HTTPS/WSS |
| **Data Format** | JSON |
| **Authentication** | JWT Bearer Token |
| **Rate Limiting** | 1000 requests/minute |
| **Uptime SLA** | 99.9% |
| **Response Time** | < 100ms (95th percentile) |

---

## 🔐 Authentication

The API uses JWT (JSON Web Token) authentication with Bearer token authorization.

### Authentication Flow

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "your_username",
  "password": "your_password"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

### Using Authentication

Include the Bearer token in all API requests:

```http
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

### Token Refresh

```http
POST /api/v1/auth/refresh
Authorization: Bearer <refresh_token>
```

---

## 🌍 Base URLs

| Environment | Base URL | Description |
|-------------|----------|-------------|
| **Production** | `https://api.neurocluster.elite` | Production API |
| **Staging** | `https://staging-api.neurocluster.elite` | Staging environment |
| **Local** | `http://localhost:8000` | Local development |

### API Versioning

The API uses URL versioning:
- Current version: `v1`
- Full path: `/api/v1/endpoint`

---

## ⚡ Rate Limiting

Rate limits are enforced per user/API key:

| Endpoint Type | Rate Limit | Window |
|---------------|------------|--------|
| **General** | 1000 requests | 1 minute |
| **Trading** | 100 requests | 1 minute |
| **Market Data** | 2000 requests | 1 minute |
| **WebSocket** | 10 connections | Per user |

### Rate Limit Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

### Rate Limit Exceeded

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json

{
  "error": "rate_limit_exceeded",
  "message": "Too many requests",
  "retry_after": 60
}
```

---

## ❌ Error Handling

The API uses conventional HTTP status codes and returns detailed error information in JSON format.

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| **200** | OK | Request successful |
| **201** | Created | Resource created |
| **400** | Bad Request | Invalid request parameters |
| **401** | Unauthorized | Authentication required |
| **403** | Forbidden | Insufficient permissions |
| **404** | Not Found | Resource not found |
| **429** | Too Many Requests | Rate limit exceeded |
| **500** | Internal Server Error | Server error |
| **503** | Service Unavailable | Service temporarily unavailable |

### Error Response Format

```json
{
  "error": "validation_error",
  "message": "Invalid input parameters",
  "details": {
    "field": "symbol",
    "issue": "Symbol 'INVALID' is not supported"
  },
  "timestamp": "2025-06-30T10:30:00Z",
  "request_id": "req_123456789"
}
```

---

## 🏥 Health & Status

### Health Check

Check API health and status.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-06-30T10:30:00Z",
  "version": "1.0.0",
  "uptime_seconds": 86400,
  "algorithm_efficiency": 99.59,
  "database_status": "healthy",
  "cache_status": "healthy"
}
```

### System Status

Get detailed system status and metrics.

```http
GET /api/v1/system/status
Authorization: Bearer <token>
```

**Response:**
```json
{
  "system": {
    "cpu_usage": 25.3,
    "memory_usage": 67.8,
    "disk_usage": 45.2,
    "active_connections": 42
  },
  "algorithm": {
    "efficiency": 99.59,
    "processing_time_ms": 0.045,
    "accuracy": 94.7,
    "memory_usage_mb": 12.4
  },
  "trading": {
    "active_positions": 15,
    "pending_orders": 3,
    "daily_pnl": 2456.78,
    "total_trades_today": 23
  }
}
```

---

## 💼 Portfolio Management

### Get Portfolio Summary

Retrieve current portfolio status and performance metrics.

```http
GET /api/v1/portfolio/summary
Authorization: Bearer <token>
```

**Response:**
```json
{
  "total_value": 125750.50,
  "cash_balance": 25750.50,
  "invested_amount": 100000.00,
  "unrealized_pnl": 3250.75,
  "realized_pnl": 1250.25,
  "daily_pnl": 890.50,
  "total_return_pct": 4.75,
  "position_count": 12,
  "risk_level": "moderate",
  "max_drawdown": -2.3,
  "sharpe_ratio": 1.85,
  "last_updated": "2025-06-30T10:30:00Z"
}
```

### Get Positions

List all current positions.

```http
GET /api/v1/portfolio/positions
Authorization: Bearer <token>
```

**Query Parameters:**
- `status` (optional): `open`, `closed`, `all`
- `symbol` (optional): Filter by symbol
- `limit` (optional): Number of results (default: 50)
- `offset` (optional): Pagination offset (default: 0)

**Response:**
```json
{
  "positions": [
    {
      "id": "pos_123456",
      "symbol": "AAPL",
      "asset_type": "stock",
      "side": "long",
      "quantity": 100,
      "entry_price": 150.25,
      "current_price": 155.50,
      "market_value": 15550.00,
      "unrealized_pnl": 525.00,
      "unrealized_pnl_pct": 3.49,
      "stop_loss": 142.74,
      "take_profit": 165.28,
      "opened_at": "2025-06-29T14:30:00Z",
      "strategy": "neurocluster_momentum",
      "risk_score": 3.2
    }
  ],
  "total_count": 12,
  "pagination": {
    "limit": 50,
    "offset": 0,
    "has_more": false
  }
}
```

### Get Position Details

Get detailed information about a specific position.

```http
GET /api/v1/portfolio/positions/{position_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "id": "pos_123456",
  "symbol": "AAPL",
  "asset_type": "stock",
  "side": "long",
  "quantity": 100,
  "entry_price": 150.25,
  "current_price": 155.50,
  "market_value": 15550.00,
  "unrealized_pnl": 525.00,
  "unrealized_pnl_pct": 3.49,
  "stop_loss": 142.74,
  "take_profit": 165.28,
  "opened_at": "2025-06-29T14:30:00Z",
  "strategy": "neurocluster_momentum",
  "risk_score": 3.2,
  "entry_signals": [
    {
      "timestamp": "2025-06-29T14:29:45Z",
      "signal_type": "BUY",
      "confidence": 0.87,
      "regime": "Bull Market",
      "reasoning": "Strong momentum breakout with high confidence"
    }
  ],
  "trade_history": [
    {
      "timestamp": "2025-06-29T14:30:00Z",
      "action": "BUY",
      "quantity": 100,
      "price": 150.25,
      "commission": 0.00
    }
  ]
}
```

### Portfolio Performance

Get detailed portfolio performance analytics.

```http
GET /api/v1/portfolio/performance
Authorization: Bearer <token>
```

**Query Parameters:**
- `period`: `1d`, `1w`, `1m`, `3m`, `6m`, `1y`, `all`
- `include_benchmark`: Include market benchmark comparison

**Response:**
```json
{
  "period": "1m",
  "start_date": "2025-05-30T00:00:00Z",
  "end_date": "2025-06-30T00:00:00Z",
  "total_return": 4.75,
  "annualized_return": 57.0,
  "volatility": 18.5,
  "sharpe_ratio": 1.85,
  "sortino_ratio": 2.34,
  "max_drawdown": -2.3,
  "win_rate": 67.5,
  "profit_factor": 1.89,
  "daily_returns": [
    {
      "date": "2025-06-29",
      "return": 0.85,
      "benchmark_return": 0.45
    }
  ],
  "benchmark_comparison": {
    "benchmark": "SPY",
    "outperformance": 2.1,
    "correlation": 0.78,
    "beta": 1.15
  }
}
```

---

## 📈 Trading Operations

### Place Order

Execute a trading order.

```http
POST /api/v1/trading/orders
Authorization: Bearer <token>
Content-Type: application/json

{
  "symbol": "AAPL",
  "side": "BUY",
  "quantity": 100,
  "order_type": "MARKET",
  "price": null,
  "stop_loss": 142.00,
  "take_profit": 165.00,
  "time_in_force": "DAY",
  "strategy": "manual"
}
```

**Order Types:**
- `MARKET`: Execute at current market price
- `LIMIT`: Execute at specified price or better
- `STOP`: Stop-loss order
- `STOP_LIMIT`: Stop-limit order

**Response:**
```json
{
  "order_id": "ord_789123456",
  "status": "submitted",
  "symbol": "AAPL",
  "side": "BUY",
  "quantity": 100,
  "order_type": "MARKET",
  "estimated_price": 155.50,
  "estimated_commission": 0.00,
  "submitted_at": "2025-06-30T10:30:00Z",
  "expected_execution": "immediate"
}
```

### Get Orders

List trading orders.

```http
GET /api/v1/trading/orders
Authorization: Bearer <token>
```

**Query Parameters:**
- `status`: `pending`, `filled`, `cancelled`, `rejected`, `all`
- `symbol`: Filter by symbol
- `from_date`: Start date (ISO 8601)
- `to_date`: End date (ISO 8601)
- `limit`: Number of results
- `offset`: Pagination offset

**Response:**
```json
{
  "orders": [
    {
      "order_id": "ord_789123456",
      "symbol": "AAPL",
      "side": "BUY",
      "quantity": 100,
      "order_type": "MARKET",
      "status": "filled",
      "filled_quantity": 100,
      "average_price": 155.48,
      "commission": 0.00,
      "submitted_at": "2025-06-30T10:30:00Z",
      "filled_at": "2025-06-30T10:30:02Z"
    }
  ],
  "total_count": 1,
  "pagination": {
    "limit": 50,
    "offset": 0,
    "has_more": false
  }
}
```

### Cancel Order

Cancel a pending order.

```http
DELETE /api/v1/trading/orders/{order_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "order_id": "ord_789123456",
  "status": "cancelled",
  "cancelled_at": "2025-06-30T10:35:00Z",
  "message": "Order cancelled successfully"
}
```

### Trading Signals

Get current trading signals from the NeuroCluster algorithm.

```http
GET /api/v1/trading/signals
Authorization: Bearer <token>
```

**Query Parameters:**
- `symbols`: Comma-separated list of symbols
- `min_confidence`: Minimum signal confidence (0.0-1.0)
- `max_signals`: Maximum number of signals to return

**Response:**
```json
{
  "signals": [
    {
      "symbol": "AAPL",
      "signal_type": "BUY",
      "confidence": 0.87,
      "regime": "Bull Market",
      "entry_price": 155.50,
      "stop_loss": 147.73,
      "take_profit": 171.33,
      "position_size": 0.08,
      "risk_reward_ratio": 2.85,
      "reasoning": "Strong momentum breakout with regime confirmation",
      "timestamp": "2025-06-30T10:30:00Z",
      "expires_at": "2025-06-30T11:30:00Z"
    }
  ],
  "generated_at": "2025-06-30T10:30:00Z",
  "algorithm_efficiency": 99.59
}
```

---

## 📊 Market Analysis

### Get Market Data

Retrieve current market data for symbols.

```http
GET /api/v1/market/data
Authorization: Bearer <token>
```

**Query Parameters:**
- `symbols`: Comma-separated list of symbols (required)
- `include_indicators`: Include technical indicators
- `timeframe`: Data timeframe (`1m`, `5m`, `15m`, `1h`, `1d`)

**Response:**
```json
{
  "data": {
    "AAPL": {
      "symbol": "AAPL",
      "asset_type": "stock",
      "price": 155.50,
      "change": 2.25,
      "change_percent": 1.47,
      "volume": 45678923,
      "market_cap": 2850000000000,
      "bid": 155.49,
      "ask": 155.51,
      "high_24h": 156.75,
      "low_24h": 152.80,
      "timestamp": "2025-06-30T10:30:00Z",
      "indicators": {
        "rsi": 65.3,
        "macd": 1.25,
        "macd_signal": 0.95,
        "bollinger_upper": 158.45,
        "bollinger_lower": 151.22,
        "sma_20": 153.75,
        "ema_50": 151.90,
        "volatility": 22.5
      }
    }
  },
  "updated_at": "2025-06-30T10:30:00Z"
}
```

### Market Analysis

Perform comprehensive market analysis using NeuroCluster algorithm.

```http
POST /api/v1/market/analyze
Authorization: Bearer <token>
Content-Type: application/json

{
  "symbols": ["AAPL", "GOOGL", "MSFT"],
  "timeframe": "1h",
  "include_sentiment": true,
  "include_patterns": true,
  "analysis_depth": "comprehensive"
}
```

**Response:**
```json
{
  "analysis": {
    "AAPL": {
      "symbol": "AAPL",
      "regime": "Bull Market",
      "confidence": 0.87,
      "price": 155.50,
      "change_percent": 1.47,
      "technical_score": 78.5,
      "momentum_score": 85.2,
      "trend_strength": "strong",
      "support_levels": [152.25, 149.80, 147.50],
      "resistance_levels": [158.75, 162.40, 165.80],
      "sentiment": {
        "score": 0.65,
        "news_sentiment": 0.72,
        "social_sentiment": 0.58,
        "analyst_sentiment": 0.68
      },
      "patterns": [
        {
          "type": "bullish_flag",
          "confidence": 0.82,
          "target_price": 168.50,
          "timeline": "5-10 days"
        }
      ],
      "recommendation": "BUY",
      "risk_level": "medium"
    }
  },
  "market_overview": {
    "overall_sentiment": "bullish",
    "volatility_regime": "normal",
    "sector_rotation": "technology_outperforming",
    "risk_on_off": "risk_on"
  },
  "timestamp": "2025-06-30T10:30:00Z"
}
```

### Historical Data

Get historical market data.

```http
GET /api/v1/market/history/{symbol}
Authorization: Bearer <token>
```

**Query Parameters:**
- `timeframe`: `1m`, `5m`, `15m`, `1h`, `1d`, `1w`
- `from_date`: Start date (ISO 8601)
- `to_date`: End date (ISO 8601)
- `limit`: Number of data points (max: 10000)

**Response:**
```json
{
  "symbol": "AAPL",
  "timeframe": "1h",
  "data": [
    {
      "timestamp": "2025-06-30T09:00:00Z",
      "open": 153.25,
      "high": 154.80,
      "low": 152.90,
      "close": 154.50,
      "volume": 2345678,
      "vwap": 154.15
    }
  ],
  "count": 24,
  "from_date": "2025-06-29T10:00:00Z",
  "to_date": "2025-06-30T10:00:00Z"
}
```

---

## 🧠 Algorithm & Signals

### Algorithm Status

Get NeuroCluster algorithm performance metrics.

```http
GET /api/v1/algorithm/status
Authorization: Bearer <token>
```

**Response:**
```json
{
  "algorithm": "NeuroCluster Elite v2.0",
  "efficiency": 99.59,
  "processing_time_ms": 0.045,
  "accuracy": 94.7,
  "memory_usage_mb": 12.4,
  "clustering_quality": 0.918,
  "last_optimization": "2025-06-30T00:00:00Z",
  "uptime_hours": 168.5,
  "processed_datapoints_today": 1847293,
  "regime_changes_today": 3,
  "signals_generated_today": 47
}
```

### Regime Detection

Get current market regime analysis.

```http
GET /api/v1/algorithm/regime
Authorization: Bearer <token>
```

**Query Parameters:**
- `symbols`: Symbols to analyze
- `timeframe`: Analysis timeframe
- `include_history`: Include regime change history

**Response:**
```json
{
  "current_regime": {
    "type": "Bull Market",
    "confidence": 0.87,
    "since": "2025-06-28T14:30:00Z",
    "duration_hours": 44.0,
    "stability_score": 0.91
  },
  "regime_probabilities": {
    "bull_market": 0.87,
    "bear_market": 0.03,
    "sideways_market": 0.08,
    "high_volatility": 0.02
  },
  "supporting_factors": [
    "Strong momentum indicators",
    "Positive volume confirmation",
    "Bullish pattern recognition",
    "Favorable sentiment metrics"
  ],
  "risk_factors": [
    "Approaching resistance level",
    "Slight overbought condition"
  ],
  "regime_history": [
    {
      "regime": "Sideways Market",
      "start": "2025-06-25T09:00:00Z",
      "end": "2025-06-28T14:30:00Z",
      "duration_hours": 77.5
    }
  ]
}
```

### Pattern Recognition

Get detected chart patterns.

```http
GET /api/v1/algorithm/patterns
Authorization: Bearer <token>
```

**Query Parameters:**
- `symbols`: Symbols to analyze
- `pattern_types`: Specific pattern types to look for
- `min_confidence`: Minimum pattern confidence

**Response:**
```json
{
  "patterns": [
    {
      "symbol": "AAPL",
      "pattern_type": "bullish_flag",
      "confidence": 0.82,
      "detected_at": "2025-06-30T10:15:00Z",
      "start_point": "2025-06-29T14:00:00Z",
      "completion_target": 168.50,
      "target_probability": 0.75,
      "timeline_days": 5,
      "support_level": 152.25,
      "resistance_level": 158.75,
      "volume_confirmation": true,
      "reliability_score": 0.88
    }
  ],
  "pattern_summary": {
    "total_patterns": 1,
    "bullish_patterns": 1,
    "bearish_patterns": 0,
    "neutral_patterns": 0,
    "average_confidence": 0.82
  }
}
```

---

## ⚖️ Risk Management

### Risk Assessment

Get portfolio risk assessment.

```http
GET /api/v1/risk/assessment
Authorization: Bearer <token>
```

**Response:**
```json
{
  "overall_risk_score": 6.5,
  "risk_level": "moderate",
  "portfolio_beta": 1.15,
  "value_at_risk_1d": -2150.75,
  "value_at_risk_5d": -4825.50,
  "maximum_drawdown": -2.3,
  "concentration_risk": 0.25,
  "correlation_risk": 0.18,
  "sector_exposure": {
    "technology": 0.45,
    "healthcare": 0.20,
    "finance": 0.15,
    "consumer": 0.20
  },
  "risk_metrics": {
    "portfolio_volatility": 18.5,
    "tracking_error": 3.2,
    "information_ratio": 0.85,
    "sharpe_ratio": 1.85
  },
  "stress_tests": {
    "market_crash_scenario": -15.2,
    "interest_rate_spike": -8.7,
    "sector_rotation": -5.1
  }
}
```

### Risk Limits

Get and set risk limits.

```http
GET /api/v1/risk/limits
Authorization: Bearer <token>
```

**Response:**
```json
{
  "limits": {
    "max_portfolio_risk": 0.02,
    "max_position_size": 0.10,
    "max_sector_exposure": 0.50,
    "daily_loss_limit": 0.03,
    "maximum_drawdown": 0.10,
    "var_limit_1d": 0.025
  },
  "current_usage": {
    "portfolio_risk": 0.015,
    "largest_position": 0.08,
    "largest_sector": 0.45,
    "daily_loss": 0.012,
    "current_drawdown": 0.023
  },
  "violations": [],
  "warnings": [
    {
      "type": "sector_concentration",
      "message": "Technology sector approaching 50% limit",
      "current_value": 0.45,
      "limit": 0.50
    }
  ]
}
```

### Update Risk Limits

```http
PUT /api/v1/risk/limits
Authorization: Bearer <token>
Content-Type: application/json

{
  "max_position_size": 0.12,
  "daily_loss_limit": 0.025
}
```

---

## 👤 User Management

### User Profile

Get user profile information.

```http
GET /api/v1/user/profile
Authorization: Bearer <token>
```

**Response:**
```json
{
  "user_id": "user_123456",
  "username": "trader_john",
  "email": "john@example.com",
  "full_name": "John Trader",
  "account_type": "premium",
  "risk_profile": "moderate",
  "trading_experience": "intermediate",
  "account_created": "2025-01-15T10:00:00Z",
  "last_login": "2025-06-30T09:30:00Z",
  "preferences": {
    "default_risk_level": "moderate",
    "notification_settings": {
      "email_alerts": true,
      "push_notifications": true,
      "trade_confirmations": true
    },
    "dashboard_layout": "standard",
    "preferred_timeframe": "1h"
  }
}
```

### Update User Preferences

```http
PUT /api/v1/user/preferences
Authorization: Bearer <token>
Content-Type: application/json

{
  "default_risk_level": "aggressive",
  "notification_settings": {
    "email_alerts": false,
    "push_notifications": true
  }
}
```

---

## 🔧 System Administration

### System Metrics

Get detailed system performance metrics (admin only).

```http
GET /api/v1/admin/metrics
Authorization: Bearer <admin_token>
```

**Response:**
```json
{
  "system_performance": {
    "cpu_usage_percent": 25.3,
    "memory_usage_percent": 67.8,
    "disk_usage_percent": 45.2,
    "network_io_mbps": 15.7,
    "database_connections": 12,
    "cache_hit_ratio": 0.94
  },
  "api_metrics": {
    "requests_per_minute": 150,
    "average_response_time_ms": 45,
    "error_rate_percent": 0.02,
    "active_connections": 42
  },
  "trading_metrics": {
    "trades_today": 1247,
    "volume_today": 15750000,
    "active_strategies": 8,
    "algorithm_efficiency": 99.59
  }
}
```

### Database Status

Get database health and statistics.

```http
GET /api/v1/admin/database
Authorization: Bearer <admin_token>
```

**Response:**
```json
{
  "status": "healthy",
  "connection_pool": {
    "active_connections": 12,
    "idle_connections": 8,
    "max_connections": 50
  },
  "tables": [
    {
      "name": "trades",
      "row_count": 54321,
      "size_mb": 125.7
    },
    {
      "name": "market_data",
      "row_count": 1234567,
      "size_mb": 2456.8
    }
  ],
  "performance": {
    "average_query_time_ms": 12.5,
    "slow_queries_today": 3,
    "deadlocks_today": 0
  }
}
```

---

## 🔌 WebSocket Streams

Real-time data streams using WebSocket connections.

### Connection

```javascript
const ws = new WebSocket('wss://api.neurocluster.elite/ws/data');

// Authentication
ws.onopen = function() {
    ws.send(JSON.stringify({
        type: 'auth',
        token: 'your_jwt_token'
    }));
};
```

### Subscribe to Market Data

```javascript
ws.send(JSON.stringify({
    type: 'subscribe',
    channel: 'market_data',
    symbols: ['AAPL', 'GOOGL', 'MSFT']
}));
```

### Message Types

#### Market Data Updates
```json
{
  "type": "market_data",
  "symbol": "AAPL",
  "price": 155.50,
  "change": 2.25,
  "change_percent": 1.47,
  "volume": 45678923,
  "timestamp": "2025-06-30T10:30:00Z"
}
```

#### Trading Signals
```json
{
  "type": "trading_signal",
  "symbol": "AAPL",
  "signal_type": "BUY",
  "confidence": 0.87,
  "regime": "Bull Market",
  "entry_price": 155.50,
  "timestamp": "2025-06-30T10:30:00Z"
}
```

#### Order Updates
```json
{
  "type": "order_update",
  "order_id": "ord_789123456",
  "status": "filled",
  "filled_quantity": 100,
  "average_price": 155.48,
  "timestamp": "2025-06-30T10:30:02Z"
}
```

#### Portfolio Updates
```json
{
  "type": "portfolio_update",
  "total_value": 125750.50,
  "daily_pnl": 890.50,
  "unrealized_pnl": 3250.75,
  "timestamp": "2025-06-30T10:30:00Z"
}
```

---

## 📚 SDKs & Libraries

### Python SDK

```bash
pip install neurocluster-elite-sdk
```

```python
from neurocluster_elite import NeuroClusterAPI

# Initialize client
client = NeuroClusterAPI(
    api_key="your_api_key",
    base_url="https://api.neurocluster.elite"
)

# Get portfolio
portfolio = client.portfolio.get_summary()
print(f"Total value: ${portfolio.total_value:,.2f}")

# Place order
order = client.trading.place_order(
    symbol="AAPL",
    side="BUY",
    quantity=100,
    order_type="MARKET"
)
print(f"Order placed: {order.order_id}")

# Get trading signals
signals = client.trading.get_signals(symbols=["AAPL", "GOOGL"])
for signal in signals:
    print(f"{signal.symbol}: {signal.signal_type} ({signal.confidence:.0%})")
```

### JavaScript/Node.js SDK

```bash
npm install neurocluster-elite-sdk
```

```javascript
const { NeuroClusterAPI } = require('neurocluster-elite-sdk');

// Initialize client
const client = new NeuroClusterAPI({
    apiKey: 'your_api_key',
    baseUrl: 'https://api.neurocluster.elite'
});

// Get portfolio
const portfolio = await client.portfolio.getSummary();
console.log(`Total value: $${portfolio.totalValue.toLocaleString()}`);

// Place order
const order = await client.trading.placeOrder({
    symbol: 'AAPL',
    side: 'BUY',
    quantity: 100,
    orderType: 'MARKET'
});
console.log(`Order placed: ${order.orderId}`);

// Subscribe to real-time data
client.websocket.subscribe('market_data', ['AAPL', 'GOOGL']);
client.websocket.on('market_data', (data) => {
    console.log(`${data.symbol}: $${data.price}`);
});
```

---

## 💡 Examples

### Complete Trading Workflow

```python
import neurocluster_elite as nc

# Initialize client
client = nc.NeuroClusterAPI(api_key="your_key")

# 1. Get market analysis
analysis = client.market.analyze(['AAPL', 'GOOGL', 'MSFT'])

# 2. Get trading signals
signals = client.trading.get_signals(min_confidence=0.8)

# 3. Filter by analysis and risk
for signal in signals:
    symbol_analysis = analysis[signal.symbol]
    
    if (signal.confidence > 0.8 and 
        symbol_analysis.risk_level in ['low', 'medium']):
        
        # 4. Calculate position size
        portfolio = client.portfolio.get_summary()
        position_size = min(
            portfolio.total_value * 0.1,  # Max 10% per position
            signal.position_size * portfolio.total_value
        )
        quantity = int(position_size / signal.entry_price)
        
        # 5. Place order with risk controls
        order = client.trading.place_order(
            symbol=signal.symbol,
            side=signal.signal_type,
            quantity=quantity,
            order_type="LIMIT",
            price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )
        
        print(f"Order placed: {order.order_id}")

# 6. Monitor positions
positions = client.portfolio.get_positions()
for position in positions:
    if position.unrealized_pnl_pct < -0.05:  # 5% loss
        print(f"Position {position.symbol} down {position.unrealized_pnl_pct:.1%}")
```

### Risk Management Example

```python
# Set up comprehensive risk controls
risk_limits = {
    "max_position_size": 0.08,      # 8% max per position
    "max_sector_exposure": 0.40,    # 40% max per sector
    "daily_loss_limit": 0.02,       # 2% daily loss limit
    "var_limit_1d": 0.025          # 2.5% VaR limit
}

client.risk.update_limits(risk_limits)

# Monitor risk in real-time
def risk_monitor():
    assessment = client.risk.get_assessment()
    
    # Check for violations
    if assessment.daily_loss > risk_limits["daily_loss_limit"]:
        print("ALERT: Daily loss limit exceeded!")
        # Close all positions or reduce exposure
        
    if assessment.overall_risk_score > 8:
        print("WARNING: High portfolio risk detected")
        # Implement risk reduction measures

# Run risk monitor every minute
import schedule
schedule.every(1).minutes.do(risk_monitor)
```

---

## 🚨 Rate Limiting Best Practices

1. **Implement exponential backoff** when receiving 429 responses
2. **Cache frequently accessed data** like market data and portfolio info
3. **Use WebSocket streams** for real-time data instead of polling
4. **Batch API calls** when possible to reduce request count
5. **Monitor rate limit headers** to avoid hitting limits

## 🔒 Security Best Practices

1. **Never expose API keys** in client-side code
2. **Use HTTPS only** for all API communications
3. **Rotate API keys regularly** (monthly recommended)
4. **Implement request signing** for additional security
5. **Use IP whitelisting** when available
6. **Monitor API access logs** for suspicious activity

## 📞 Support

- **Documentation**: https://docs.neurocluster.elite
- **API Status**: https://status.neurocluster.elite
- **Support Email**: api-support@neurocluster.elite
- **Developer Forum**: https://community.neurocluster.elite

---

*This documentation is for NeuroCluster Elite API v1.0. Last updated: 2025-06-30*