# 📚 NeuroCluster Elite API Reference

This comprehensive API reference covers all public interfaces, endpoints, and methods available in the NeuroCluster Elite trading platform.

## 🎯 Table of Contents

- [Core Algorithm API](#core-algorithm-api)
- [Trading Engine API](#trading-engine-api)
- [Data Management API](#data-management-api)
- [Strategy API](#strategy-api)
- [Risk Management API](#risk-management-api)
- [Integration APIs](#integration-apis)
- [WebSocket Streams](#websocket-streams)
- [Authentication](#authentication)
- [Error Codes](#error-codes)
- [Rate Limits](#rate-limits)

---

## 🧠 Core Algorithm API

### NeuroCluster Elite Algorithm

The core NeuroCluster algorithm with 99.59% efficiency and 0.045ms processing time.

#### `NeuroClusterElite` Class

```python
from src.core.neurocluster_elite import NeuroClusterElite, RegimeType, AssetType

# Initialize
neurocluster = NeuroClusterElite(config={
    'similarity_threshold': 0.75,
    'learning_rate': 0.14,
    'decay_rate': 0.02,
    'max_clusters': 12
})

# Detect market regime
regime, confidence = neurocluster.detect_regime(market_data)
```

#### Methods

##### `detect_regime(market_data: Dict[str, MarketData]) -> Tuple[RegimeType, float]`

Detects current market regime using the proven NeuroCluster algorithm.

**Parameters:**
- `market_data`: Dictionary mapping symbols to MarketData objects

**Returns:**
- `Tuple[RegimeType, float]`: Detected regime and confidence level (0.0-1.0)

**Example:**
```python
market_data = {
    'AAPL': MarketData(
        symbol='AAPL',
        asset_type=AssetType.STOCK,
        price=150.25,
        volume=1500000,
        timestamp=datetime.now()
    )
}

regime, confidence = neurocluster.detect_regime(market_data)
print(f"Regime: {regime.value}, Confidence: {confidence:.2%}")
```

##### `get_performance_metrics() -> Dict[str, Any]`

Returns algorithm performance metrics.

**Returns:**
- `Dict[str, Any]`: Performance metrics including efficiency, processing time, etc.

**Example:**
```python
metrics = neurocluster.get_performance_metrics()
{
    'efficiency_rate': 99.59,
    'avg_processing_time_ms': 0.045,
    'total_processed': 50000,
    'cluster_count': 8,
    'memory_usage_mb': 12.4
}
```

##### `save_state(file_path: str) -> None`

Saves algorithm state to disk for persistence.

##### `load_state(file_path: str) -> None`

Loads previously saved algorithm state.

---

## 💹 Trading Engine API

### Advanced Trading Engine

Multi-asset trading engine with AI-powered strategy selection.

#### `AdvancedTradingEngine` Class

```python
from src.trading.trading_engine import AdvancedTradingEngine

engine = AdvancedTradingEngine(
    neurocluster=neurocluster,
    data_manager=data_manager,
    config={
        'trading_mode': 'PAPER',
        'initial_capital': 100000.0,
        'max_positions': 10
    }
)
```

#### Methods

##### `start_trading() -> Dict[str, Any]`

Starts the trading engine.

**Returns:**
- `Dict[str, Any]`: Status response

**Example:**
```python
result = engine.start_trading()
{
    'status': 'success',
    'message': 'Trading engine started',
    'mode': 'PAPER',
    'timestamp': '2025-06-30T10:00:00Z'
}
```

##### `stop_trading() -> Dict[str, Any]`

Stops the trading engine and closes all positions.

##### `run_trading_cycle() -> Dict[str, Any]`

Executes a single trading cycle.

**Returns:**
- `Dict[str, Any]`: Cycle results including regime, signals, and performance

**Example:**
```python
result = await engine.run_trading_cycle()
{
    'status': 'success',
    'regime': 'BULL',
    'confidence': 0.85,
    'signals_generated': 3,
    'signals_executed': 2,
    'portfolio_value': 105000.0,
    'cycle_time_ms': 45.2
}
```

##### `get_portfolio_status() -> Dict[str, Any]`

Returns current portfolio status.

**Example:**
```python
status = engine.get_portfolio_status()
{
    'total_value': 105000.0,
    'cash_balance': 25000.0,
    'positions': [
        {
            'symbol': 'AAPL',
            'quantity': 100,
            'market_value': 15025.0,
            'unrealized_pnl': 525.0,
            'unrealized_pnl_pct': 3.61
        }
    ],
    'total_pnl': 5000.0,
    'total_pnl_pct': 5.0
}
```

##### `place_order(order_request: Dict[str, Any]) -> Dict[str, Any]`

Places a trading order.

**Parameters:**
```python
order_request = {
    'symbol': 'AAPL',
    'side': 'BUY',  # 'BUY' or 'SELL'
    'quantity': 100,
    'order_type': 'MARKET',  # 'MARKET', 'LIMIT', 'STOP_LOSS'
    'price': None,  # Required for LIMIT orders
    'time_in_force': 'DAY'  # 'DAY', 'GTC', 'IOC'
}
```

**Returns:**
```python
{
    'order_id': 'ORD-12345',
    'status': 'FILLED',
    'symbol': 'AAPL',
    'quantity': 100,
    'fill_price': 150.25,
    'timestamp': '2025-06-30T10:15:00Z'
}
```

---

## 📊 Data Management API

### Multi-Asset Data Manager

Unified data management for stocks, crypto, forex, and commodities.

#### `MultiAssetDataManager` Class

```python
from src.data.multi_asset_manager import MultiAssetDataManager

data_manager = MultiAssetDataManager(config={
    'cache_enabled': True,
    'cache_ttl_seconds': 300,
    'rate_limit_calls': 100
})
```

#### Methods

##### `fetch_market_data(symbols: List[str], asset_types: Dict[str, AssetType]) -> Dict[str, MarketData]`

Fetches current market data for multiple symbols.

**Parameters:**
- `symbols`: List of symbols to fetch
- `asset_types`: Mapping of symbols to asset types

**Example:**
```python
symbols = ['AAPL', 'BTC-USD', 'EUR/USD']
asset_types = {
    'AAPL': AssetType.STOCK,
    'BTC-USD': AssetType.CRYPTO,
    'EUR/USD': AssetType.FOREX
}

data = await data_manager.fetch_market_data(symbols, asset_types)
```

##### `get_historical_data(symbol: str, asset_type: AssetType, period: str) -> pd.DataFrame`

Fetches historical price data.

**Parameters:**
- `symbol`: Symbol to fetch
- `asset_type`: Type of asset
- `period`: Time period ('1d', '5d', '1mo', '1y', etc.)

**Returns:**
- `pd.DataFrame`: Historical OHLCV data

---

## 🎯 Strategy API

### Trading Strategies

AI-powered trading strategies for different market regimes.

#### Base Strategy Interface

```python
from src.trading.strategies.base_strategy import BaseStrategy, TradingSignal

class CustomStrategy(BaseStrategy):
    def generate_signal(self, data: MarketData, regime: RegimeType, confidence: float) -> Optional[TradingSignal]:
        # Strategy implementation
        pass
```

#### Pre-built Strategies

##### Bull Market Strategy
```python
from src.trading.strategies.bull_strategy import BullMarketStrategy

strategy = BullMarketStrategy(config={
    'min_rsi': 50,
    'min_momentum': 0.02,
    'volume_threshold': 1.5
})
```

##### Volatility Strategy
```python
from src.trading.strategies.volatility_strategy import AdvancedVolatilityStrategy

strategy = AdvancedVolatilityStrategy(config={
    'volatility_threshold': 0.3,
    'breakout_factor': 1.2
})
```

#### Strategy Selection API

```python
from src.trading.strategy_selector import StrategySelector

selector = StrategySelector()
strategies = selector.select_strategies(
    regime=RegimeType.BULL,
    asset_types={'AAPL': AssetType.STOCK}
)
```

---

## ⚖️ Risk Management API

### Risk Manager

Advanced risk management with Kelly Criterion and portfolio optimization.

#### `RiskManager` Class

```python
from src.trading.risk_manager import RiskManager

risk_manager = RiskManager(config={
    'max_portfolio_risk': 0.02,
    'max_position_size': 0.1,
    'max_drawdown': 0.15
})
```

#### Methods

##### `calculate_position_size(signal: TradingSignal, portfolio_value: float) -> float`

Calculates optimal position size using Kelly Criterion.

##### `validate_signals(signals: List[TradingSignal], portfolio_value: float, positions: Dict) -> List[TradingSignal]`

Validates and filters trading signals based on risk constraints.

##### `get_risk_metrics() -> Dict[str, float]`

Returns current portfolio risk metrics.

**Example:**
```python
metrics = risk_manager.get_risk_metrics()
{
    'portfolio_beta': 1.15,
    'portfolio_volatility': 0.18,
    'var_95': 0.025,
    'expected_shortfall': 0.035,
    'sharpe_ratio': 1.8,
    'max_drawdown': 0.08
}
```

---

## 🔗 Integration APIs

### Broker Integrations

#### Alpaca Trading API

```python
from src.integrations.brokers.alpaca import AlpacaTrading

alpaca = AlpacaTrading(config={
    'api_key': 'your_api_key',
    'secret_key': 'your_secret_key',
    'base_url': 'https://paper-api.alpaca.markets'
})

# Get account info
account = await alpaca.get_account_info()

# Place order
order = await alpaca.place_order(
    symbol='AAPL',
    qty=100,
    side='buy',
    type='market'
)
```

#### Interactive Brokers API

```python
from src.integrations.brokers.interactive_brokers import InteractiveBrokersAPI

ib = InteractiveBrokersAPI(config={
    'host': 'localhost',
    'port': 7497,
    'client_id': 1
})

# Connect
await ib.connect_async()

# Create contract
contract = ib.create_stock_contract('AAPL')

# Place order
order = ib.create_market_order('BUY', 100)
ib.placeOrder(ib.nextOrderId(), contract, order)
```

### Exchange Integrations

#### Binance API

```python
from src.integrations.exchanges.binance import BinanceExchange

binance = BinanceExchange(config={
    'api_key': 'your_api_key',
    'secret_key': 'your_secret_key',
    'testnet': True
})

# Get ticker
ticker = await binance.get_ticker('BTCUSDT')

# Place order
order = await binance.place_order(
    symbol='BTCUSDT',
    side='BUY',
    type='MARKET',
    quantity=0.001
)
```

---

## 🌐 WebSocket Streams

### Real-time Data Streams

#### Market Data Stream

```python
# WebSocket endpoint
wss://api.neurocluster-elite.com/ws/market-data

# Subscribe to symbols
{
    "action": "subscribe",
    "symbols": ["AAPL", "BTC-USD"],
    "channels": ["ticker", "orderbook"]
}

# Market data updates
{
    "channel": "ticker",
    "symbol": "AAPL",
    "data": {
        "price": 150.25,
        "change": 2.75,
        "volume": 1500000,
        "timestamp": "2025-06-30T10:15:00Z"
    }
}
```

#### Trading Signals Stream

```python
# WebSocket endpoint
wss://api.neurocluster-elite.com/ws/signals

# Signal updates
{
    "channel": "signals",
    "data": {
        "symbol": "AAPL",
        "signal_type": "BUY",
        "confidence": 0.85,
        "entry_price": 150.25,
        "stop_loss": 142.75,
        "take_profit": 165.00,
        "timestamp": "2025-06-30T10:15:00Z"
    }
}
```

---

## 🔐 Authentication

### API Key Authentication

Include API key in request headers:

```http
GET /api/v1/portfolio
Authorization: Bearer your_api_key_here
Content-Type: application/json
```

### JWT Token Authentication

For web dashboard and mobile apps:

```http
POST /api/v1/auth/login
Content-Type: application/json

{
    "username": "your_username",
    "password": "your_password"
}

Response:
{
    "access_token": "jwt_token_here",
    "token_type": "bearer",
    "expires_in": 3600
}
```

Use token in subsequent requests:

```http
GET /api/v1/portfolio
Authorization: Bearer jwt_token_here
```

---

## ⚠️ Error Codes

### HTTP Status Codes

| Code | Description | Example |
|------|-------------|---------|
| 200 | Success | Request successful |
| 400 | Bad Request | Invalid parameters |
| 401 | Unauthorized | Invalid API key |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 429 | Rate Limited | Too many requests |
| 500 | Server Error | Internal server error |

### Error Response Format

```json
{
    "error": {
        "code": "INVALID_SYMBOL",
        "message": "Symbol 'INVALID' not found",
        "details": {
            "symbol": "INVALID",
            "suggestions": ["AAPL", "GOOGL", "MSFT"]
        },
        "timestamp": "2025-06-30T10:15:00Z"
    }
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `INVALID_SYMBOL` | Symbol not recognized |
| `INSUFFICIENT_FUNDS` | Not enough cash for trade |
| `POSITION_LIMIT_EXCEEDED` | Too many open positions |
| `RISK_LIMIT_EXCEEDED` | Trade violates risk limits |
| `MARKET_CLOSED` | Market is closed for trading |
| `INVALID_ORDER_TYPE` | Unsupported order type |
| `RATE_LIMITED` | API rate limit exceeded |
| `INTERNAL_ERROR` | Unexpected server error |

---

## 📊 Rate Limits

### API Rate Limits

| Endpoint Category | Limit | Window |
|------------------|-------|---------|
| Market Data | 1000 req/min | 1 minute |
| Trading | 100 req/min | 1 minute |
| Portfolio | 500 req/min | 1 minute |
| Account | 200 req/min | 1 minute |
| WebSocket | 50 connections | Per user |

### Rate Limit Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1625097600
X-RateLimit-Window: 60
```

### Rate Limit Exceeded Response

```json
{
    "error": {
        "code": "RATE_LIMITED",
        "message": "Rate limit exceeded",
        "details": {
            "limit": 1000,
            "window": 60,
            "reset_time": "2025-06-30T10:16:00Z"
        }
    }
}
```

---

## 📝 SDK Examples

### Python SDK

```python
import asyncio
from neurocluster_elite import NeuroClusterClient

async def main():
    # Initialize client
    client = NeuroClusterClient(api_key='your_api_key')
    
    # Get portfolio
    portfolio = await client.get_portfolio()
    print(f"Portfolio Value: ${portfolio.total_value:,.2f}")
    
    # Get market data
    data = await client.get_market_data(['AAPL', 'GOOGL'])
    for symbol, market_data in data.items():
        print(f"{symbol}: ${market_data.price:.2f}")
    
    # Place order
    order = await client.place_order(
        symbol='AAPL',
        side='BUY',
        quantity=100,
        order_type='MARKET'
    )
    print(f"Order Status: {order.status}")

if __name__ == "__main__":
    asyncio.run(main())
```

### JavaScript SDK

```javascript
import { NeuroClusterClient } from 'neurocluster-elite-js';

const client = new NeuroClusterClient({
    apiKey: 'your_api_key',
    baseUrl: 'https://api.neurocluster-elite.com'
});

// Get portfolio
const portfolio = await client.getPortfolio();
console.log(`Portfolio Value: $${portfolio.totalValue.toLocaleString()}`);

// Real-time market data
client.subscribeToMarketData(['AAPL', 'GOOGL'], (data) => {
    console.log(`${data.symbol}: $${data.price}`);
});

// Place order
const order = await client.placeOrder({
    symbol: 'AAPL',
    side: 'BUY',
    quantity: 100,
    orderType: 'MARKET'
});
console.log(`Order Status: ${order.status}`);
```

---

## 📱 Mobile API

### REST Endpoints

Base URL: `https://api.neurocluster-elite.com/mobile/v1`

#### Get Portfolio Summary
```http
GET /mobile/v1/portfolio/summary
Authorization: Bearer jwt_token

Response:
{
    "totalValue": 105000.00,
    "todayPnl": 1250.00,
    "todayPnlPct": 1.20,
    "topGainer": {
        "symbol": "AAPL",
        "pnl": 525.00,
        "pnlPct": 3.61
    },
    "topLoser": {
        "symbol": "GOOGL",
        "pnl": -125.00,
        "pnlPct": -0.85
    }
}
```

#### Get Trading Signals
```http
GET /mobile/v1/signals/active
Authorization: Bearer jwt_token

Response:
{
    "signals": [
        {
            "symbol": "AAPL",
            "signalType": "BUY",
            "confidence": 0.85,
            "entryPrice": 150.25,
            "targetPrice": 165.00,
            "stopLoss": 142.75,
            "timestamp": "2025-06-30T10:15:00Z"
        }
    ]
}
```

---

## 🔧 Configuration API

### Update Trading Configuration

```http
PUT /api/v1/config/trading
Authorization: Bearer your_api_key
Content-Type: application/json

{
    "maxPositions": 15,
    "riskPerTrade": 0.02,
    "stopLossPct": 0.05,
    "takeProfitPct": 0.10,
    "autoTrading": false
}
```

### Get Algorithm Parameters

```http
GET /api/v1/config/algorithm

Response:
{
    "similarityThreshold": 0.75,
    "learningRate": 0.14,
    "decayRate": 0.02,
    "maxClusters": 12,
    "vectorizationEnabled": true,
    "driftDetection": true
}
```

---

## 📚 Additional Resources

- **[Strategy Development Guide](STRATEGY_GUIDE.md)** - Build custom trading strategies
- **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - Deploy in production
- **[Voice Commands Reference](VOICE_COMMANDS.md)** - Voice control interface
- **[GitHub Repository](https://github.com/neurocluster-elite/neurocluster-elite)** - Source code
- **[Community Discord](https://discord.gg/neurocluster-elite)** - Get support

---

**Last Updated:** June 30, 2025  
**API Version:** v1.0.0  
**Contact:** [api-support@neurocluster-elite.com](mailto:api-support@neurocluster-elite.com)