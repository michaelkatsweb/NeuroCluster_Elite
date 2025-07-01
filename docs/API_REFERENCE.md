# 📚 NeuroCluster Elite API Reference

## Overview
RESTful API providing programmatic access to all NeuroCluster Elite features.

## Base URL
- **Production**: `https://api.neurocluster-elite.com/v1`
- **Staging**: `https://staging-api.neurocluster-elite.com/v1`
- **Local**: `http://localhost:8000/v1`

## Authentication

### JWT Token Authentication
```bash
# Login to get token
curl -X POST /auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'

# Response
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### Using Token
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" /api/v1/portfolio/balance
```

## Endpoints

### Authentication
- `POST /auth/login` - User login
- `POST /auth/logout` - User logout
- `POST /auth/refresh` - Refresh token
- `GET /auth/profile` - Get user profile

### Market Data
- `GET /market/data` - Real-time market data
- `GET /market/history` - Historical data
- `GET /market/analysis` - Market analysis
- `GET /market/news` - Market news

### Trading
- `POST /trading/execute` - Execute trade
- `GET /trading/orders` - Get orders
- `DELETE /trading/orders/{id}` - Cancel order
- `GET /trading/history` - Trading history

### Portfolio
- `GET /portfolio/balance` - Account balance
- `GET /portfolio/positions` - Current positions
- `GET /portfolio/performance` - Performance metrics
- `GET /portfolio/risk` - Risk assessment

### Analytics
- `GET /analytics/performance` - Algorithm performance
- `GET /analytics/market-regime` - Market regime detection
- `GET /analytics/sentiment` - Sentiment analysis
- `GET /analytics/predictions` - Price predictions

## Request/Response Format

### Request Headers
```
Content-Type: application/json
Authorization: Bearer {token}
User-Agent: YourApp/1.0
```

### Response Format
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "price": 150.00,
    "change": 2.50
  },
  "timestamp": "2025-07-01T14:30:00Z",
  "request_id": "req_123456789"
}
```

### Error Format
```json
{
  "success": false,
  "error": {
    "code": "INVALID_SYMBOL",
    "message": "The provided symbol is not valid",
    "details": {}
  },
  "timestamp": "2025-07-01T14:30:00Z",
  "request_id": "req_123456789"
}
```

## Examples

### Get Market Data
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "https://api.neurocluster-elite.com/v1/market/data?symbol=AAPL"
```

Response:
```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "price": 150.25,
    "change": 2.75,
    "change_percent": 1.86,
    "volume": 45678900,
    "market_cap": 2451000000000,
    "timestamp": "2025-07-01T14:30:00Z"
  }
}
```

### Execute Trade
```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "quantity": 10,
    "side": "buy",
    "order_type": "market"
  }' \
  "https://api.neurocluster-elite.com/v1/trading/execute"
```

### Get Portfolio Balance
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "https://api.neurocluster-elite.com/v1/portfolio/balance"
```

Response:
```json
{
  "success": true,
  "data": {
    "total_value": 125000.00,
    "cash_balance": 25000.00,
    "invested_value": 100000.00,
    "day_change": 1250.00,
    "day_change_percent": 1.01
  }
}
```

## Rate Limits

- **Standard Users**: 100 requests/minute
- **Premium Users**: 1000 requests/minute
- **Enterprise**: Custom limits

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1625140800
```

## Webhooks

### Real-time Updates
```bash
# Subscribe to price updates
curl -X POST \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"symbol": "AAPL", "callback_url": "https://your-app.com/webhook"}' \
  "https://api.neurocluster-elite.com/v1/webhooks/subscribe"
```

### Webhook Payload
```json
{
  "event": "price_update",
  "symbol": "AAPL",
  "price": 150.25,
  "timestamp": "2025-07-01T14:30:00Z"
}
```

## WebSocket API

### Connection
```javascript
const ws = new WebSocket('wss://api.neurocluster-elite.com/v1/ws');

// Authenticate
ws.send(JSON.stringify({
  "action": "authenticate",
  "token": "YOUR_JWT_TOKEN"
}));

// Subscribe to updates
ws.send(JSON.stringify({
  "action": "subscribe",
  "symbol": "AAPL"
}));
```

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 429 | Too Many Requests |
| 500 | Internal Server Error |

## SDKs

### Python SDK
```python
from neurocluster_client import NeuroClusterClient

client = NeuroClusterClient(api_key="YOUR_API_KEY")
data = client.get_market_data("AAPL")
```

### JavaScript SDK
```javascript
import { NeuroClusterClient } from 'neurocluster-js';

const client = new NeuroClusterClient({ apiKey: 'YOUR_API_KEY' });
const data = await client.getMarketData('AAPL');
```

## Support

- **Documentation**: docs.neurocluster-elite.com
- **API Support**: api-support@neurocluster-elite.com
- **Status Page**: status.neurocluster-elite.com
