# Platform API Reference

## Polymarket

- **Type**: Central Limit Order Book (CLOB)
- **Chain**: Polygon (chain ID 137)
- **Settlement**: On-chain
- **Matching**: Off-chain
- **Auth**: EIP-712 signing

### Endpoints
- REST API: `https://clob.polymarket.com`
- WebSocket: `wss://ws-subscriptions-clob.polymarket.com/ws/market`
- Docs: https://docs.polymarket.com

### Key Endpoints
- `GET /markets` - List active markets
- `GET /book?token_id=X` - Orderbook for a token
- `POST /order` - Place an order
- `GET /trades` - Recent trades

---

## Kalshi

- **Type**: US-regulated exchange
- **Auth**: API key + secret header signing
- **Demo**: `https://demo-api.kalshi.co/trade-api/v2`
- **Production**: `https://trading-api.kalshi.co/trade-api/v2`
- **Docs**: https://trading-api.readme.io

### Key Endpoints
- `GET /markets` - List markets
- `POST /portfolio/orders` - Place an order
- `GET /portfolio/positions` - Current positions
- `POST /login` - Authenticate

### Notes
- Always use demo environment first with mock funds
- API requests require specific header signing
- Developer Agreement applies

---

## Unified Wrapper

For a unified API across both platforms, consider:
- **pmxt**: Inspired by CCXT but for prediction markets
- Provides consistent interface for market data and order execution
