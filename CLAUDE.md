# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Adaptive AI Grid Trading Bot for Kraken - An autonomous cryptocurrency trading bot that uses machine learning to detect market regimes and adapt grid trading behavior accordingly. Designed for $400 starting capital on a Kraken business account.

## Quick Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env      # Then edit with your API keys
cp config/config.yaml.example config/config.yaml

# Download historical data
python scripts/download_historical.py --pairs XBTUSD

# Compute features
python scripts/compute_features.py --pairs XBTUSD --timeframes 1m 5m 15m 1h 4h

# Train model
python scripts/train_model.py

# Run bot (paper trading)
python -m src.main config/config.yaml

# Run tests
pytest tests/
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR                             │
│  Coordinates all components, handles startup/shutdown           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Data Ingester│  │ ML Inference │  │   Grid Manager       │  │
│  │ - WebSocket  │  │ - Features   │  │ - Calculate levels   │  │
│  │ - OHLCV      │  │ - Regime     │  │ - Place orders       │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│         │                  │                    │               │
│         └──────────────────┼────────────────────┘               │
│                            ▼                                    │
│                   ┌──────────────┐                              │
│                   │ Risk Manager │                              │
│                   │ - Stop-loss  │                              │
│                   │ - Drawdown   │                              │
│                   └──────────────┘                              │
│                            │                                    │
│                            ▼                                    │
│                   ┌──────────────┐                              │
│                   │ Kraken API   │                              │
│                   └──────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `src/api/` | Kraken REST + WebSocket clients with rate limiting |
| `src/data/` | Historical data download, OHLCV aggregation, Parquet storage |
| `src/features/` | Technical indicator computation (ATR, ADX, BBands, RSI, MACD, etc.) |
| `src/regime/` | Market regime labeling and classification |
| `src/models/` | XGBoost regime classifier with walk-forward validation |
| `src/grid/` | Adaptive grid calculation and order management |
| `src/core/` | Orchestrator and risk management |
| `src/persistence/` | SQLite database for state and order tracking |
| `config/` | Configuration dataclasses and YAML loading |

## Feature Engineering

The feature pipeline computes 100+ technical features across multiple categories:

| Category | Features | Purpose |
|----------|----------|---------|
| **Price** | Returns, momentum, MA distances, pivot points | Price dynamics and trend following |
| **Volume** | OBV, buy/sell ratio, CMF, Klinger | Volume-based confirmation |
| **Volatility** | ATR, Bollinger, Parkinson, Yang-Zhang | Risk measurement |
| **Trend** | ADX/DI, MACD, RSI, Ichimoku | Trend strength and direction |

### Feature Computation Commands

```bash
# Compute all features for XBTUSD
python scripts/compute_features.py --pairs XBTUSD

# Compute for specific timeframes with regime labeling
python scripts/compute_features.py --pairs XBTUSD --timeframes 5m 1h 4h --label-regimes

# Show feature summary
python scripts/compute_features.py --pairs XBTUSD --summary
```

## Market Regimes

The ML model classifies markets into 5 regimes, each with different grid behavior:

| Regime | Detection | Grid Action |
|--------|-----------|-------------|
| RANGING | ADX < 25 | Normal grid, tighter spacing |
| TRENDING_UP | ADX > 25, +DI > -DI | Shift up, fewer buys |
| TRENDING_DOWN | ADX > 25, -DI > +DI | Pause buys, stop-loss |
| HIGH_VOLATILITY | ATR > 80th percentile | Widen spacing |
| BREAKOUT | BB breach + volume spike | Pause entirely |

## Model Training

The XGBoost regime classifier uses walk-forward validation for time-series:

```bash
# Train with default settings
python scripts/train_model.py --pair XBTUSD --timeframe 5m

# Train with hyperparameter tuning
python scripts/train_model.py --pair XBTUSD --timeframe 5m --tune

# Train with walk-forward validation
python scripts/train_model.py --pair XBTUSD --timeframe 5m --walk-forward

# List trained models
python scripts/train_model.py --list-models

# Set production model
python scripts/train_model.py --set-production MODEL_ID
```

### Model Components

| Component | Purpose |
|-----------|---------|
| `DataPreparation` | Time-series aware train/val/test splitting, feature scaling |
| `RegimeClassifier` | XGBoost multi-class classifier with early stopping |
| `WalkForwardValidator` | Expanding/rolling window validation for financial data |
| `ModelEvaluator` | Confusion matrix, per-class metrics, trading-specific metrics |
| `ModelRegistry` | Model versioning, production model management |

## Kraken API Notes

- **Historical Data**: Use `/0/public/Trades` endpoint (unlimited) instead of `/0/public/OHLC` (720 candle limit)
- **Rate Limiting**: Intermediate tier = 20 max counter, 0.5/sec decay. Use token bucket.
- **Authentication**: HMAC-SHA512 with nonce (millisecond timestamp)
- **Pair Names**: Kraken uses XBTUSD (not BTCUSD) for Bitcoin

## API Components (src/api/)

| Component | Purpose |
|-----------|---------|
| `auth.py` | HMAC-SHA512 authentication, nonce generation, credential loading |
| `kraken_private.py` | Private REST API client (balance, orders, trades) |
| `order_manager.py` | Grid order lifecycle, position tracking, P&L calculation |
| `websocket_client.py` | Real-time data streaming (ticker, OHLC, trades, order updates) |
| `kraken_errors.py` | Error classification, retry logic, circuit breaker |
| `rate_limiter.py` | Token bucket rate limiting for API calls |

### API Usage Examples

```python
# Private API (requires credentials)
from src.api import KrakenPrivateClient, OrderSide, OrderType

client = KrakenPrivateClient.from_env()  # Uses KRAKEN_API_KEY, KRAKEN_API_SECRET
balance = client.get_balance()
result = client.add_order(
    pair="XBTUSD",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    volume=0.001,
    price=50000,
)

# Order Manager (grid trading)
from src.api import OrderManager, GridOrderType
from decimal import Decimal

manager = OrderManager(client)
order = manager.create_grid_order(
    level=5,
    price=Decimal("50000"),
    side=GridOrderType.BUY,
    volume=Decimal("0.001"),
)
manager.submit_order(order)

# WebSocket (real-time data)
from src.api import KrakenWebSocketClient

async with KrakenWebSocketClient() as ws:
    await ws.subscribe_ticker(["BTC/USD"])
    async for msg in ws:
        print(msg)
```

### Error Handling

The API module provides comprehensive error handling:

| Error Type | Cause | Retry Strategy |
|------------|-------|----------------|
| `RateLimitError` | API rate limit exceeded | Wait for retry_after |
| `AuthenticationError` | Invalid API key/secret | No retry |
| `InsufficientFundsError` | Not enough balance | No retry |
| `OrderError` | Order-related issues | Depends on error |
| `NetworkError` | Connection issues | Exponential backoff |

```python
from src.api import with_retry, RetryConfig, CircuitBreaker

# Automatic retry
@with_retry(RetryConfig(max_retries=3))
def api_call():
    ...

# Circuit breaker for preventing cascading failures
breaker = CircuitBreaker(failure_threshold=5, reset_timeout=60)
```

## Risk Rules

- Max 2% capital risk per grid level
- Stop-loss: 15% below lowest grid
- Max drawdown: 20% halts trading
- Max exposure: 70% in positions
- Pause if model confidence < 60%

## Data Flow

```
Kraken Trades API → Raw Trades (Parquet) → OHLCV → Features → Regime Labels
                                                        ↓
                                              ML Model Training
                                                        ↓
Live WebSocket → Real-time Features → Regime Prediction → Grid Adjustment
```

## Configuration

- **API credentials**: Set in `.env` file (never commit)
- **Trading parameters**: Set in `config/config.yaml`
- **Environment overrides**: `KRAKEN_*` env vars override YAML values

## Paper Trading

Set `PAPER_TRADING=true` in `.env` or `validate: true` in AddOrder requests. Orders are validated but not executed.
