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
python -m src.main config/config.yaml --paper

# Run bot (dry-run validation only)
python -m src.main config/config.yaml --dry-run
```

## Testing

```bash
# Run all tests
pytest tests/

# Run single test file
pytest tests/test_grid_calculator.py

# Run single test function
pytest tests/test_grid_calculator.py::test_function_name -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run async tests (asyncio_mode is auto-configured in pyproject.toml)
pytest tests/test_orchestrator.py -v
```

## Code Quality

```bash
# Format code
black src/ tests/ scripts/
isort src/ tests/ scripts/

# Lint
ruff check src/ tests/ scripts/

# Type check
mypy src/
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
| `src/grid/` | Adaptive grid calculation, order management, rebalancing |
| `src/core/` | Orchestrator, risk manager, state manager, health monitoring, alerts |
| `src/persistence/` | SQLite database for state and order tracking |
| `src/operations/` | Grid execution and operational logic |
| `src/utils/` | Configuration loading, shared utilities |
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

## Core Runtime Components

| Component | Purpose |
|-----------|---------|
| `Orchestrator` | Main loop coordinator, handles startup/shutdown, signal handling |
| `RiskManager` | Enforces risk rules, triggers halt/pause actions |
| `StateManager` | Persists bot state to SQLite, enables crash recovery |
| `Portfolio` | Tracks equity, positions, high-water mark drawdown |
| `HealthChecker` | Monitors system health (API, WebSocket, memory) |
| `ProcessLock` | PID file locking to prevent multiple instances |
| `OrderReconciler` | Syncs local state with Kraken's order state |

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

## CLI Arguments

```bash
python -m src.main config/config.yaml [OPTIONS]

Options:
  --paper           Force paper trading mode (orders validated but not executed)
  --fresh           Clear saved state and start fresh
  --dry-run         Validate configuration and exit without trading
  --log-level LEVEL Set logging level (DEBUG, INFO, WARNING, ERROR)
  --log-file PATH   Log file path
  --force-start     Override PID lock (use if previous instance crashed)
  --no-resume       Skip state restoration (but keep history)
  --max-state-age N Maximum state age in seconds for resume (default: 86400)
```

## Paper Trading

Set `PAPER_TRADING=true` in `.env` or use `--paper` flag. Orders are validated but not executed.

## GCP VM Deployment

The bot runs on a Google Cloud VM as a systemd service.

### VM Specs

| Property | Value |
|----------|-------|
| **VM Name** | krakengridbot |
| **External IP** | 35.227.103.77 |
| **OS** | Ubuntu 24.04.3 LTS |
| **CPU** | 2 cores (Intel Xeon @ 2.20GHz) |
| **RAM** | 3.8 GB |
| **Disk** | 43 GB |
| **Python** | 3.12.3 |

### SSH Access

```bash
# SSH into VM (using configured alias)
ssh kraken-bot

# Direct SSH (if alias not configured)
ssh -i ~/.ssh/kraken_gcp beamo_beamosupport_com@35.227.103.77

# Copy file to VM
scp local_file.txt kraken-bot:~/kraken-grid-trading/

# Copy file from VM
scp kraken-bot:~/kraken-grid-trading/remote_file.txt ./
```

SSH config (~/.ssh/config):
```
Host kraken-bot gcp-kraken
    HostName 35.227.103.77
    User beamo_beamosupport_com
    IdentityFile ~/.ssh/kraken_gcp
    IdentitiesOnly yes
    ServerAliveInterval 60
```

### Service Management

```bash
# Check bot status
ssh kraken-bot "sudo systemctl status kraken-bot"

# Start/stop/restart
ssh kraken-bot "sudo systemctl start kraken-bot"
ssh kraken-bot "sudo systemctl stop kraken-bot"
ssh kraken-bot "sudo systemctl restart kraken-bot"

# View logs (live)
ssh kraken-bot "journalctl -u kraken-bot -f"

# View last 100 log lines
ssh kraken-bot "journalctl -u kraken-bot -n 100"

# Quick deploy (pull and restart)
ssh kraken-bot "cd ~/kraken-grid-trading && git pull origin main && sudo systemctl restart kraken-bot"

# Full deploy (with dependencies)
ssh kraken-bot "cd ~/kraken-grid-trading && git pull && source venv/bin/activate && pip install -r requirements.txt && sudo systemctl restart kraken-bot"
```

### VM File Paths

| Path | Purpose |
|------|---------|
| `~/kraken-grid-trading` | Bot installation directory |
| `~/kraken-grid-trading/venv` | Python virtual environment |
| `~/kraken-grid-trading/.env` | API credentials |
| `~/kraken-grid-trading/config/config.yaml` | Bot configuration |
| `~/kraken-grid-trading/data/` | Database and state files |
| `~/kraken-grid-trading/logs/` | Log files |

See `GCP_VM_REFERENCE.md` for complete VM documentation.
