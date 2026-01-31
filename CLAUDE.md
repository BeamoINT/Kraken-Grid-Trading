# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Adaptive AI Grid Trading Bot for Kraken - An autonomous cryptocurrency trading bot that uses machine learning to detect market regimes and adapt grid trading behavior accordingly. Designed for $400 starting capital on a Kraken business account.

## Quick Commands

```bash
# Setup (requires Python >= 3.11)
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
python scripts/train_model.py --pair XBTUSD --timeframe 5m

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
```

Note: `asyncio_mode = "auto"` is configured in pyproject.toml, so async tests work without extra markers.

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

## Market Regimes

The ML model classifies markets into 5 regimes, each with different grid behavior:

| Regime | Detection | Grid Action |
|--------|-----------|-------------|
| RANGING | ADX < 25 | Normal grid, tighter spacing |
| TRENDING_UP | ADX > 25, +DI > -DI | Shift up, fewer buys |
| TRENDING_DOWN | ADX > 25, -DI > +DI | Pause buys, stop-loss |
| HIGH_VOLATILITY | ATR > 80th percentile | Widen spacing |
| BREAKOUT | BB breach + volume spike | Pause entirely |

## Kraken API Notes

- **Historical Data**: Use `/0/public/Trades` endpoint (unlimited) instead of `/0/public/OHLC` (720 candle limit)
- **Rate Limiting**: Intermediate tier = 20 max counter, 0.5/sec decay. Use token bucket.
- **Authentication**: HMAC-SHA512 with nonce (millisecond timestamp)
- **Pair Names**: Kraken uses XBTUSD (not BTCUSD) for Bitcoin

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

The bot runs on a Google Cloud VM (`krakengridbot` at `35.227.103.77`) as a systemd service.

```bash
# SSH into VM
ssh kraken-bot

# Check status / view logs
ssh kraken-bot "sudo systemctl status kraken-bot"
ssh kraken-bot "journalctl -u kraken-bot -f"

# Quick deploy
ssh kraken-bot "cd ~/kraken-grid-trading && git pull origin main && sudo systemctl restart kraken-bot"

# Full deploy (with dependencies)
ssh kraken-bot "cd ~/kraken-grid-trading && git pull && source venv/bin/activate && pip install -r requirements.txt && sudo systemctl restart kraken-bot"
```

See `GCP_VM_REFERENCE.md` for complete VM documentation including SSH config, troubleshooting, and service management.
