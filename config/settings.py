"""
Configuration dataclasses for Kraken Grid Trading bot.

All configuration parameters are defined here with sensible defaults.
Values can be overridden via config.yaml or environment variables.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum


class RateLimitTier(Enum):
    """Kraken API rate limiting tiers."""
    STARTER = "starter"
    INTERMEDIATE = "intermediate"
    PRO = "pro"


class GridSpacing(Enum):
    """Grid spacing strategies."""
    EQUAL = "equal"
    GEOMETRIC = "geometric"


class ModelType(Enum):
    """ML model types for regime classification."""
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    ENSEMBLE = "ensemble"


# ===========================================
# KRAKEN API CONFIGURATION
# ===========================================

@dataclass
class KrakenConfig:
    """Kraken API configuration."""

    # API endpoints
    rest_base_url: str = "https://api.kraken.com"
    ws_url: str = "wss://ws.kraken.com/v2"

    # Public endpoints
    trades_endpoint: str = "/0/public/Trades"
    ohlc_endpoint: str = "/0/public/OHLC"
    ticker_endpoint: str = "/0/public/Ticker"
    asset_pairs_endpoint: str = "/0/public/AssetPairs"

    # Rate limiting
    tier: RateLimitTier = RateLimitTier.INTERMEDIATE
    rate_limit_buffer: float = 0.8  # Use 80% of capacity
    request_timeout: int = 30  # seconds

    # Max trades per API call (Kraken limit)
    max_trades_per_call: int = 1000

    @property
    def max_counter(self) -> int:
        """Get max API counter for tier."""
        return 15 if self.tier == RateLimitTier.STARTER else 20

    @property
    def decay_rate(self) -> float:
        """Get counter decay rate for tier (per second)."""
        rates = {
            RateLimitTier.STARTER: 0.33,
            RateLimitTier.INTERMEDIATE: 0.5,
            RateLimitTier.PRO: 1.0,
        }
        return rates[self.tier]


@dataclass
class WebSocketConfig:
    """WebSocket connection configuration."""

    url: str = "wss://ws.kraken.com/v2"  # Legacy - prefer public_url/private_url
    public_url: str = "wss://ws.kraken.com/v2"
    private_url: str = "wss://ws-auth.kraken.com/v2"
    ping_interval: float = 30.0  # seconds
    ping_timeout: float = 10.0
    reconnect_delay: float = 5.0  # Initial delay
    max_reconnect_delay: float = 300.0  # Max backoff (5 min)
    reconnect_multiplier: float = 2.0
    ohlc_interval: int = 5  # Subscribe to 5-minute candles
    message_queue_size: int = 1000  # Max messages in async queue


# ===========================================
# TRADING CONFIGURATION
# ===========================================

@dataclass
class TradingConfig:
    """Trading pair configuration."""

    pair: str = "XBTUSD"  # Kraken pair name
    base_currency: str = "XBT"  # BTC is XBT on Kraken
    quote_currency: str = "USD"

    # Kraken pair mapping (for WebSocket which uses different names)
    @property
    def ws_symbol(self) -> str:
        """Get WebSocket symbol in ISO format for WS API v2 (e.g., BTC/USD)."""
        # Convert XBTUSD -> BTC/USD (WS API v2 uses ISO 4217 format with slash)
        base = self.base_currency.replace("XBT", "BTC")
        return f"{base}/{self.quote_currency}"


@dataclass
class GridConfig:
    """Grid trading parameters."""

    num_levels: int = 10  # Number of grid levels
    range_percent: float = 5.0  # Grid spans +/- 5% from center
    order_size_quote: float = 40.0  # USD per grid level
    spacing: GridSpacing = GridSpacing.EQUAL
    rebalance_threshold: float = 0.1  # 10% deviation triggers rebalance

    @property
    def total_capital_required(self) -> float:
        """Calculate total capital needed for grid."""
        return self.num_levels * self.order_size_quote


@dataclass
class RiskConfig:
    """Risk management parameters - CRITICAL for capital preservation."""

    max_position_percent: float = 70.0  # Max 70% in positions
    max_open_orders: int = 20
    stop_loss_percent: float = 15.0  # Below lowest grid
    max_drawdown_percent: float = 20.0  # Halt if exceeded
    min_confidence: float = 0.6  # Pause if model confidence < 60%
    order_risk_percent: float = 2.0  # Max 2% risk per order


# ===========================================
# DATA STORAGE CONFIGURATION
# ===========================================

@dataclass
class StorageConfig:
    """Data storage paths and settings."""

    base_path: Path = field(default_factory=lambda: Path("data"))
    raw_trades_dir: str = "raw"
    ohlcv_dir: str = "ohlcv"
    features_dir: str = "features"
    models_dir: str = "models"

    # Parquet settings
    compression: str = "snappy"
    row_group_size: int = 100_000

    @property
    def raw_path(self) -> Path:
        return self.base_path / self.raw_trades_dir

    @property
    def ohlcv_path(self) -> Path:
        return self.base_path / self.ohlcv_dir

    @property
    def features_path(self) -> Path:
        return self.base_path / self.features_dir

    @property
    def models_path(self) -> Path:
        return self.base_path / self.models_dir


@dataclass
class DatabaseConfig:
    """SQLite database configuration."""

    path: str = "data/trading.db"
    backup_interval: int = 3600  # Hourly backups

    @property
    def url(self) -> str:
        """Get SQLAlchemy database URL."""
        return f"sqlite+aiosqlite:///{self.path}"


@dataclass
class RecoveryConfig:
    """Crash recovery and graceful shutdown configuration."""

    # State age limits
    max_state_age_seconds: int = 86400  # 24 hours (default)
    warn_state_age_seconds: int = 3600  # 1 hour (warning only)

    # Reconciliation
    always_reconcile: bool = True  # Always reconcile with exchange on startup
    reconcile_timeout_seconds: float = 60.0

    # PID lock
    pid_lock_path: str = "data/trading.pid"
    stale_lock_timeout_seconds: int = 300  # 5 minutes

    # Write-ahead log
    enable_wal: bool = True
    wal_retention_hours: int = 24


# ===========================================
# FEATURE ENGINEERING CONFIGURATION
# ===========================================

@dataclass
class FeatureConfig:
    """Feature engineering parameters."""

    # Lookback windows
    short_window: int = 14
    medium_window: int = 50
    long_window: int = 200

    # ATR and volatility
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0

    # ADX trend indicator
    adx_period: int = 14

    # Volume
    vwap_period: int = 20
    volume_ma_period: int = 20

    # Microstructure
    large_trade_percentile: float = 95.0
    imbalance_window: int = 100

    # OHLCV timeframes to compute
    timeframes: List[str] = field(
        default_factory=lambda: ["1m", "5m", "15m", "1h", "4h"]
    )

    # Timeframe in minutes mapping
    @staticmethod
    def timeframe_minutes(tf: str) -> int:
        """Convert timeframe string to minutes."""
        mapping = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440,
        }
        return mapping.get(tf, 1)


# ===========================================
# REGIME DETECTION CONFIGURATION
# ===========================================

@dataclass
class RegimeConfig:
    """Market regime detection thresholds."""

    # ADX thresholds for trend detection
    adx_trending_threshold: float = 25.0
    adx_strong_trend_threshold: float = 40.0

    # Volatility percentiles
    high_vol_percentile: float = 80.0
    low_vol_percentile: float = 20.0

    # Moving average periods for trend direction
    ma_short: int = 20
    ma_long: int = 50

    # Breakout detection
    breakout_vol_multiplier: float = 2.0
    breakout_atr_multiplier: float = 1.5

    # Forward look for supervised learning labels
    forward_look: int = 5


# ===========================================
# ML MODEL CONFIGURATION
# ===========================================

@dataclass
class XGBoostConfig:
    """XGBoost hyperparameters."""

    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42


@dataclass
class ModelConfig:
    """ML model configuration."""

    model_type: ModelType = ModelType.XGBOOST

    # Input configuration
    lookback_candles: int = 100  # Rolling window size

    # Train/val/test splits (chronological)
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    # Performance thresholds
    min_accuracy: float = 0.65  # Minimum acceptable accuracy
    min_class_recall: float = 0.50  # Per-class recall threshold

    # Model-specific configs
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)


# ===========================================
# MONITORING CONFIGURATION
# ===========================================

@dataclass
class MonitoringConfig:
    """Health checks and metrics configuration."""

    health_check_interval: float = 60.0  # seconds
    metrics_interval: float = 300.0  # 5 minutes
    daily_summary_hour: int = 0  # UTC hour


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    file_path: str = "logs/trading.log"
    max_size_mb: int = 10
    backup_count: int = 5
    json_format: bool = True


# ===========================================
# MAIN BOT CONFIGURATION
# ===========================================

@dataclass
class BotConfig:
    """Complete bot configuration combining all sub-configs."""

    # Core configs
    kraken: KrakenConfig = field(default_factory=KrakenConfig)
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    # Data configs
    storage: StorageConfig = field(default_factory=StorageConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)

    # ML configs
    features: FeatureConfig = field(default_factory=FeatureConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # Operational configs
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Paper trading mode
    paper_trading: bool = True  # Default to paper trading for safety

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check capital vs grid requirements
        required = self.grid.total_capital_required
        if required > 400:  # Assuming $400 starting capital
            errors.append(
                f"Grid requires ${required}, exceeds $400 capital"
            )

        # Check risk parameters are sensible
        if self.risk.max_drawdown_percent > 25:
            errors.append(
                "Max drawdown > 25% is very risky with $400 capital"
            )

        if self.risk.stop_loss_percent > 20:
            errors.append(
                "Stop loss > 20% risks too much capital per position"
            )

        # Check ML config
        split_sum = (
            self.model.train_split +
            self.model.val_split +
            self.model.test_split
        )
        if abs(split_sum - 1.0) > 0.01:
            errors.append(
                f"Train/val/test splits must sum to 1.0, got {split_sum}"
            )

        return errors
