"""
Configuration loader for Kraken Grid Trading bot.

Loads configuration from:
1. YAML file (config/config.yaml)
2. Environment variables (KRAKEN_* prefix)
3. .env file (via python-dotenv)

Environment variables override YAML values.
API credentials MUST be set via environment (never in YAML).
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from config.settings import (
    BotConfig,
    KrakenConfig,
    WebSocketConfig,
    TradingConfig,
    GridConfig,
    RiskConfig,
    StorageConfig,
    DatabaseConfig,
    FeatureConfig,
    RegimeConfig,
    ModelConfig,
    MonitoringConfig,
    LoggingConfig,
    XGBoostConfig,
    RateLimitTier,
    GridSpacing,
    ModelType,
)

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Loads and validates bot configuration.

    Configuration priority (highest to lowest):
    1. Environment variables (KRAKEN_*)
    2. YAML config file
    3. Default values in dataclasses
    """

    ENV_PREFIX = "KRAKEN_"

    def __init__(
        self,
        config_path: Optional[str] = None,
        env_file: Optional[str] = None,
    ):
        """
        Initialize config loader.

        Args:
            config_path: Path to YAML config file. If None, uses config/config.yaml
            env_file: Path to .env file. If None, uses .env in project root
        """
        self._config_path = Path(config_path) if config_path else Path("config/config.yaml")
        self._env_file = Path(env_file) if env_file else Path(".env")

        # Load .env file if exists
        if self._env_file.exists():
            load_dotenv(self._env_file)
            logger.debug(f"Loaded environment from {self._env_file}")

    def load(self) -> BotConfig:
        """
        Load complete bot configuration.

        Returns:
            BotConfig with all settings populated

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        # Load YAML config
        yaml_config = self._load_yaml()

        # Build config objects
        config = self._build_config(yaml_config)

        # Validate
        errors = config.validate()
        if errors:
            for error in errors:
                logger.warning(f"Config warning: {error}")

        return config

    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not self._config_path.exists():
            logger.warning(f"Config file not found: {self._config_path}, using defaults")
            return {}

        with open(self._config_path) as f:
            config = yaml.safe_load(f) or {}

        logger.info(f"Loaded config from {self._config_path}")
        return config

    def _get_env(self, key: str, default: Any = None) -> Any:
        """
        Get environment variable with KRAKEN_ prefix.

        Args:
            key: Variable name (without prefix)
            default: Default value if not set

        Returns:
            Environment variable value or default
        """
        full_key = f"{self.ENV_PREFIX}{key}"
        value = os.environ.get(full_key)

        if value is None:
            return default

        # Type conversion based on default type
        if isinstance(default, bool):
            return value.lower() in ("true", "1", "yes")
        elif isinstance(default, int):
            return int(value)
        elif isinstance(default, float):
            return float(value)

        return value

    def _get_required_env(self, key: str) -> str:
        """
        Get required environment variable.

        Args:
            key: Variable name (without prefix)

        Returns:
            Environment variable value

        Raises:
            ValueError: If variable not set
        """
        full_key = f"{self.ENV_PREFIX}{key}"
        value = os.environ.get(full_key)

        if not value:
            raise ValueError(
                f"Required environment variable {full_key} not set. "
                f"Set it in .env file or environment."
            )

        return value

    def _build_config(self, yaml_config: Dict[str, Any]) -> BotConfig:
        """Build BotConfig from YAML and environment."""

        # Helper to get nested YAML value
        def get_yaml(section: str, key: str, default: Any = None) -> Any:
            return yaml_config.get(section, {}).get(key, default)

        # Build Kraken API config
        kraken_yaml = yaml_config.get("api", {})
        tier_str = self._get_env("API_TIER", kraken_yaml.get("tier", "intermediate"))
        kraken = KrakenConfig(
            tier=RateLimitTier(tier_str),
            rate_limit_buffer=self._get_env(
                "RATE_LIMIT_BUFFER",
                kraken_yaml.get("rate_limit_buffer", 0.8)
            ),
            request_timeout=self._get_env(
                "REQUEST_TIMEOUT",
                kraken_yaml.get("request_timeout", 30)
            ),
        )

        # Build WebSocket config
        ws_yaml = yaml_config.get("websocket", {})
        websocket = WebSocketConfig(
            reconnect_delay=ws_yaml.get("reconnect_delay", 5.0),
            max_reconnect_delay=ws_yaml.get("max_reconnect_delay", 300.0),
            ping_interval=ws_yaml.get("ping_interval", 30.0),
            ohlc_interval=ws_yaml.get("ohlc_interval", 5),
        )

        # Build Trading config
        trading_yaml = yaml_config.get("trading", {})
        trading = TradingConfig(
            pair=self._get_env("TRADING_PAIR", trading_yaml.get("pair", "XBTUSD")),
            base_currency=trading_yaml.get("base_currency", "XBT"),
            quote_currency=trading_yaml.get("quote_currency", "USD"),
        )

        # Build Grid config
        grid_yaml = yaml_config.get("grid", {})
        spacing_str = self._get_env(
            "GRID_SPACING",
            grid_yaml.get("spacing", "equal")
        )
        grid = GridConfig(
            num_levels=self._get_env(
                "GRID_NUM_LEVELS",
                grid_yaml.get("num_levels", 10)
            ),
            range_percent=self._get_env(
                "GRID_RANGE_PERCENT",
                grid_yaml.get("range_percent", 5.0)
            ),
            order_size_quote=self._get_env(
                "GRID_ORDER_SIZE_QUOTE",
                grid_yaml.get("order_size_quote", 40.0)
            ),
            spacing=GridSpacing(spacing_str),
            rebalance_threshold=grid_yaml.get("rebalance_threshold", 0.1),
        )

        # Build Risk config
        risk_yaml = yaml_config.get("risk", {})
        risk = RiskConfig(
            max_position_percent=risk_yaml.get("max_position_percent", 70.0),
            max_open_orders=risk_yaml.get("max_open_orders", 20),
            stop_loss_percent=risk_yaml.get("stop_loss_percent", 15.0),
            max_drawdown_percent=risk_yaml.get("max_drawdown_percent", 20.0),
            min_confidence=risk_yaml.get("min_confidence", 0.6),
            order_risk_percent=risk_yaml.get("order_risk_percent", 2.0),
        )

        # Build Storage config
        storage_yaml = yaml_config.get("storage", {})
        storage = StorageConfig(
            base_path=Path(storage_yaml.get("base_path", "data")),
            raw_trades_dir=storage_yaml.get("raw_trades_dir", "raw"),
            ohlcv_dir=storage_yaml.get("ohlcv_dir", "ohlcv"),
            features_dir=storage_yaml.get("features_dir", "features"),
            models_dir=storage_yaml.get("models_dir", "models"),
            compression=storage_yaml.get("compression", "snappy"),
        )

        # Build Database config
        db_yaml = yaml_config.get("database", {})
        database = DatabaseConfig(
            path=db_yaml.get("path", "data/trading.db"),
            backup_interval=db_yaml.get("backup_interval", 3600),
        )

        # Build Feature config
        features_yaml = yaml_config.get("features", {})
        features = FeatureConfig(
            short_window=features_yaml.get("short_window", 14),
            medium_window=features_yaml.get("medium_window", 50),
            long_window=features_yaml.get("long_window", 200),
            atr_period=features_yaml.get("atr_period", 14),
            bb_period=features_yaml.get("bb_period", 20),
            bb_std=features_yaml.get("bb_std", 2.0),
            adx_period=features_yaml.get("adx_period", 14),
            vwap_period=features_yaml.get("vwap_period", 20),
            volume_ma_period=features_yaml.get("volume_ma_period", 20),
        )

        # Build Regime config
        regime_yaml = yaml_config.get("regime", {})
        regime = RegimeConfig(
            adx_trending_threshold=regime_yaml.get("adx_trending_threshold", 25.0),
            adx_strong_trend_threshold=regime_yaml.get("adx_strong_trend_threshold", 40.0),
            high_vol_percentile=regime_yaml.get("high_vol_percentile", 80.0),
            low_vol_percentile=regime_yaml.get("low_vol_percentile", 20.0),
            ma_short=regime_yaml.get("ma_short", 20),
            ma_long=regime_yaml.get("ma_long", 50),
            breakout_vol_multiplier=regime_yaml.get("breakout_vol_multiplier", 2.0),
            breakout_atr_multiplier=regime_yaml.get("breakout_atr_multiplier", 1.5),
            forward_look=regime_yaml.get("forward_look", 5),
        )

        # Build Model config
        model_yaml = yaml_config.get("model", {})
        xgb_yaml = model_yaml.get("xgboost", {})

        model_type_str = model_yaml.get("type", "xgboost")
        model = ModelConfig(
            model_type=ModelType(model_type_str),
            lookback_candles=model_yaml.get("lookback_candles", 100),
            train_split=model_yaml.get("train_split", 0.7),
            val_split=model_yaml.get("val_split", 0.15),
            test_split=model_yaml.get("test_split", 0.15),
            xgboost=XGBoostConfig(
                n_estimators=xgb_yaml.get("n_estimators", 100),
                max_depth=xgb_yaml.get("max_depth", 6),
                learning_rate=xgb_yaml.get("learning_rate", 0.1),
                min_child_weight=xgb_yaml.get("min_child_weight", 1),
                subsample=xgb_yaml.get("subsample", 0.8),
                colsample_bytree=xgb_yaml.get("colsample_bytree", 0.8),
            ),
        )

        # Build Monitoring config
        monitoring_yaml = yaml_config.get("monitoring", {})
        monitoring = MonitoringConfig(
            health_check_interval=monitoring_yaml.get("health_check_interval", 60.0),
            metrics_interval=monitoring_yaml.get("metrics_interval", 300.0),
            daily_summary_hour=monitoring_yaml.get("daily_summary_hour", 0),
        )

        # Build Logging config
        logging_yaml = yaml_config.get("logging", {})
        log_config = LoggingConfig(
            level=self._get_env("LOG_LEVEL", logging_yaml.get("level", "INFO")),
            file_path=logging_yaml.get("file_path", "logs/trading.log"),
            max_size_mb=logging_yaml.get("max_size_mb", 10),
            backup_count=logging_yaml.get("backup_count", 5),
            json_format=logging_yaml.get("json_format", True),
        )

        # Paper trading from environment
        paper_trading = self._get_env("PAPER_TRADING", True)
        if isinstance(paper_trading, str):
            paper_trading = paper_trading.lower() in ("true", "1", "yes")

        return BotConfig(
            kraken=kraken,
            websocket=websocket,
            trading=trading,
            grid=grid,
            risk=risk,
            storage=storage,
            database=database,
            features=features,
            regime=regime,
            model=model,
            monitoring=monitoring,
            logging=log_config,
            paper_trading=paper_trading,
        )

    def get_api_credentials(self) -> tuple[str, str]:
        """
        Get API credentials from environment.

        API credentials MUST be set via environment variables,
        never stored in config files.

        Returns:
            Tuple of (api_key, api_secret)

        Raises:
            ValueError: If credentials not set
        """
        api_key = self._get_required_env("API_KEY")
        api_secret = self._get_required_env("API_SECRET")
        return api_key, api_secret
