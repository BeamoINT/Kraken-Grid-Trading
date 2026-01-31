"""
Periodic data synchronization service.

Keeps historical trade data and OHLCV candles up to date during live trading.
Runs as a background task that periodically fetches new data without
interrupting the main trading loop.

Features:
- Incremental trade downloads (only new trades since last sync)
- OHLCV candle updates for configured timeframes
- Configurable sync intervals
- Non-blocking async operation
- Graceful error handling
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

from .kraken_client import KrakenPublicClient
from .trade_ingestion import TradeIngestionPipeline
from .ohlcv_builder import OHLCVBuilder

logger = logging.getLogger(__name__)


@dataclass
class DataSyncConfig:
    """Configuration for data synchronization."""

    # Sync intervals
    trade_sync_interval: float = 3600.0  # 1 hour
    ohlcv_rebuild_interval: float = 14400.0  # 4 hours (only rebuild OHLCV this often)

    # Data paths
    raw_data_path: Path = Path("data/raw")
    ohlcv_path: Path = Path("data/ohlcv")

    # Pairs and timeframes
    pairs: List[str] = None  # Will be set from trading config
    timeframes: List[str] = None  # Default timeframes for OHLCV

    # Limits
    max_trades_per_sync: int = 100_000  # Don't download more than this per sync
    sync_lookback_hours: int = 24  # Only sync last N hours of data

    def __post_init__(self):
        if self.pairs is None:
            self.pairs = ["XBTUSD"]
        if self.timeframes is None:
            self.timeframes = ["1h", "4h"]  # Match model timeframes


class DataSyncService:
    """
    Background service for keeping historical data up to date.

    This service runs periodically during live trading to:
    1. Download new trades since last sync
    2. Update OHLCV candles with new data

    It's designed to be non-blocking and gracefully handle errors
    without affecting the main trading loop.
    """

    def __init__(
        self,
        config: DataSyncConfig,
        client: Optional[KrakenPublicClient] = None,
    ):
        """
        Initialize data sync service.

        Args:
            config: Sync configuration
            client: Kraken API client (creates one if not provided)
        """
        self.config = config
        self._client = client or KrakenPublicClient()

        # Create pipelines
        self._trade_pipeline = TradeIngestionPipeline(
            client=self._client,
            storage_path=config.raw_data_path,
            batch_size=10_000,  # Smaller batches for live updates
        )

        self._ohlcv_builder = OHLCVBuilder(
            trades_path=config.raw_data_path,
            ohlcv_path=config.ohlcv_path,
        )

        # State tracking
        self._last_trade_sync: Dict[str, datetime] = {}
        self._last_ohlcv_rebuild: Dict[str, datetime] = {}
        self._sync_stats: Dict[str, Any] = {
            "total_syncs": 0,
            "total_trades_downloaded": 0,
            "last_sync_time": None,
            "errors": [],
        }

        # Control
        self._running = False
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the sync service (called from orchestrator)."""
        self._running = True
        self._shutdown_event.clear()
        logger.info(
            f"Data sync service started - "
            f"trade sync every {self.config.trade_sync_interval}s, "
            f"OHLCV rebuild every {self.config.ohlcv_rebuild_interval}s"
        )

    async def stop(self) -> None:
        """Stop the sync service gracefully."""
        self._running = False
        self._shutdown_event.set()
        logger.info("Data sync service stopped")

    async def sync_once(self, pair: str) -> Dict[str, Any]:
        """
        Perform a single sync cycle for a trading pair.

        Downloads new trades and optionally rebuilds OHLCV.

        Args:
            pair: Trading pair (e.g., "XBTUSD")

        Returns:
            Dict with sync results
        """
        result = {
            "pair": pair,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trades_downloaded": 0,
            "ohlcv_updated": False,
            "errors": [],
        }

        # Download new trades
        try:
            trades_result = await self._sync_trades(pair)
            result["trades_downloaded"] = trades_result.get("new_trades", 0)
            self._sync_stats["total_trades_downloaded"] += result["trades_downloaded"]
        except Exception as e:
            error_msg = f"Trade sync error for {pair}: {e}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            self._sync_stats["errors"].append({
                "time": datetime.now(timezone.utc).isoformat(),
                "type": "trade_sync",
                "error": str(e),
            })

        # Check if OHLCV rebuild is needed
        now = datetime.now(timezone.utc)
        last_rebuild = self._last_ohlcv_rebuild.get(pair)

        if last_rebuild is None or (now - last_rebuild).total_seconds() >= self.config.ohlcv_rebuild_interval:
            try:
                await self._rebuild_ohlcv(pair)
                result["ohlcv_updated"] = True
                self._last_ohlcv_rebuild[pair] = now
            except Exception as e:
                error_msg = f"OHLCV rebuild error for {pair}: {e}"
                logger.error(error_msg)
                result["errors"].append(error_msg)

        self._sync_stats["total_syncs"] += 1
        self._sync_stats["last_sync_time"] = result["timestamp"]

        if result["trades_downloaded"] > 0 or result["ohlcv_updated"]:
            logger.info(
                f"Data sync for {pair}: "
                f"{result['trades_downloaded']} trades, "
                f"OHLCV {'updated' if result['ohlcv_updated'] else 'skipped'}"
            )

        return result

    async def _sync_trades(self, pair: str) -> Dict[str, Any]:
        """
        Download new trades since last sync.

        Uses incremental download starting from checkpoint.
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        def _download():
            # Load checkpoint to get last trade ID
            last_id = self._trade_pipeline._load_checkpoint(pair)

            if last_id is None:
                # No previous data - this shouldn't happen in live trading
                # but handle gracefully by starting from recent history
                logger.warning(f"No checkpoint for {pair}, skipping trade sync")
                return {"new_trades": 0, "message": "No checkpoint found"}

            # Download incrementally from last checkpoint
            stats = self._trade_pipeline.download_historical(
                pair=pair,
                start_from=last_id,
                max_trades=self.config.max_trades_per_sync,
            )

            return {
                "new_trades": stats.get("total_trades", 0),
                "last_timestamp": stats.get("last_timestamp"),
            }

        result = await loop.run_in_executor(None, _download)
        self._last_trade_sync[pair] = datetime.now(timezone.utc)

        return result

    async def _rebuild_ohlcv(self, pair: str) -> Dict[str, Any]:
        """
        Incrementally rebuild OHLCV candles.

        Only processes new data since last build.
        """
        loop = asyncio.get_event_loop()

        def _build():
            results = {}
            for timeframe in self.config.timeframes:
                try:
                    stats = self._ohlcv_builder.build_ohlcv(
                        pair=pair,
                        incremental=True,  # Only process new data
                    )
                    results[timeframe] = stats
                except Exception as e:
                    logger.error(f"OHLCV build error for {pair} {timeframe}: {e}")
                    results[timeframe] = {"error": str(e)}

            return results

        return await loop.run_in_executor(None, _build)

    def get_stats(self) -> Dict[str, Any]:
        """Get sync statistics."""
        return {
            **self._sync_stats,
            "last_trade_sync": {
                pair: ts.isoformat() if ts else None
                for pair, ts in self._last_trade_sync.items()
            },
            "last_ohlcv_rebuild": {
                pair: ts.isoformat() if ts else None
                for pair, ts in self._last_ohlcv_rebuild.items()
            },
        }


async def create_data_sync_task(
    service: DataSyncService,
    shutdown_event: asyncio.Event,
    interval: float,
) -> None:
    """
    Background task that runs periodic data syncs.

    This is designed to be added to the orchestrator's task list.

    Args:
        service: DataSyncService instance
        shutdown_event: Event signaling shutdown
        interval: Seconds between sync attempts
    """
    await service.start()

    try:
        while not shutdown_event.is_set():
            # Sync all configured pairs
            for pair in service.config.pairs:
                if shutdown_event.is_set():
                    break

                try:
                    await service.sync_once(pair)
                except Exception as e:
                    logger.error(f"Unexpected error in data sync for {pair}: {e}")

            # Wait for next sync cycle or shutdown
            try:
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=interval,
                )
                # If we get here, shutdown was signaled
                break
            except asyncio.TimeoutError:
                # Normal timeout, continue to next sync
                continue

    except asyncio.CancelledError:
        logger.debug("Data sync task cancelled")
    finally:
        await service.stop()
