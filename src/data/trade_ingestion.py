"""
Historical trade data ingestion from Kraken.

Downloads all historical trades using the /Trades endpoint (unlimited history)
and stores them in Parquet format partitioned by date.

Features:
- Pagination using 'since' parameter
- Checkpoint support for resumable downloads
- Parquet storage with date partitioning
- Progress tracking with callbacks

Usage:
    client = KrakenPublicClient()
    pipeline = TradeIngestionPipeline(client, Path("data/raw"))
    stats = pipeline.download_historical("XBTUSD", start_from="0")
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.kraken_client import KrakenPublicClient

logger = logging.getLogger(__name__)


class TradeIngestionPipeline:
    """
    Downloads and stores historical trade data from Kraken.

    Data is stored in Parquet format with the following schema:
    - timestamp_ns: int64 (Unix timestamp in nanoseconds)
    - timestamp: timestamp[us, tz=UTC] (datetime for easy querying)
    - price: float64
    - volume: float64
    - side: string ('b' = buy, 's' = sell)
    - order_type: string ('m' = market, 'l' = limit)
    - date: string (YYYY-MM-DD for partitioning)

    Storage structure:
        data/raw/{pair}/date={YYYY-MM-DD}/trades.parquet
    """

    PARQUET_SCHEMA = pa.schema([
        ("timestamp_ns", pa.int64()),
        ("timestamp", pa.timestamp("us", tz="UTC")),
        ("price", pa.float64()),
        ("volume", pa.float64()),
        ("side", pa.string()),
        ("order_type", pa.string()),
        ("date", pa.string()),
    ])

    def __init__(
        self,
        client: KrakenPublicClient,
        storage_path: Path,
        checkpoint_path: Optional[Path] = None,
        batch_size: int = 50_000,
    ):
        """
        Initialize trade ingestion pipeline.

        Args:
            client: Kraken API client
            storage_path: Base path for storing raw trades
            checkpoint_path: Path for checkpoint file (default: storage_path/checkpoint.json)
            batch_size: Number of trades to buffer before writing
        """
        self.client = client
        self.storage_path = Path(storage_path)
        self.checkpoint_path = checkpoint_path or (self.storage_path / "checkpoint.json")
        self.batch_size = batch_size

        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _load_checkpoint(self, pair: str) -> Optional[str]:
        """Load last processed trade ID for a pair."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, "r") as f:
                checkpoints = json.load(f)
                return checkpoints.get(pair)
        return None

    def _save_checkpoint(self, pair: str, last_id: str) -> None:
        """Save progress checkpoint."""
        checkpoints = {}
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, "r") as f:
                checkpoints = json.load(f)

        checkpoints[pair] = last_id

        with open(self.checkpoint_path, "w") as f:
            json.dump(checkpoints, f, indent=2)

    def _parse_trades(self, raw_trades: List[List]) -> pd.DataFrame:
        """
        Parse raw trade data from Kraken API response.

        Raw format: [price, volume, time, side, type, misc]
        """
        if not raw_trades:
            return pd.DataFrame()

        records = []
        for trade in raw_trades:
            price = float(trade[0])
            volume = float(trade[1])
            timestamp = float(trade[2])  # Unix timestamp with decimals
            side = trade[3]
            order_type = trade[4]
            # misc = trade[5]  # Not stored

            # Convert to nanoseconds for precise storage
            timestamp_ns = int(timestamp * 1_000_000_000)

            # Create datetime for partitioning
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            date_str = dt.strftime("%Y-%m-%d")

            records.append({
                "timestamp_ns": timestamp_ns,
                "timestamp": pd.Timestamp(timestamp, unit="s", tz="UTC").as_unit("us"),
                "price": price,
                "volume": volume,
                "side": side,
                "order_type": order_type,
                "date": date_str,
            })

        return pd.DataFrame(records)

    def _write_batch(self, df: pd.DataFrame, pair: str) -> None:
        """
        Write a batch of trades to Parquet, partitioned by date.

        Handles deduplication by timestamp_ns.
        """
        if df.empty:
            return

        pair_path = self.storage_path / pair
        pair_path.mkdir(exist_ok=True)

        # Group by date for partitioning
        for date, group in df.groupby("date"):
            date_path = pair_path / f"date={date}"
            date_path.mkdir(exist_ok=True)

            file_path = date_path / "trades.parquet"

            # Convert to PyArrow table
            table = pa.Table.from_pandas(
                group,
                schema=self.PARQUET_SCHEMA,
                preserve_index=False,
            )

            if file_path.exists():
                # Read existing and merge
                existing = pq.read_table(file_path)
                combined = pa.concat_tables([existing, table])

                # Deduplicate by timestamp_ns
                df_combined = combined.to_pandas()
                df_combined = df_combined.drop_duplicates(
                    subset=["timestamp_ns"],
                    keep="last"
                )
                df_combined = df_combined.sort_values("timestamp_ns")

                table = pa.Table.from_pandas(
                    df_combined,
                    schema=self.PARQUET_SCHEMA,
                    preserve_index=False,
                )

            pq.write_table(
                table,
                file_path,
                compression="snappy",
                row_group_size=100_000,
            )

    def download_historical(
        self,
        pair: str,
        start_from: Optional[str] = None,
        end_timestamp: Optional[float] = None,
        progress_callback: Optional[Callable[[int, float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Download all historical trades for a pair.

        Args:
            pair: Kraken pair name (e.g., "XBTUSD")
            start_from: Trade ID to start from. Use "0" for beginning of history.
                       If None, resumes from checkpoint or starts from "0".
            end_timestamp: Stop when reaching this Unix timestamp (optional)
            progress_callback: Called with (total_trades, last_timestamp) periodically

        Returns:
            Dict with download statistics:
            - pair: Trading pair
            - total_trades: Number of trades downloaded
            - last_id: Last trade ID for resuming
            - last_timestamp: Timestamp of last trade
            - start_time: Download start time
            - end_time: Download end time
        """
        # Determine starting point
        if start_from is None:
            start_from = self._load_checkpoint(pair) or "0"

        logger.info(f"Starting download for {pair} from since={start_from}")

        start_time = datetime.now(timezone.utc)
        total_trades = 0
        batch_buffer: List[pd.DataFrame] = []
        last_id = start_from
        last_timestamp: Optional[float] = None

        try:
            while True:
                # Fetch trades
                trades, new_last_id = self.client.get_trades(pair, since=last_id)

                if not trades:
                    logger.info("No more trades available")
                    break

                # Check if we've passed the end timestamp
                if end_timestamp:
                    last_trade_time = float(trades[-1][2])
                    if last_trade_time >= end_timestamp:
                        # Filter trades beyond end_timestamp
                        trades = [t for t in trades if float(t[2]) < end_timestamp]
                        if not trades:
                            logger.info(f"Reached end timestamp {end_timestamp}")
                            break

                # Parse trades
                df = self._parse_trades(trades)
                batch_buffer.append(df)
                total_trades += len(trades)
                last_timestamp = float(trades[-1][2])

                # Write batch when buffer is large enough
                buffer_size = sum(len(b) for b in batch_buffer)
                if buffer_size >= self.batch_size:
                    combined = pd.concat(batch_buffer, ignore_index=True)
                    self._write_batch(combined, pair)
                    batch_buffer = []
                    self._save_checkpoint(pair, new_last_id)

                    dt_str = datetime.fromtimestamp(
                        last_timestamp, tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M")

                    logger.info(
                        f"Progress: {total_trades:,} trades, "
                        f"last: {dt_str}"
                    )

                if progress_callback:
                    progress_callback(total_trades, last_timestamp)

                # Check for no new data (same last_id means we're caught up)
                if new_last_id == last_id:
                    logger.info("Reached end of available data")
                    break

                last_id = new_last_id

        finally:
            # Flush remaining buffer
            if batch_buffer:
                combined = pd.concat(batch_buffer, ignore_index=True)
                self._write_batch(combined, pair)
                self._save_checkpoint(pair, last_id)
                logger.info(f"Flushed final batch of {len(combined)} trades")

        end_time = datetime.now(timezone.utc)

        return {
            "pair": pair,
            "total_trades": total_trades,
            "last_id": last_id,
            "last_timestamp": last_timestamp,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
        }

    def get_latest_trade_id(self, pair: str) -> Optional[str]:
        """
        Get the latest trade ID from stored data for incremental updates.

        Scans the most recent partition to find the last trade.
        """
        pair_path = self.storage_path / pair

        if not pair_path.exists():
            return None

        # Find most recent partition
        date_dirs = sorted(pair_path.glob("date=*"), reverse=True)
        if not date_dirs:
            return None

        latest_file = date_dirs[0] / "trades.parquet"
        if latest_file.exists():
            df = pq.read_table(latest_file).to_pandas()
            if not df.empty:
                max_ts = df["timestamp_ns"].max()
                return str(max_ts)

        return None

    def get_data_summary(self, pair: str) -> Dict[str, Any]:
        """
        Get summary of stored data for a pair.

        Returns:
            Dict with:
            - total_trades: Total number of stored trades
            - date_range: (earliest_date, latest_date)
            - partitions: Number of date partitions
        """
        pair_path = self.storage_path / pair

        if not pair_path.exists():
            return {"total_trades": 0, "date_range": None, "partitions": 0}

        date_dirs = sorted(pair_path.glob("date=*"))
        if not date_dirs:
            return {"total_trades": 0, "date_range": None, "partitions": 0}

        total_trades = 0
        for date_dir in date_dirs:
            file_path = date_dir / "trades.parquet"
            if file_path.exists():
                metadata = pq.read_metadata(file_path)
                total_trades += metadata.num_rows

        earliest = date_dirs[0].name.split("=")[1]
        latest = date_dirs[-1].name.split("=")[1]

        return {
            "total_trades": total_trades,
            "date_range": (earliest, latest),
            "partitions": len(date_dirs),
        }

    def load_trades(
        self,
        pair: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load trades from storage.

        Args:
            pair: Trading pair
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with trades
        """
        pair_path = self.storage_path / pair

        if not pair_path.exists():
            return pd.DataFrame()

        # Get partitions within date range
        date_dirs = sorted(pair_path.glob("date=*"))

        if start_date:
            date_dirs = [d for d in date_dirs if d.name.split("=")[1] >= start_date]
        if end_date:
            date_dirs = [d for d in date_dirs if d.name.split("=")[1] <= end_date]

        if not date_dirs:
            return pd.DataFrame()

        # Read and concatenate
        dfs = []
        for date_dir in date_dirs:
            file_path = date_dir / "trades.parquet"
            if file_path.exists():
                dfs.append(pq.read_table(file_path).to_pandas())

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        return df.sort_values("timestamp").reset_index(drop=True)
