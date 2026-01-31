"""
OHLCV candle builder from raw trade data.

Aggregates raw trades into OHLCV candles at multiple timeframes:
- 1m, 5m, 15m, 1h, 4h

Includes additional metrics:
- VWAP (volume-weighted average price)
- Buy/Sell volume (taker direction)
- Trade count

Why build from trades instead of using OHLC endpoint:
- OHLC endpoint limited to 720 candles
- Trades endpoint has unlimited history
- Can compute custom metrics (VWAP, directional volume)

Usage:
    builder = OHLCVBuilder(trades_path, ohlcv_path)
    builder.build_ohlcv("XBTUSD")
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class OHLCVBuilder:
    """
    Aggregates raw trades into OHLCV candles at multiple timeframes.

    Approach:
    1. Build 1-minute candles directly from trades
    2. Resample 1m candles to higher timeframes

    This is more efficient than rebuilding from trades for each timeframe.

    Schema:
    - timestamp: timestamp[us, tz=UTC]
    - open: float64
    - high: float64
    - low: float64
    - close: float64
    - volume: float64
    - trade_count: int64
    - vwap: float64
    - buy_volume: float64
    - sell_volume: float64
    """

    OHLCV_SCHEMA = pa.schema([
        ("timestamp", pa.timestamp("us", tz="UTC")),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.float64()),
        ("trade_count", pa.int64()),
        ("vwap", pa.float64()),
        ("buy_volume", pa.float64()),
        ("sell_volume", pa.float64()),
    ])

    TIMEFRAME_MINUTES = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "1h": 60,
        "4h": 240,
    }

    def __init__(
        self,
        trades_path: Path,
        ohlcv_path: Path,
        timeframes: Optional[List[str]] = None,
    ):
        """
        Initialize OHLCV builder.

        Args:
            trades_path: Path to raw trades directory
            ohlcv_path: Path for storing OHLCV data
            timeframes: List of timeframes to build (default: all)
        """
        self.trades_path = Path(trades_path)
        self.ohlcv_path = Path(ohlcv_path)
        self.timeframes = timeframes or list(self.TIMEFRAME_MINUTES.keys())

        self.ohlcv_path.mkdir(parents=True, exist_ok=True)

    def _load_trades(
        self,
        pair: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load raw trades from Parquet storage."""
        pair_path = self.trades_path / pair

        if not pair_path.exists():
            raise ValueError(f"No trades found for {pair}")

        # Get all date partitions
        date_dirs = sorted(pair_path.glob("date=*"))

        if start_date:
            date_dirs = [d for d in date_dirs if d.name.split("=")[1] >= start_date]
        if end_date:
            date_dirs = [d for d in date_dirs if d.name.split("=")[1] <= end_date]

        if not date_dirs:
            return pd.DataFrame()

        # Read all partitions
        # Use ParquetFile to read single files without dataset/schema inference
        dfs = []
        for date_dir in date_dirs:
            file_path = date_dir / "trades.parquet"
            if file_path.exists():
                # Read single file directly to avoid schema merge issues
                pf = pq.ParquetFile(file_path)
                df = pf.read().to_pandas()
                # Drop 'date' column if present - it's redundant with partition
                # and can have inconsistent types across files
                if "date" in df.columns:
                    df = df.drop(columns=["date"])
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.debug(f"Loaded {len(df):,} trades for {pair}")
        return df

    def _trades_to_1m_ohlcv(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate trades to 1-minute OHLCV candles.

        Includes:
        - Standard OHLCV
        - VWAP: Volume-weighted average price
        - Buy/Sell volume: Directional volume
        - Trade count
        """
        if trades.empty:
            return pd.DataFrame()

        # Ensure timestamp is the index
        df = trades.set_index("timestamp")

        # Calculate dollar volume for VWAP
        df["dollar_volume"] = df["price"] * df["volume"]

        # Directional volume
        df["buy_volume"] = df["volume"].where(df["side"] == "b", 0)
        df["sell_volume"] = df["volume"].where(df["side"] == "s", 0)

        # Resample to 1-minute using standard OHLCV rules
        ohlcv = df.resample("1min").agg({
            "price": ["first", "max", "min", "last"],
            "volume": "sum",
            "dollar_volume": "sum",
            "buy_volume": "sum",
            "sell_volume": "sum",
        })

        # Flatten column names
        ohlcv.columns = [
            "open", "high", "low", "close",
            "volume", "dollar_volume",
            "buy_volume", "sell_volume"
        ]

        # Add trade count
        ohlcv["trade_count"] = df.resample("1min").size()

        # Calculate VWAP
        ohlcv["vwap"] = ohlcv["dollar_volume"] / ohlcv["volume"]
        # Handle zero volume candles
        ohlcv["vwap"] = ohlcv["vwap"].fillna(ohlcv["close"])

        # Drop intermediate column
        ohlcv = ohlcv.drop(columns=["dollar_volume"])

        # Drop rows with no trades (NaN in OHLC)
        ohlcv = ohlcv.dropna(subset=["open"])

        # Reset index
        ohlcv = ohlcv.reset_index()

        return ohlcv

    def _resample_ohlcv(
        self,
        ohlcv_1m: pd.DataFrame,
        target_timeframe: str,
    ) -> pd.DataFrame:
        """
        Resample 1-minute OHLCV to a higher timeframe.

        OHLCV resampling rules:
        - Open: First value in period
        - High: Maximum high
        - Low: Minimum low
        - Close: Last value in period
        - Volume: Sum
        """
        if ohlcv_1m.empty:
            return pd.DataFrame()

        minutes = self.TIMEFRAME_MINUTES[target_timeframe]
        freq = f"{minutes}min"

        df = ohlcv_1m.set_index("timestamp")

        # Resample with proper aggregation
        resampled = df.resample(freq, label="left", closed="left").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "trade_count": "sum",
            "buy_volume": "sum",
            "sell_volume": "sum",
        })

        # Recalculate VWAP for the period
        # Need to weight by volume from 1m candles
        vwap_weighted = (df["vwap"] * df["volume"]).resample(freq).sum()
        volume_sum = df["volume"].resample(freq).sum()
        resampled["vwap"] = vwap_weighted / volume_sum
        resampled["vwap"] = resampled["vwap"].fillna(resampled["close"])

        # Drop incomplete candles
        resampled = resampled.dropna(subset=["open"])
        resampled = resampled.reset_index()

        return resampled

    def _write_ohlcv(
        self,
        ohlcv: pd.DataFrame,
        pair: str,
        timeframe: str,
        append: bool = False,
    ) -> None:
        """Write OHLCV data to Parquet."""
        if ohlcv.empty:
            return

        tf_path = self.ohlcv_path / timeframe / pair
        tf_path.mkdir(parents=True, exist_ok=True)

        file_path = tf_path / "ohlcv.parquet"

        table = pa.Table.from_pandas(
            ohlcv,
            schema=self.OHLCV_SCHEMA,
            preserve_index=False,
        )

        if append and file_path.exists():
            # Read existing and append
            existing = pq.read_table(file_path).to_pandas()
            combined = pd.concat([existing, ohlcv], ignore_index=True)

            # Deduplicate by timestamp
            combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
            combined = combined.sort_values("timestamp").reset_index(drop=True)

            table = pa.Table.from_pandas(
                combined,
                schema=self.OHLCV_SCHEMA,
                preserve_index=False,
            )

        pq.write_table(table, file_path, compression="snappy")

    def _get_last_processed_date(self, pair: str) -> Optional[str]:
        """Get the last processed date for incremental updates."""
        file_path = self.ohlcv_path / "1m" / pair / "ohlcv.parquet"

        if file_path.exists():
            df = pq.read_table(file_path).to_pandas()
            if not df.empty:
                last_ts = df["timestamp"].max()
                return last_ts.strftime("%Y-%m-%d")

        return None

    def build_ohlcv(
        self,
        pair: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        incremental: bool = False,
    ) -> Dict[str, Any]:
        """
        Build OHLCV data at all timeframes for a trading pair.

        Args:
            pair: Trading pair (e.g., "XBTUSD")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            incremental: If True, only process new data since last build

        Returns:
            Dict with build statistics
        """
        logger.info(f"Building OHLCV for {pair}")

        # If incremental, find last processed date
        if incremental:
            last_date = self._get_last_processed_date(pair)
            if last_date:
                start_date = last_date
                logger.info(f"Incremental mode: starting from {start_date}")

        # Load trades
        trades = self._load_trades(pair, start_date, end_date)

        if trades.empty:
            logger.warning(f"No trades found for {pair}")
            return {"pair": pair, "candles": {}, "status": "no_data"}

        logger.info(f"Loaded {len(trades):,} trades")

        # Build 1-minute candles first
        ohlcv_1m = self._trades_to_1m_ohlcv(trades)
        logger.info(f"Built {len(ohlcv_1m):,} 1-minute candles")

        # Store 1m and build other timeframes
        results = {}

        for tf in self.timeframes:
            if tf == "1m":
                ohlcv = ohlcv_1m
            else:
                ohlcv = self._resample_ohlcv(ohlcv_1m, tf)

            # Write to Parquet
            self._write_ohlcv(ohlcv, pair, tf, append=incremental)
            results[tf] = len(ohlcv)

            logger.info(f"  {tf}: {len(ohlcv):,} candles")

        return {
            "pair": pair,
            "candles": results,
            "status": "success",
            "incremental": incremental,
        }

    def load_ohlcv(
        self,
        pair: str,
        timeframe: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data for a pair and timeframe.

        Args:
            pair: Trading pair
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h)
            start: Start timestamp (optional)
            end: End timestamp (optional)

        Returns:
            DataFrame with OHLCV data
        """
        file_path = self.ohlcv_path / timeframe / pair / "ohlcv.parquet"

        if not file_path.exists():
            raise ValueError(f"No OHLCV data for {pair} at {timeframe}")

        df = pq.read_table(file_path).to_pandas()

        if start is not None:
            df = df[df["timestamp"] >= start]
        if end is not None:
            df = df[df["timestamp"] <= end]

        return df.sort_values("timestamp").reset_index(drop=True)

    def get_summary(self, pair: str) -> Dict[str, Any]:
        """
        Get summary of OHLCV data for a pair.

        Returns:
            Dict with candle counts per timeframe
        """
        summary = {}

        for tf in self.timeframes:
            file_path = self.ohlcv_path / tf / pair / "ohlcv.parquet"
            if file_path.exists():
                metadata = pq.read_metadata(file_path)
                summary[tf] = {
                    "candles": metadata.num_rows,
                    "file_size_mb": file_path.stat().st_size / (1024 * 1024),
                }
            else:
                summary[tf] = {"candles": 0, "file_size_mb": 0}

        return summary
