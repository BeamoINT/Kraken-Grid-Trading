#!/usr/bin/env python3
"""
CLI script to download historical trade data from Kraken.

Downloads unlimited historical trades using the /Trades endpoint
and stores them in Parquet format for ML training.

Usage:
    # Download all history for BTC/USD
    python scripts/download_historical.py --pairs XBTUSD

    # Download from beginning of time
    python scripts/download_historical.py --pairs XBTUSD --since 0

    # Resume interrupted download
    python scripts/download_historical.py --pairs XBTUSD --resume

    # Build OHLCV after download
    python scripts/download_historical.py --pairs XBTUSD --build-ohlcv

    # Show data summary
    python scripts/download_historical.py --pairs XBTUSD --summary
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone

from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.kraken_client import KrakenPublicClient
from src.data.trade_ingestion import TradeIngestionPipeline
from src.data.ohlcv_builder import OHLCVBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


def create_progress_callback():
    """Create a progress callback with tqdm."""
    pbar = None
    last_ts = [None]  # Use list to allow mutation in closure

    def callback(total_trades: int, last_timestamp: float):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(
                desc="Downloading trades",
                unit=" trades",
                dynamic_ncols=True,
            )

        # Calculate progress increment
        if last_ts[0] is None:
            increment = total_trades
        else:
            increment = total_trades - pbar.n

        pbar.update(increment)

        # Update description with current date
        if last_timestamp:
            dt = datetime.fromtimestamp(last_timestamp, tz=timezone.utc)
            pbar.set_postfix_str(dt.strftime("%Y-%m-%d"))

        last_ts[0] = last_timestamp

    def close():
        nonlocal pbar
        if pbar is not None:
            pbar.close()

    return callback, close


def download_trades(args):
    """Download historical trades."""
    client = KrakenPublicClient(
        rate_limit_delay=1.0,  # Safe rate limit
        max_retries=5,
        retry_delay=10.0,
    )

    storage_path = Path(args.output)
    pipeline = TradeIngestionPipeline(
        client=client,
        storage_path=storage_path,
        batch_size=50_000,
    )

    for pair in args.pairs:
        logger.info(f"{'='*50}")
        logger.info(f"Downloading {pair}")
        logger.info(f"{'='*50}")

        # Determine starting point
        start_from = args.since
        if args.resume:
            start_from = None  # Will use checkpoint

        # Create progress callback
        progress_cb, close_cb = create_progress_callback()

        try:
            stats = pipeline.download_historical(
                pair=pair,
                start_from=start_from,
                progress_callback=progress_cb,
            )

            close_cb()

            logger.info(f"\nDownload complete for {pair}:")
            logger.info(f"  Total trades: {stats['total_trades']:,}")
            logger.info(f"  Duration: {stats['duration_seconds']:.1f} seconds")

            if stats['last_timestamp']:
                dt = datetime.fromtimestamp(
                    stats['last_timestamp'],
                    tz=timezone.utc
                )
                logger.info(f"  Last trade: {dt.isoformat()}")

        except KeyboardInterrupt:
            close_cb()
            logger.info("\nDownload interrupted. Progress has been saved.")
            logger.info("Run with --resume to continue.")
            break

        except Exception as e:
            close_cb()
            logger.error(f"Download failed for {pair}: {e}")
            raise

    client.close()

    # Build OHLCV if requested
    if args.build_ohlcv:
        build_ohlcv(args)


def build_ohlcv(args):
    """Build OHLCV candles from downloaded trades."""
    trades_path = Path(args.output)
    ohlcv_path = Path(args.ohlcv_output)

    builder = OHLCVBuilder(
        trades_path=trades_path,
        ohlcv_path=ohlcv_path,
        timeframes=["1m", "5m", "15m", "1h", "4h"],
    )

    for pair in args.pairs:
        logger.info(f"Building OHLCV for {pair}...")

        try:
            # Use chunked processing for large datasets (memory-efficient)
            if args.chunked:
                stats = builder.build_ohlcv_chunked(
                    pair=pair,
                    chunk_months=args.chunk_months,
                )
            else:
                stats = builder.build_ohlcv(
                    pair=pair,
                    incremental=args.incremental,
                )

            logger.info(f"OHLCV build complete for {pair}:")
            for tf, count in stats.get("candles", {}).items():
                logger.info(f"  {tf}: {count:,} candles")

        except Exception as e:
            logger.error(f"OHLCV build failed for {pair}: {e}")
            raise


def show_summary(args):
    """Show summary of stored data."""
    trades_path = Path(args.output)
    ohlcv_path = Path(args.ohlcv_output)

    # Trade ingestion summary
    client = KrakenPublicClient()
    pipeline = TradeIngestionPipeline(
        client=client,
        storage_path=trades_path,
    )

    # OHLCV builder for summary
    builder = OHLCVBuilder(
        trades_path=trades_path,
        ohlcv_path=ohlcv_path,
    )

    for pair in args.pairs:
        logger.info(f"\n{'='*50}")
        logger.info(f"Summary for {pair}")
        logger.info(f"{'='*50}")

        # Raw trades summary
        trade_summary = pipeline.get_data_summary(pair)
        logger.info("\nRaw Trades:")
        logger.info(f"  Total trades: {trade_summary['total_trades']:,}")
        if trade_summary['date_range']:
            logger.info(
                f"  Date range: {trade_summary['date_range'][0]} to "
                f"{trade_summary['date_range'][1]}"
            )
        logger.info(f"  Partitions: {trade_summary['partitions']}")

        # OHLCV summary
        ohlcv_summary = builder.get_summary(pair)
        logger.info("\nOHLCV Candles:")
        for tf, info in ohlcv_summary.items():
            if info['candles'] > 0:
                logger.info(
                    f"  {tf}: {info['candles']:,} candles "
                    f"({info['file_size_mb']:.1f} MB)"
                )
            else:
                logger.info(f"  {tf}: No data")

    client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Download historical trade data from Kraken",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download BTC/USD trades
  python scripts/download_historical.py --pairs XBTUSD

  # Download from beginning and build OHLCV
  python scripts/download_historical.py --pairs XBTUSD --since 0 --build-ohlcv

  # Resume interrupted download
  python scripts/download_historical.py --pairs XBTUSD --resume

  # Show data summary
  python scripts/download_historical.py --pairs XBTUSD --summary
        """,
    )

    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="Trading pairs to download (e.g., XBTUSD ETHUSD)",
    )

    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Trade ID to start from. Use '0' for all history.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory for raw trades (default: data/raw)",
    )

    parser.add_argument(
        "--ohlcv-output",
        type=str,
        default="data/ohlcv",
        help="Output directory for OHLCV data (default: data/ohlcv)",
    )

    parser.add_argument(
        "--build-ohlcv",
        action="store_true",
        help="Build OHLCV candles after downloading",
    )

    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental OHLCV build (only process new data)",
    )

    parser.add_argument(
        "--chunked",
        action="store_true",
        help="Use memory-efficient chunked processing for OHLCV build (recommended for large datasets)",
    )

    parser.add_argument(
        "--chunk-months",
        type=int,
        default=3,
        dest="chunk_months",
        help="Months per chunk when using --chunked (default: 3)",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary of stored data and exit",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.summary:
        show_summary(args)
    else:
        download_trades(args)


if __name__ == "__main__":
    main()
