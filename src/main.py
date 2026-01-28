#!/usr/bin/env python3
"""
Kraken Grid Trading Bot - Main Entry Point.

Usage:
    python -m src.main config/config.yaml
    python -m src.main config/config.yaml --paper
    python -m src.main config/config.yaml --fresh
    python -m src.main config/config.yaml --dry-run

Environment:
    KRAKEN_API_KEY: Kraken API key (required for live trading)
    KRAKEN_API_SECRET: Kraken API secret (required for live trading)
    PAPER_TRADING: Set to 'true' to force paper trading mode
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

from config.settings import BotConfig
from src.utils.config_loader import ConfigLoader
from src.core import Orchestrator, OrchestratorConfig, ProcessLock, ProcessLockError
from src.core.orchestrator import OrchestratorState


def setup_logging(level: str, log_file: Optional[str] = None) -> None:
    """
    Configure logging for the bot.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
    """
    # Create formatters
    console_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    file_format = (
        "%(asctime)s [%(levelname)s] %(name)s "
        "(%(filename)s:%(lineno)d): %(message)s"
    )

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(logging.Formatter(console_format))
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(logging.Formatter(file_format))
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Kraken Adaptive Grid Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with paper trading (recommended for testing)
  python -m src.main config/config.yaml --paper

  # Run with fresh state (clear previous session)
  python -m src.main config/config.yaml --fresh

  # Validate configuration without running
  python -m src.main config/config.yaml --dry-run

  # Run with debug logging
  python -m src.main config/config.yaml --log-level DEBUG
        """,
    )

    parser.add_argument(
        "config",
        type=str,
        help="Path to configuration YAML file",
    )

    parser.add_argument(
        "--paper",
        action="store_true",
        help="Force paper trading mode (orders validated but not executed)",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear saved state and start fresh",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and exit without trading",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (default: none, console only)",
    )

    parser.add_argument(
        "--force-start",
        action="store_true",
        help="Force start even if another instance appears to be running",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Skip state restoration (but keep history)",
    )

    parser.add_argument(
        "--max-state-age",
        type=int,
        default=86400,
        help="Maximum state age in seconds for resume (default: 86400 = 24h)",
    )

    return parser.parse_args()


def print_banner() -> None:
    """Print startup banner."""
    banner = """
    ╔═══════════════════════════════════════════════════╗
    ║     Kraken Adaptive Grid Trading Bot              ║
    ║     AI-Powered Market Regime Detection            ║
    ╚═══════════════════════════════════════════════════╝
    """
    print(banner)


def print_config_summary(config: BotConfig) -> None:
    """Print configuration summary."""
    print("\nConfiguration:")
    print(f"  Trading pair: {config.trading.pair}")
    print(f"  Paper trading: {config.paper_trading}")
    print(f"  Grid levels: {config.grid.num_levels}")
    print(f"  Grid range: {config.grid.range_percent}%")
    print(f"  Order size: ${config.grid.order_size_quote}")
    print(f"  Total capital: ${config.grid.total_capital_required}")
    print(f"  Max drawdown: {config.risk.max_drawdown_percent}%")
    print(f"  Stop-loss: {config.risk.stop_loss_percent}%")
    print(f"  Min confidence: {config.risk.min_confidence}")
    print()


async def main(args: argparse.Namespace) -> int:
    """
    Main async entry point.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    logger = logging.getLogger(__name__)
    process_lock = None

    # Load configuration
    try:
        loader = ConfigLoader(args.config)
        config = loader.load()
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        return 1
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1

    # Override paper trading if specified
    if args.paper:
        config.paper_trading = True

    # Print summary
    print_config_summary(config)

    # Validate configuration
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return 1

    logger.info("Configuration validated successfully")

    # Dry run - just validate and exit
    if args.dry_run:
        print("Dry run complete - configuration is valid")
        return 0

    # Acquire process lock to prevent multiple instances
    lock_path = getattr(config.recovery, "pid_lock_path", "data/trading.pid") if hasattr(config, "recovery") else "data/trading.pid"
    process_lock = ProcessLock(lock_path)

    try:
        process_lock.acquire(force=args.force_start)
        logger.info("Acquired process lock")
    except ProcessLockError as e:
        print(f"Error: {e}")
        print("Use --force-start to override (ensure previous instance is dead)")
        return 1

    orchestrator = None
    try:
        # Safety check for live trading
        if not config.paper_trading:
            print("\n" + "=" * 50)
            print("WARNING: LIVE TRADING MODE")
            print("Real orders will be placed on Kraken.")
            print("=" * 50)

            # Check for API credentials
            import os
            if not os.getenv("KRAKEN_API_KEY") or not os.getenv("KRAKEN_API_SECRET"):
                print("Error: KRAKEN_API_KEY and KRAKEN_API_SECRET must be set")
                return 1

            print("\nPress Ctrl+C within 5 seconds to abort...")
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                print("\nAborted")
                return 0
            print("Starting live trading...")

        # Create orchestrator config with recovery settings
        orch_config = OrchestratorConfig(
            max_state_age_seconds=args.max_state_age,
            skip_state_restore=args.no_resume,
        )

        # Create orchestrator
        orchestrator = Orchestrator(config, orch_config)

        # Track shutdown state for three-stage handling
        shutdown_requested = False
        force_shutdown_requested = False

        # Setup signal handlers with three-stage shutdown
        def signal_handler(sig, frame):
            nonlocal shutdown_requested, force_shutdown_requested

            if force_shutdown_requested:
                # Third signal - immediate exit
                logger.critical("Third signal received - immediate exit")
                sys.exit(1)

            if shutdown_requested:
                # Second signal - emergency stop (save state only, skip order cancel)
                logger.warning("Second signal received - emergency shutdown (saving state only)")
                force_shutdown_requested = True
                asyncio.create_task(orchestrator.stop(cancel_orders=False, emergency=True))
                return

            # First signal - graceful shutdown
            logger.info(f"Received signal {sig}, initiating graceful shutdown...")
            shutdown_requested = True
            asyncio.create_task(orchestrator.stop())

        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Initialize
        logger.info("Initializing orchestrator...")
        await orchestrator.initialize()

        # Clear state if requested
        if args.fresh:
            if orchestrator.state_manager:
                orchestrator.state_manager.clear_state()
                logger.info("Cleared saved state")

        # Start
        logger.info("Starting orchestrator...")
        await orchestrator.start()

        logger.info("Bot is running. Press Ctrl+C to stop.")

        # Run until stopped
        while orchestrator.state != OrchestratorState.STOPPED:
            await asyncio.sleep(1)

        logger.info("Orchestrator stopped normally")
        return 0

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        if orchestrator:
            await orchestrator.stop()
        return 0

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        try:
            if orchestrator:
                await orchestrator.stop(cancel_orders=True)
        except Exception as stop_error:
            logger.error(f"Error during shutdown: {stop_error}")
        return 1

    finally:
        # Always release the process lock
        if process_lock:
            process_lock.release()
            logger.info("Released process lock")


def run() -> None:
    """Synchronous entry point."""
    # Print banner
    print_banner()

    # Parse arguments
    args = parse_args()

    # Setup logging
    log_file = args.log_file or "logs/trading.log"
    setup_logging(args.log_level, log_file)

    logger = logging.getLogger(__name__)
    logger.info("Starting Kraken Grid Trading Bot")
    logger.info(f"Config: {args.config}")
    logger.info(f"Log level: {args.log_level}")

    # Run async main
    try:
        exit_code = asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Interrupted")
        exit_code = 0

    logger.info(f"Exiting with code {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    run()
