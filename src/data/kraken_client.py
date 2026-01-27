"""
Kraken public API client for historical data retrieval.

This client is used for:
- Downloading historical trade data (unlimited via /Trades endpoint)
- Getting OHLC data (limited to 720 candles via /OHLC endpoint)
- Getting current ticker prices

For private/trading endpoints, see src/api/kraken_rest.py
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.api.rate_limiter import SyncRateLimiter

logger = logging.getLogger(__name__)


class KrakenAPIError(Exception):
    """Exception raised for Kraken API errors."""

    def __init__(self, errors: List[str], response: Optional[Dict] = None):
        self.errors = errors
        self.response = response
        super().__init__(f"Kraken API error: {errors}")


@dataclass
class Trade:
    """Parsed trade from Kraken API."""

    price: float
    volume: float
    timestamp: float  # Unix timestamp with decimals
    side: str  # 'b' for buy, 's' for sell
    order_type: str  # 'm' for market, 'l' for limit
    misc: str


class KrakenPublicClient:
    """
    Client for Kraken public REST API.

    Handles:
    - Rate limiting (respects Kraken limits)
    - Retries with exponential backoff
    - Error parsing

    Usage:
        client = KrakenPublicClient()
        trades, last_id = client.get_trades("XBTUSD", since="0")
    """

    BASE_URL = "https://api.kraken.com/0/public"

    def __init__(
        self,
        rate_limit_delay: float = 1.0,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        timeout: int = 30,
    ):
        """
        Initialize Kraken public API client.

        Args:
            rate_limit_delay: Minimum delay between calls (seconds)
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries (seconds)
            timeout: Request timeout (seconds)
        """
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        # Rate limiter for public endpoints
        # Public endpoints are IP-limited, ~1 call/second is safe
        self._rate_limiter = SyncRateLimiter(
            max_counter=15,
            decay_rate=0.5,
            min_delay=rate_limit_delay,
        )

        # Configure session with retries
        self._session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("https://", adapter)

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make API request with rate limiting and error handling.

        Args:
            endpoint: API endpoint (e.g., "/Trades")
            params: Query parameters

        Returns:
            API result dict

        Raises:
            KrakenAPIError: If API returns error
            requests.RequestException: If request fails
        """
        self._rate_limiter.acquire()

        url = f"{self.BASE_URL}{endpoint}"

        for attempt in range(self._max_retries):
            try:
                response = self._session.get(
                    url,
                    params=params,
                    timeout=self._timeout,
                )
                response.raise_for_status()

                data = response.json()

                # Check for API errors
                if data.get("error"):
                    errors = data["error"]

                    # Check for rate limit error
                    if any("EAPI:Rate limit" in str(e) for e in errors):
                        logger.warning(
                            f"Rate limit hit, waiting {self._retry_delay * (attempt + 1)}s"
                        )
                        time.sleep(self._retry_delay * (attempt + 1))
                        continue

                    raise KrakenAPIError(errors, data)

                return data.get("result", {})

            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self._max_retries}): {e}"
                )
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
                else:
                    raise

        raise KrakenAPIError(["Max retries exceeded"])

    def get_trades(
        self,
        pair: str,
        since: Optional[str] = None,
        count: Optional[int] = None,
    ) -> Tuple[List[List], str]:
        """
        Fetch trades for a trading pair.

        The Trades endpoint has NO LIMIT on historical data - you can
        fetch the entire trade history back to 2013 by paginating
        with the 'since' parameter.

        Args:
            pair: Trading pair (e.g., "XBTUSD")
            since: Trade ID to start from (nanosecond timestamp string).
                   Use "0" to get earliest trades available.
            count: Number of trades to return (max 1000, optional)

        Returns:
            Tuple of (trades_list, last_trade_id)
            - trades_list: List of trade arrays [price, volume, time, side, type, misc]
            - last_trade_id: Use as 'since' for next pagination call
        """
        params: Dict[str, Any] = {"pair": pair}

        if since is not None:
            params["since"] = since
        if count is not None:
            params["count"] = min(count, 1000)

        result = self._make_request("/Trades", params)

        # Result format: {"XBTUSD": [[...], ...], "last": "123456789"}
        # Note: Kraken may return different pair key format
        trades = []
        for key, value in result.items():
            if key == "last":
                continue
            trades = value
            break

        last_id = result.get("last", "")

        logger.debug(
            f"Got {len(trades)} trades for {pair}, last_id={last_id}"
        )

        return trades, last_id

    def get_ohlc(
        self,
        pair: str,
        interval: int = 1,
        since: Optional[int] = None,
    ) -> Tuple[List[List], int]:
        """
        Fetch OHLC data for a trading pair.

        WARNING: Limited to 720 most recent candles.
        For historical data, use get_trades() and aggregate manually.

        Args:
            pair: Trading pair (e.g., "XBTUSD")
            interval: Timeframe in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
            since: Unix timestamp to return data after

        Returns:
            Tuple of (ohlc_list, last_timestamp)
            - ohlc_list: List of [time, open, high, low, close, vwap, volume, count]
            - last_timestamp: Last candle timestamp
        """
        params: Dict[str, Any] = {
            "pair": pair,
            "interval": interval,
        }

        if since is not None:
            params["since"] = since

        result = self._make_request("/OHLC", params)

        # Result format: {"XBTUSD": [[...], ...], "last": 123456789}
        ohlc = []
        last = 0
        for key, value in result.items():
            if key == "last":
                last = value
            else:
                ohlc = value

        return ohlc, last

    def get_ticker(
        self,
        pairs: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get current ticker data for pairs.

        Args:
            pairs: List of trading pairs

        Returns:
            Dict mapping pair to ticker data with keys:
            - 'a': ask [price, whole_lot_volume, lot_volume]
            - 'b': bid [price, whole_lot_volume, lot_volume]
            - 'c': last trade [price, volume]
            - 'v': volume [today, 24h]
            - 'p': vwap [today, 24h]
            - 't': trades [today, 24h]
            - 'l': low [today, 24h]
            - 'h': high [today, 24h]
            - 'o': open
        """
        params = {"pair": ",".join(pairs)}
        return self._make_request("/Ticker", params)

    def get_asset_pairs(
        self,
        pairs: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get trading pair information.

        Useful for getting:
        - Price precision (pair_decimals)
        - Lot precision (lot_decimals)
        - Minimum order size (ordermin)

        Args:
            pairs: Specific pairs to query (None for all)

        Returns:
            Dict mapping pair to info
        """
        params = {}
        if pairs:
            params["pair"] = ",".join(pairs)

        return self._make_request("/AssetPairs", params)

    def get_server_time(self) -> Dict[str, Any]:
        """
        Get server time.

        Useful for checking connectivity and time sync.

        Returns:
            Dict with 'unixtime' and 'rfc1123' keys
        """
        return self._make_request("/Time")

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
