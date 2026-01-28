"""
Kraken API Authentication.

Implements HMAC-SHA512 authentication for Kraken private API endpoints.
All private endpoints require authentication with API key and secret.

Usage:
    auth = KrakenAuth(api_key="...", api_secret="...")
    headers = auth.sign_request("/0/private/Balance", {"nonce": "..."})
"""

import base64
import hashlib
import hmac
import time
import urllib.parse
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class KrakenCredentials:
    """API credentials for Kraken authentication."""
    api_key: str
    api_secret: str

    def __post_init__(self):
        """Validate credentials format."""
        if not self.api_key:
            raise ValueError("API key is required")
        if not self.api_secret:
            raise ValueError("API secret is required")

        # API secret should be base64 encoded
        try:
            base64.b64decode(self.api_secret)
        except Exception:
            raise ValueError("API secret must be base64 encoded")


class KrakenAuth:
    """
    Authentication handler for Kraken private API.

    Kraken uses HMAC-SHA512 with the following signing scheme:
    1. Calculate SHA256 of (nonce + POST data)
    2. Decode API secret from base64
    3. Calculate HMAC-SHA512 of (URI path + SHA256 hash) using decoded secret
    4. Base64 encode the signature

    The nonce must be an always-increasing number (typically millisecond timestamp).
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
    ):
        """
        Initialize authentication handler.

        Args:
            api_key: Kraken API key
            api_secret: Kraken API secret (base64 encoded)
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._decoded_secret = base64.b64decode(api_secret)
        self._last_nonce = 0

    @classmethod
    def from_credentials(cls, credentials: KrakenCredentials) -> "KrakenAuth":
        """Create from credentials object."""
        return cls(
            api_key=credentials.api_key,
            api_secret=credentials.api_secret,
        )

    def generate_nonce(self) -> int:
        """
        Generate a unique, always-increasing nonce.

        Uses millisecond timestamp to ensure uniqueness.
        Kraken requires nonces to be strictly increasing within a session.

        Returns:
            Nonce value (milliseconds since epoch)
        """
        # Use millisecond timestamp
        nonce = int(time.time() * 1000)

        # Ensure strictly increasing
        if nonce <= self._last_nonce:
            nonce = self._last_nonce + 1

        self._last_nonce = nonce
        return nonce

    def sign_request(
        self,
        uri_path: str,
        data: Dict[str, Any],
        nonce: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Sign a request for Kraken private API.

        Args:
            uri_path: API endpoint path (e.g., "/0/private/Balance")
            data: POST data dictionary (will have nonce added if not present)
            nonce: Optional explicit nonce (generated if not provided)

        Returns:
            Headers dict with API-Key and API-Sign
        """
        # Generate nonce if not provided
        if nonce is None:
            nonce = self.generate_nonce()

        # Add nonce to data
        data_with_nonce = {"nonce": str(nonce), **data}

        # URL encode the data
        post_data = urllib.parse.urlencode(data_with_nonce)

        # Create message: nonce + post_data
        message = str(nonce) + post_data

        # SHA256 hash of message
        sha256_hash = hashlib.sha256(message.encode("utf-8")).digest()

        # HMAC-SHA512 of (uri_path + sha256_hash)
        hmac_message = uri_path.encode("utf-8") + sha256_hash
        signature = hmac.new(
            self._decoded_secret,
            hmac_message,
            hashlib.sha512,
        ).digest()

        # Base64 encode signature
        signature_b64 = base64.b64encode(signature).decode("utf-8")

        return {
            "API-Key": self._api_key,
            "API-Sign": signature_b64,
        }

    def get_signed_data(
        self,
        uri_path: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """
        Get both signed headers and data with nonce.

        Convenience method that returns both the headers and
        the data dictionary with nonce added.

        Args:
            uri_path: API endpoint path
            data: Optional POST data

        Returns:
            Tuple of (headers, data_with_nonce)
        """
        if data is None:
            data = {}

        nonce = self.generate_nonce()
        data_with_nonce = {"nonce": str(nonce), **data}
        headers = self.sign_request(uri_path, data, nonce)

        return headers, data_with_nonce


class NonceManager:
    """
    Thread-safe nonce manager for concurrent API calls.

    Ensures nonces are strictly increasing even with parallel requests.
    """

    def __init__(self):
        """Initialize nonce manager."""
        self._last_nonce = 0
        import threading
        self._lock = threading.Lock()

    def get_nonce(self) -> int:
        """
        Get next nonce value (thread-safe).

        Returns:
            Strictly increasing nonce
        """
        with self._lock:
            nonce = int(time.time() * 1000)
            if nonce <= self._last_nonce:
                nonce = self._last_nonce + 1
            self._last_nonce = nonce
            return nonce


class AsyncNonceManager:
    """
    Async-safe nonce manager for concurrent async API calls.

    Uses asyncio.Lock for async context.
    """

    def __init__(self):
        """Initialize async nonce manager."""
        self._last_nonce = 0
        self._lock = None  # Created lazily to avoid event loop issues

    async def get_nonce(self) -> int:
        """
        Get next nonce value (async-safe).

        Returns:
            Strictly increasing nonce
        """
        import asyncio

        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            nonce = int(time.time() * 1000)
            if nonce <= self._last_nonce:
                nonce = self._last_nonce + 1
            self._last_nonce = nonce
            return nonce


def load_credentials_from_env() -> KrakenCredentials:
    """
    Load API credentials from environment variables.

    Expects:
        KRAKEN_API_KEY: API key
        KRAKEN_API_SECRET: API secret (base64 encoded)

    Returns:
        KrakenCredentials instance

    Raises:
        ValueError: If environment variables not set
    """
    import os

    api_key = os.environ.get("KRAKEN_API_KEY", "")
    api_secret = os.environ.get("KRAKEN_API_SECRET", "")

    if not api_key or not api_secret:
        raise ValueError(
            "KRAKEN_API_KEY and KRAKEN_API_SECRET environment variables required"
        )

    return KrakenCredentials(api_key=api_key, api_secret=api_secret)


def load_credentials_from_file(path: str) -> KrakenCredentials:
    """
    Load API credentials from a file.

    File format (one per line):
        api_key=YOUR_KEY
        api_secret=YOUR_SECRET

    Or JSON format:
        {"api_key": "...", "api_secret": "..."}

    Args:
        path: Path to credentials file

    Returns:
        KrakenCredentials instance
    """
    import json
    from pathlib import Path

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Credentials file not found: {path}")

    content = file_path.read_text().strip()

    # Try JSON first
    if content.startswith("{"):
        data = json.loads(content)
        return KrakenCredentials(
            api_key=data["api_key"],
            api_secret=data["api_secret"],
        )

    # Parse key=value format
    api_key = ""
    api_secret = ""

    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("api_key="):
            api_key = line.split("=", 1)[1]
        elif line.startswith("api_secret="):
            api_secret = line.split("=", 1)[1]

    return KrakenCredentials(api_key=api_key, api_secret=api_secret)
