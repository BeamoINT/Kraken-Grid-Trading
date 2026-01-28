"""
Tests for Kraken API authentication.

Tests cover:
- HMAC-SHA512 signature generation
- Nonce generation and uniqueness
- Credential validation
"""

import base64
import hashlib
import hmac
import time
import pytest
from unittest.mock import patch

from src.api.auth import (
    KrakenAuth,
    KrakenCredentials,
    NonceManager,
    AsyncNonceManager,
    load_credentials_from_env,
)


# Test credentials (not real - for testing only)
TEST_API_KEY = "test_api_key_12345"
TEST_API_SECRET = base64.b64encode(b"test_secret_bytes_for_testing").decode()


class TestKrakenCredentials:
    """Tests for KrakenCredentials dataclass."""

    def test_valid_credentials(self):
        """Test creating valid credentials."""
        creds = KrakenCredentials(
            api_key=TEST_API_KEY,
            api_secret=TEST_API_SECRET,
        )
        assert creds.api_key == TEST_API_KEY
        assert creds.api_secret == TEST_API_SECRET

    def test_empty_api_key_raises(self):
        """Test that empty API key raises ValueError."""
        with pytest.raises(ValueError, match="API key is required"):
            KrakenCredentials(api_key="", api_secret=TEST_API_SECRET)

    def test_empty_api_secret_raises(self):
        """Test that empty API secret raises ValueError."""
        with pytest.raises(ValueError, match="API secret is required"):
            KrakenCredentials(api_key=TEST_API_KEY, api_secret="")

    def test_invalid_base64_secret_raises(self):
        """Test that invalid base64 secret raises ValueError."""
        with pytest.raises(ValueError, match="base64 encoded"):
            KrakenCredentials(api_key=TEST_API_KEY, api_secret="not-valid-base64!!!")


class TestKrakenAuth:
    """Tests for KrakenAuth class."""

    def test_init(self):
        """Test authentication handler initialization."""
        auth = KrakenAuth(TEST_API_KEY, TEST_API_SECRET)
        assert auth._api_key == TEST_API_KEY
        assert auth._api_secret == TEST_API_SECRET

    def test_from_credentials(self):
        """Test creating auth from credentials object."""
        creds = KrakenCredentials(TEST_API_KEY, TEST_API_SECRET)
        auth = KrakenAuth.from_credentials(creds)
        assert auth._api_key == TEST_API_KEY

    def test_generate_nonce_is_increasing(self):
        """Test that nonces are strictly increasing."""
        auth = KrakenAuth(TEST_API_KEY, TEST_API_SECRET)

        nonces = [auth.generate_nonce() for _ in range(100)]

        # Check all unique
        assert len(set(nonces)) == len(nonces)

        # Check strictly increasing
        for i in range(1, len(nonces)):
            assert nonces[i] > nonces[i - 1]

    def test_generate_nonce_based_on_time(self):
        """Test that nonces are based on millisecond timestamp."""
        auth = KrakenAuth(TEST_API_KEY, TEST_API_SECRET)

        before = int(time.time() * 1000)
        nonce = auth.generate_nonce()
        after = int(time.time() * 1000)

        assert before <= nonce <= after + 1

    def test_sign_request_returns_headers(self):
        """Test that sign_request returns proper headers."""
        auth = KrakenAuth(TEST_API_KEY, TEST_API_SECRET)

        headers = auth.sign_request(
            uri_path="/0/private/Balance",
            data={},
            nonce=1234567890123,
        )

        assert "API-Key" in headers
        assert "API-Sign" in headers
        assert headers["API-Key"] == TEST_API_KEY

    def test_sign_request_signature_format(self):
        """Test that signature is valid base64."""
        auth = KrakenAuth(TEST_API_KEY, TEST_API_SECRET)

        headers = auth.sign_request(
            uri_path="/0/private/Balance",
            data={"asset": "USD"},
            nonce=1234567890123,
        )

        # Should be valid base64
        signature = headers["API-Sign"]
        try:
            decoded = base64.b64decode(signature)
            # HMAC-SHA512 produces 64 bytes
            assert len(decoded) == 64
        except Exception:
            pytest.fail("Signature is not valid base64")

    def test_sign_request_deterministic(self):
        """Test that same inputs produce same signature."""
        auth = KrakenAuth(TEST_API_KEY, TEST_API_SECRET)

        headers1 = auth.sign_request(
            uri_path="/0/private/Balance",
            data={},
            nonce=1234567890123,
        )

        headers2 = auth.sign_request(
            uri_path="/0/private/Balance",
            data={},
            nonce=1234567890123,
        )

        assert headers1["API-Sign"] == headers2["API-Sign"]

    def test_sign_request_different_paths_different_signatures(self):
        """Test that different paths produce different signatures."""
        auth = KrakenAuth(TEST_API_KEY, TEST_API_SECRET)

        headers1 = auth.sign_request(
            uri_path="/0/private/Balance",
            data={},
            nonce=1234567890123,
        )

        headers2 = auth.sign_request(
            uri_path="/0/private/TradeBalance",
            data={},
            nonce=1234567890123,
        )

        assert headers1["API-Sign"] != headers2["API-Sign"]

    def test_sign_request_different_data_different_signatures(self):
        """Test that different data produces different signatures."""
        auth = KrakenAuth(TEST_API_KEY, TEST_API_SECRET)

        headers1 = auth.sign_request(
            uri_path="/0/private/Balance",
            data={"asset": "USD"},
            nonce=1234567890123,
        )

        headers2 = auth.sign_request(
            uri_path="/0/private/Balance",
            data={"asset": "EUR"},
            nonce=1234567890123,
        )

        assert headers1["API-Sign"] != headers2["API-Sign"]

    def test_get_signed_data(self):
        """Test get_signed_data convenience method."""
        auth = KrakenAuth(TEST_API_KEY, TEST_API_SECRET)

        headers, data = auth.get_signed_data(
            uri_path="/0/private/Balance",
            data={"asset": "USD"},
        )

        assert "API-Key" in headers
        assert "API-Sign" in headers
        assert "nonce" in data
        assert data["asset"] == "USD"

    def test_verify_signature_algorithm(self):
        """Test that signature follows Kraken's algorithm exactly."""
        auth = KrakenAuth(TEST_API_KEY, TEST_API_SECRET)

        uri_path = "/0/private/Balance"
        nonce = 1234567890123
        data = {"asset": "USD"}

        headers = auth.sign_request(uri_path, data, nonce)

        # Manually compute expected signature
        post_data = f"nonce={nonce}&asset=USD"
        message = str(nonce) + post_data
        sha256_hash = hashlib.sha256(message.encode("utf-8")).digest()
        hmac_message = uri_path.encode("utf-8") + sha256_hash
        decoded_secret = base64.b64decode(TEST_API_SECRET)
        expected_sig = base64.b64encode(
            hmac.new(decoded_secret, hmac_message, hashlib.sha512).digest()
        ).decode()

        assert headers["API-Sign"] == expected_sig


class TestNonceManager:
    """Tests for thread-safe NonceManager."""

    def test_nonce_is_increasing(self):
        """Test that nonces are strictly increasing."""
        manager = NonceManager()

        nonces = [manager.get_nonce() for _ in range(100)]

        for i in range(1, len(nonces)):
            assert nonces[i] > nonces[i - 1]

    def test_thread_safety(self):
        """Test nonce generation is thread-safe."""
        import threading

        manager = NonceManager()
        nonces = []
        lock = threading.Lock()

        def get_nonces(n: int):
            local_nonces = []
            for _ in range(n):
                local_nonces.append(manager.get_nonce())
            with lock:
                nonces.extend(local_nonces)

        threads = [
            threading.Thread(target=get_nonces, args=(100,))
            for _ in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All nonces should be unique
        assert len(set(nonces)) == len(nonces)


class TestAsyncNonceManager:
    """Tests for async-safe AsyncNonceManager."""

    @pytest.mark.asyncio
    async def test_async_nonce_is_increasing(self):
        """Test that async nonces are strictly increasing."""
        manager = AsyncNonceManager()

        nonces = []
        for _ in range(100):
            nonces.append(await manager.get_nonce())

        for i in range(1, len(nonces)):
            assert nonces[i] > nonces[i - 1]

    @pytest.mark.asyncio
    async def test_async_concurrency(self):
        """Test nonce generation with concurrent async tasks."""
        import asyncio

        manager = AsyncNonceManager()

        async def get_many_nonces(n: int):
            return [await manager.get_nonce() for _ in range(n)]

        # Run multiple concurrent tasks
        results = await asyncio.gather(*[
            get_many_nonces(50) for _ in range(10)
        ])

        # Flatten results
        all_nonces = [n for result in results for n in result]

        # All should be unique
        assert len(set(all_nonces)) == len(all_nonces)


class TestLoadCredentialsFromEnv:
    """Tests for environment variable credential loading."""

    def test_load_from_env(self):
        """Test loading credentials from environment."""
        with patch.dict('os.environ', {
            'KRAKEN_API_KEY': TEST_API_KEY,
            'KRAKEN_API_SECRET': TEST_API_SECRET,
        }):
            creds = load_credentials_from_env()
            assert creds.api_key == TEST_API_KEY
            assert creds.api_secret == TEST_API_SECRET

    def test_missing_env_vars_raises(self):
        """Test that missing env vars raises ValueError."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="environment variables required"):
                load_credentials_from_env()

    def test_partial_env_vars_raises(self):
        """Test that partial env vars raises ValueError."""
        with patch.dict('os.environ', {'KRAKEN_API_KEY': TEST_API_KEY}, clear=True):
            with pytest.raises(ValueError, match="environment variables required"):
                load_credentials_from_env()
