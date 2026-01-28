"""
Tests for State Manager module.

Tests:
- State save and load
- Order history tracking
- Session management
- Database operations
"""

import pytest
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from src.core import StateManager, BotState


class TestBotState:
    """Tests for BotState dataclass."""

    def test_bot_state_creation(self):
        """Test creating a BotState."""
        state = BotState(
            timestamp=datetime.utcnow(),
            version="1.0",
            last_regime="RANGING",
            last_confidence=0.75,
            is_trading=True,
        )

        assert state.version == "1.0"
        assert state.last_regime == "RANGING"
        assert state.last_confidence == 0.75
        assert state.is_trading is True
        assert state.is_paused is False

    def test_bot_state_to_dict(self):
        """Test serialization to dictionary."""
        ts = datetime(2025, 1, 15, 12, 0, 0)
        state = BotState(
            timestamp=ts,
            last_regime="TRENDING_UP",
            last_confidence=0.85,
            is_trading=True,
            grid_snapshot={"levels": [1, 2, 3]},
        )

        data = state.to_dict()

        assert data["timestamp"] == ts.isoformat()
        assert data["last_regime"] == "TRENDING_UP"
        assert data["last_confidence"] == 0.85
        assert data["grid_snapshot"] == {"levels": [1, 2, 3]}

    def test_bot_state_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "timestamp": "2025-01-15T12:00:00",
            "version": "1.0",
            "last_regime": "BREAKOUT",
            "last_confidence": 0.65,
            "is_trading": False,
            "is_paused": True,
            "pause_reason": "Low confidence",
        }

        state = BotState.from_dict(data)

        assert state.timestamp == datetime(2025, 1, 15, 12, 0, 0)
        assert state.last_regime == "BREAKOUT"
        assert state.last_confidence == 0.65
        assert state.is_paused is True
        assert state.pause_reason == "Low confidence"


class TestStateManager:
    """Tests for StateManager class."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name

    @pytest.fixture
    def state_manager(self, temp_db):
        """Create state manager with temporary database."""
        return StateManager(db_path=temp_db)

    def test_initialization(self, state_manager):
        """Test state manager initialization."""
        assert state_manager.has_state() is False

    def test_save_and_load_state(self, state_manager):
        """Test saving and loading state."""
        state = BotState(
            timestamp=datetime.utcnow(),
            last_regime="RANGING",
            last_confidence=0.80,
            is_trading=True,
        )

        state_manager.save_state(state)
        assert state_manager.has_state() is True

        loaded = state_manager.load_state()
        assert loaded is not None
        assert loaded.last_regime == "RANGING"
        assert loaded.last_confidence == 0.80
        assert loaded.is_trading is True

    def test_clear_state(self, state_manager):
        """Test clearing state."""
        state = BotState(timestamp=datetime.utcnow())
        state_manager.save_state(state)

        assert state_manager.has_state() is True

        state_manager.clear_state()
        assert state_manager.has_state() is False

    def test_state_age(self, state_manager):
        """Test state age calculation."""
        # No state
        assert state_manager.get_state_age_seconds() is None

        # Save state
        state = BotState(timestamp=datetime.utcnow())
        state_manager.save_state(state)

        age = state_manager.get_state_age_seconds()
        assert age is not None
        assert age >= 0
        assert age < 5  # Should be very recent

    def test_partial_state_updates(self, state_manager):
        """Test partial state updates."""
        # Save initial state
        state = BotState(
            timestamp=datetime.utcnow(),
            last_regime="RANGING",
        )
        state_manager.save_state(state)

        # Update grid state only
        state_manager.save_grid_state({"center": 50000, "levels": 10})

        loaded = state_manager.load_state()
        assert loaded.grid_snapshot == {"center": 50000, "levels": 10}
        assert loaded.last_regime == "RANGING"  # Preserved

    def test_save_and_get_order(self, state_manager):
        """Test order history tracking."""
        state_manager.save_order(
            grid_id="grid_001",
            exchange_id="EXCH123",
            level=5,
            side="BUY",
            price=Decimal("50000"),
            volume=Decimal("0.001"),
            state="OPEN",
        )

        orders = state_manager.get_order_history()
        assert len(orders) == 1
        assert orders[0]["grid_id"] == "grid_001"
        assert orders[0]["side"] == "BUY"

    def test_update_order_state(self, state_manager):
        """Test updating order state."""
        state_manager.save_order(
            grid_id="grid_002",
            exchange_id=None,
            level=3,
            side="SELL",
            price=Decimal("51000"),
            volume=Decimal("0.001"),
            state="PENDING_SUBMIT",
        )

        state_manager.update_order_state(
            grid_id="grid_002",
            state="OPEN",
            exchange_id="EXCH456",
        )

        orders = state_manager.get_order_history()
        assert orders[0]["state"] == "OPEN"
        assert orders[0]["exchange_id"] == "EXCH456"

    def test_save_and_get_fills(self, state_manager):
        """Test fill history tracking."""
        state_manager.save_order(
            grid_id="grid_003",
            exchange_id="EXCH789",
            level=4,
            side="BUY",
            price=Decimal("49000"),
            volume=Decimal("0.001"),
            state="OPEN",
        )

        state_manager.save_fill(
            order_id="grid_003",
            fill_volume=Decimal("0.0005"),
            fill_price=Decimal("49000"),
            fee=Decimal("0.05"),
            timestamp=datetime.utcnow(),
        )

        fills = state_manager.get_fill_history()
        assert len(fills) == 1
        assert fills[0]["order_id"] == "grid_003"
        assert fills[0]["fill_volume"] == "0.0005"

    def test_order_history_filtering(self, state_manager):
        """Test order history with filters."""
        # Save multiple orders
        for i in range(5):
            state = "FILLED" if i < 3 else "OPEN"
            state_manager.save_order(
                grid_id=f"grid_{i:03d}",
                exchange_id=f"EXCH{i}",
                level=i,
                side="BUY" if i % 2 == 0 else "SELL",
                price=Decimal("50000"),
                volume=Decimal("0.001"),
                state=state,
            )

        # Filter by state
        open_orders = state_manager.get_order_history(state="OPEN")
        assert len(open_orders) == 2

        filled_orders = state_manager.get_order_history(state="FILLED")
        assert len(filled_orders) == 3

    def test_active_orders(self, state_manager):
        """Test getting active orders."""
        state_manager.save_order(
            grid_id="active_001",
            exchange_id="E1",
            level=1,
            side="BUY",
            price=Decimal("50000"),
            volume=Decimal("0.001"),
            state="OPEN",
        )
        state_manager.save_order(
            grid_id="active_002",
            exchange_id="E2",
            level=2,
            side="SELL",
            price=Decimal("51000"),
            volume=Decimal("0.001"),
            state="FILLED",
        )

        active = state_manager.get_active_orders()
        assert len(active) == 1
        assert active[0]["grid_id"] == "active_001"


class TestSessionManagement:
    """Tests for session management."""

    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name

    @pytest.fixture
    def state_manager(self, temp_db):
        return StateManager(db_path=temp_db)

    def test_start_session(self, state_manager):
        """Test starting a session."""
        session_id = state_manager.start_session(
            initial_capital=Decimal("400")
        )

        assert session_id is not None
        assert session_id.startswith("session_")

        session = state_manager.get_session(session_id)
        assert session is not None
        assert session["status"] == "active"

    def test_end_session(self, state_manager):
        """Test ending a session."""
        session_id = state_manager.start_session()

        stats = {
            "total_trades": 15,
            "total_pnl": "25.50",
            "max_drawdown_pct": 5.5,
            "final_capital": "425.50",
        }
        state_manager.end_session(session_id, stats)

        session = state_manager.get_session(session_id)
        assert session["status"] == "completed"
        assert session["total_trades"] == 15
        assert session["total_pnl"] == "25.50"

    def test_get_active_session(self, state_manager):
        """Test getting active session."""
        # No active session
        assert state_manager.get_active_session() is None

        # Start session
        session_id = state_manager.start_session()
        active = state_manager.get_active_session()

        assert active is not None
        assert active["id"] == session_id

    def test_session_history(self, state_manager):
        """Test session history."""
        # Create multiple sessions
        for i in range(3):
            session_id = state_manager.start_session()
            state_manager.end_session(session_id, {"total_trades": i})

        history = state_manager.get_session_history(limit=10)
        assert len(history) == 3


class TestMetrics:
    """Tests for metrics tracking."""

    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name

    @pytest.fixture
    def state_manager(self, temp_db):
        return StateManager(db_path=temp_db)

    def test_save_metrics_snapshot(self, state_manager):
        """Test saving metrics snapshot."""
        session_id = state_manager.start_session()

        state_manager.save_metrics_snapshot(
            session_id=session_id,
            equity=Decimal("410"),
            drawdown_pct=2.5,
            regime="TRENDING_UP",
            confidence=0.85,
            active_orders=8,
            position_qty=Decimal("0.005"),
        )

        metrics = state_manager.get_metrics_history(session_id)
        assert len(metrics) == 1
        assert metrics[0]["regime"] == "TRENDING_UP"
        assert metrics[0]["confidence"] == 0.85

    def test_multiple_metrics(self, state_manager):
        """Test multiple metrics snapshots."""
        session_id = state_manager.start_session()

        for i in range(5):
            state_manager.save_metrics_snapshot(
                session_id=session_id,
                equity=Decimal(f"{400 + i * 10}"),
                drawdown_pct=i * 0.5,
                regime="RANGING",
                confidence=0.75,
                active_orders=10 - i,
                position_qty=Decimal("0.001"),
            )

        metrics = state_manager.get_metrics_history(session_id, limit=3)
        assert len(metrics) == 3


class TestDatabaseOperations:
    """Tests for database operations."""

    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name

    @pytest.fixture
    def state_manager(self, temp_db):
        return StateManager(db_path=temp_db)

    def test_get_stats(self, state_manager):
        """Test getting database stats."""
        stats = state_manager.get_stats()

        assert "total_orders" in stats
        assert "total_fills" in stats
        assert "total_sessions" in stats
        assert "has_state" in stats
        assert stats["has_state"] is False

    def test_vacuum(self, state_manager):
        """Test database vacuum."""
        # Should not raise
        state_manager.vacuum()

    def test_cleanup_old_data(self, state_manager):
        """Test cleaning up old data."""
        # Add some data
        state_manager.save_order(
            grid_id="old_001",
            exchange_id=None,
            level=1,
            side="BUY",
            price=Decimal("50000"),
            volume=Decimal("0.001"),
            state="FILLED",
        )

        # Cleanup (won't delete recent data)
        deleted = state_manager.cleanup_old_data(days=30)

        assert "orders" in deleted
        assert "fills" in deleted
