"""
State Manager for Bot Persistence.

Provides:
- State persistence for crash recovery
- Order and fill history tracking
- Session management
- SQLite-based atomic operations
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal objects."""

    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


@dataclass
class BotState:
    """Complete bot state for persistence."""

    timestamp: datetime
    version: str = "1.0"

    # Grid state (from GridExecutor.get_snapshot())
    grid_snapshot: Optional[Dict[str, Any]] = None

    # Risk state (from RiskManager.to_dict())
    risk_state: Optional[Dict[str, Any]] = None

    # Position (from GridPosition)
    position: Optional[Dict[str, Any]] = None

    # ML state
    last_regime: Optional[str] = None
    last_confidence: float = 0.0

    # Trading state
    is_trading: bool = False
    is_paused: bool = False
    pause_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "grid_snapshot": self.grid_snapshot,
            "risk_state": self.risk_state,
            "position": self.position,
            "last_regime": self.last_regime,
            "last_confidence": self.last_confidence,
            "is_trading": self.is_trading,
            "is_paused": self.is_paused,
            "pause_reason": self.pause_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BotState":
        """Deserialize from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            version=data.get("version", "1.0"),
            grid_snapshot=data.get("grid_snapshot"),
            risk_state=data.get("risk_state"),
            position=data.get("position"),
            last_regime=data.get("last_regime"),
            last_confidence=data.get("last_confidence", 0.0),
            is_trading=data.get("is_trading", False),
            is_paused=data.get("is_paused", False),
            pause_reason=data.get("pause_reason", ""),
        )


class StateManager:
    """
    Manages persistent state for crash recovery.

    Uses SQLite for atomic operations and durability.

    Usage:
        state_mgr = StateManager("data/trading.db")

        # Save state
        state = BotState(
            timestamp=datetime.utcnow(),
            grid_snapshot=executor.get_snapshot(),
            risk_state=risk_manager.to_dict(),
        )
        state_mgr.save_state(state)

        # Load on restart
        if state_mgr.has_state():
            state = state_mgr.load_state()
            # Restore components from state
    """

    SCHEMA = """
    -- Bot state (single row, updated atomically)
    CREATE TABLE IF NOT EXISTS bot_state (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        timestamp TEXT NOT NULL,
        version TEXT NOT NULL,
        state_json TEXT NOT NULL,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    -- Order history
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        grid_id TEXT UNIQUE NOT NULL,
        exchange_id TEXT,
        level INTEGER,
        side TEXT,
        price TEXT,
        volume TEXT,
        state TEXT,
        created_at TEXT,
        updated_at TEXT
    );

    -- Fill history
    CREATE TABLE IF NOT EXISTS fills (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id TEXT NOT NULL,
        fill_volume TEXT,
        fill_price TEXT,
        fee TEXT,
        timestamp TEXT
    );

    -- Session tracking
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        initial_capital TEXT,
        final_capital TEXT,
        total_trades INTEGER DEFAULT 0,
        total_pnl TEXT,
        max_drawdown_pct REAL,
        status TEXT DEFAULT 'active'
    );

    -- Metrics snapshots (for analysis)
    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        timestamp TEXT,
        equity TEXT,
        drawdown_pct REAL,
        regime TEXT,
        confidence REAL,
        active_orders INTEGER,
        position_qty TEXT
    );

    -- Pending operations (Write-Ahead Log for crash recovery)
    CREATE TABLE IF NOT EXISTS pending_operations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        operation_type TEXT NOT NULL,
        operation_data TEXT NOT NULL,
        grid_id TEXT,
        created_at TEXT NOT NULL,
        completed_at TEXT,
        status TEXT DEFAULT 'pending'
    );

    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_orders_grid_id ON orders(grid_id);
    CREATE INDEX IF NOT EXISTS idx_orders_state ON orders(state);
    CREATE INDEX IF NOT EXISTS idx_fills_order_id ON fills(order_id);
    CREATE INDEX IF NOT EXISTS idx_fills_timestamp ON fills(timestamp);
    CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
    CREATE INDEX IF NOT EXISTS idx_metrics_session ON metrics(session_id);
    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
    CREATE INDEX IF NOT EXISTS idx_pending_ops_status ON pending_operations(status);
    CREATE INDEX IF NOT EXISTS idx_pending_ops_grid_id ON pending_operations(grid_id);
    """

    def __init__(self, db_path: str = "data/trading.db"):
        """
        Initialize state manager.

        Args:
            db_path: Path to SQLite database file
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()
        logger.info(f"StateManager initialized with database: {self._db_path}")

    def _init_database(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self._db_path) as conn:
            conn.executescript(self.SCHEMA)
            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # === Core State Operations ===

    def save_state(self, state: BotState) -> None:
        """
        Save complete bot state atomically.

        Uses INSERT OR REPLACE to ensure single row.

        Args:
            state: BotState to persist
        """
        state_json = json.dumps(state.to_dict(), cls=DecimalEncoder)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO bot_state (id, timestamp, version, state_json, updated_at)
                VALUES (1, ?, ?, ?, ?)
                """,
                (
                    state.timestamp.isoformat(),
                    state.version,
                    state_json,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

        logger.debug(f"State saved at {state.timestamp}")

    def load_state(self) -> Optional[BotState]:
        """
        Load most recent bot state.

        Returns:
            BotState if exists, None otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT state_json FROM bot_state WHERE id = 1"
            )
            row = cursor.fetchone()

        if row is None:
            return None

        data = json.loads(row["state_json"])
        state = BotState.from_dict(data)
        logger.info(f"Loaded state from {state.timestamp}")
        return state

    def has_state(self) -> bool:
        """Check if saved state exists."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM bot_state WHERE id = 1"
            )
            row = cursor.fetchone()
            return row["count"] > 0

    def clear_state(self) -> None:
        """Clear saved state (fresh start)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM bot_state WHERE id = 1")
            conn.commit()
        logger.info("Bot state cleared")

    def get_state_age_seconds(self) -> Optional[float]:
        """
        Get age of saved state in seconds.

        Returns:
            Age in seconds, or None if no state exists
        """
        state = self.load_state()
        if state is None:
            return None

        age = (datetime.utcnow() - state.timestamp).total_seconds()
        return age

    # === Partial State Operations ===

    def save_grid_state(self, snapshot: Dict[str, Any]) -> None:
        """
        Save just grid state (for frequent updates).

        Updates only the grid_snapshot field in existing state.

        Args:
            snapshot: Grid snapshot dictionary
        """
        state = self.load_state()
        if state is None:
            state = BotState(timestamp=datetime.utcnow())

        state.grid_snapshot = snapshot
        state.timestamp = datetime.utcnow()
        self.save_state(state)

    def save_risk_state(self, risk_dict: Dict[str, Any]) -> None:
        """
        Save just risk state.

        Args:
            risk_dict: Risk manager state dictionary
        """
        state = self.load_state()
        if state is None:
            state = BotState(timestamp=datetime.utcnow())

        state.risk_state = risk_dict
        state.timestamp = datetime.utcnow()
        self.save_state(state)

    def save_position(self, position_dict: Dict[str, Any]) -> None:
        """
        Save just position state.

        Args:
            position_dict: Position state dictionary
        """
        state = self.load_state()
        if state is None:
            state = BotState(timestamp=datetime.utcnow())

        state.position = position_dict
        state.timestamp = datetime.utcnow()
        self.save_state(state)

    # === Order History ===

    def save_order(
        self,
        grid_id: str,
        exchange_id: Optional[str],
        level: int,
        side: str,
        price: Decimal,
        volume: Decimal,
        state: str,
    ) -> None:
        """
        Save order to history table.

        Args:
            grid_id: Internal grid order ID
            exchange_id: Exchange order ID (may be None initially)
            level: Grid level index
            side: Order side (BUY/SELL)
            price: Order price
            volume: Order volume
            state: Order state (PENDING_SUBMIT, OPEN, etc.)
        """
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO orders
                (grid_id, exchange_id, level, side, price, volume, state, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE(
                    (SELECT created_at FROM orders WHERE grid_id = ?), ?
                ), ?)
                """,
                (
                    grid_id,
                    exchange_id,
                    level,
                    side,
                    str(price),
                    str(volume),
                    state,
                    grid_id,
                    now,
                    now,
                ),
            )
            conn.commit()

    def update_order_state(
        self,
        grid_id: str,
        state: str,
        exchange_id: Optional[str] = None,
    ) -> None:
        """
        Update order state.

        Args:
            grid_id: Internal grid order ID
            state: New state
            exchange_id: Exchange order ID (if now known)
        """
        now = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            if exchange_id:
                conn.execute(
                    """
                    UPDATE orders SET state = ?, exchange_id = ?, updated_at = ?
                    WHERE grid_id = ?
                    """,
                    (state, exchange_id, now, grid_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE orders SET state = ?, updated_at = ?
                    WHERE grid_id = ?
                    """,
                    (state, now, grid_id),
                )
            conn.commit()

    def save_fill(
        self,
        order_id: str,
        fill_volume: Decimal,
        fill_price: Decimal,
        fee: Decimal,
        timestamp: datetime,
    ) -> None:
        """
        Save fill to history table.

        Args:
            order_id: Grid order ID
            fill_volume: Filled volume
            fill_price: Fill price
            fee: Trading fee
            timestamp: Fill timestamp
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO fills (order_id, fill_volume, fill_price, fee, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    order_id,
                    str(fill_volume),
                    str(fill_price),
                    str(fee),
                    timestamp.isoformat(),
                ),
            )
            conn.commit()

    def get_order_history(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        state: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query order history.

        Args:
            start: Start timestamp filter
            end: End timestamp filter
            state: Order state filter

        Returns:
            List of order dictionaries
        """
        query = "SELECT * FROM orders WHERE 1=1"
        params: List[Any] = []

        if start:
            query += " AND created_at >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND created_at <= ?"
            params.append(end.isoformat())
        if state:
            query += " AND state = ?"
            params.append(state)

        query += " ORDER BY created_at DESC"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_fill_history(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        order_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query fill history.

        Args:
            start: Start timestamp filter
            end: End timestamp filter
            order_id: Filter by specific order

        Returns:
            List of fill dictionaries
        """
        query = "SELECT * FROM fills WHERE 1=1"
        params: List[Any] = []

        if start:
            query += " AND timestamp >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND timestamp <= ?"
            params.append(end.isoformat())
        if order_id:
            query += " AND order_id = ?"
            params.append(order_id)

        query += " ORDER BY timestamp DESC"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_active_orders(self) -> List[Dict[str, Any]]:
        """Get all orders in active states."""
        active_states = (
            "PENDING_SUBMIT",
            "SUBMITTED",
            "OPEN",
            "PARTIALLY_FILLED",
        )
        placeholders = ",".join("?" * len(active_states))

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM orders WHERE state IN ({placeholders})",
                active_states,
            )
            rows = cursor.fetchall()

        return [dict(row) for row in rows]

    # === Session Tracking ===

    def start_session(self, initial_capital: Optional[Decimal] = None) -> str:
        """
        Start new trading session.

        Args:
            initial_capital: Starting capital for session

        Returns:
            Session ID
        """
        session_id = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, started_at, initial_capital, status)
                VALUES (?, ?, ?, 'active')
                """,
                (
                    session_id,
                    datetime.utcnow().isoformat(),
                    str(initial_capital) if initial_capital else None,
                ),
            )
            conn.commit()

        logger.info(f"Started session: {session_id}")
        return session_id

    def end_session(self, session_id: str, stats: Dict[str, Any]) -> None:
        """
        End trading session with final stats.

        Args:
            session_id: Session to end
            stats: Final statistics (total_trades, total_pnl, max_drawdown_pct, final_capital)
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE sessions SET
                    ended_at = ?,
                    final_capital = ?,
                    total_trades = ?,
                    total_pnl = ?,
                    max_drawdown_pct = ?,
                    status = 'completed'
                WHERE id = ?
                """,
                (
                    datetime.utcnow().isoformat(),
                    stats.get("final_capital"),
                    stats.get("total_trades", 0),
                    stats.get("total_pnl"),
                    stats.get("max_drawdown_pct"),
                    session_id,
                ),
            )
            conn.commit()

        logger.info(f"Ended session: {session_id}")

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,),
            )
            row = cursor.fetchone()

        return dict(row) if row else None

    def get_active_session(self) -> Optional[Dict[str, Any]]:
        """Get currently active session if any."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM sessions WHERE status = 'active' ORDER BY started_at DESC LIMIT 1"
            )
            row = cursor.fetchone()

        return dict(row) if row else None

    def get_session_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent sessions."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()

        return [dict(row) for row in rows]

    # === Metrics Snapshots ===

    def save_metrics_snapshot(
        self,
        session_id: str,
        equity: Decimal,
        drawdown_pct: float,
        regime: str,
        confidence: float,
        active_orders: int,
        position_qty: Decimal,
    ) -> None:
        """
        Save metrics snapshot for analysis.

        Args:
            session_id: Current session
            equity: Current equity
            drawdown_pct: Current drawdown percentage
            regime: Current market regime
            confidence: ML model confidence
            active_orders: Number of active orders
            position_qty: Current position quantity
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO metrics
                (session_id, timestamp, equity, drawdown_pct, regime, confidence, active_orders, position_qty)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    datetime.utcnow().isoformat(),
                    str(equity),
                    drawdown_pct,
                    regime,
                    confidence,
                    active_orders,
                    str(position_qty),
                ),
            )
            conn.commit()

    def get_metrics_history(
        self,
        session_id: str,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Get metrics history for session."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM metrics
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (session_id, limit),
            )
            rows = cursor.fetchall()

        return [dict(row) for row in rows]

    # === Utilities ===

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._get_connection() as conn:
            order_count = conn.execute(
                "SELECT COUNT(*) as count FROM orders"
            ).fetchone()["count"]
            fill_count = conn.execute(
                "SELECT COUNT(*) as count FROM fills"
            ).fetchone()["count"]
            session_count = conn.execute(
                "SELECT COUNT(*) as count FROM sessions"
            ).fetchone()["count"]
            active_orders = len(self.get_active_orders())

        return {
            "total_orders": order_count,
            "total_fills": fill_count,
            "total_sessions": session_count,
            "active_orders": active_orders,
            "has_state": self.has_state(),
            "db_path": str(self._db_path),
        }

    def vacuum(self) -> None:
        """Optimize database (reclaim space)."""
        with self._get_connection() as conn:
            conn.execute("VACUUM")
        logger.info("Database vacuumed")

    def cleanup_old_data(self, days: int = 30) -> Dict[str, int]:
        """
        Clean up old data older than specified days.

        Args:
            days: Delete data older than this many days

        Returns:
            Dictionary with counts of deleted rows
        """
        cutoff = datetime.utcnow().isoformat()
        # Calculate cutoff (simple approach - in production use proper date math)
        from datetime import timedelta
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        deleted = {"orders": 0, "fills": 0, "metrics": 0}

        with self._get_connection() as conn:
            # Only delete completed/failed orders
            cursor = conn.execute(
                """
                DELETE FROM orders
                WHERE created_at < ? AND state IN ('FILLED', 'CANCELED', 'FAILED')
                """,
                (cutoff,),
            )
            deleted["orders"] = cursor.rowcount

            cursor = conn.execute(
                "DELETE FROM fills WHERE timestamp < ?",
                (cutoff,),
            )
            deleted["fills"] = cursor.rowcount

            cursor = conn.execute(
                "DELETE FROM metrics WHERE timestamp < ?",
                (cutoff,),
            )
            deleted["metrics"] = cursor.rowcount

            conn.commit()

        logger.info(f"Cleaned up old data: {deleted}")
        return deleted

    # === Write-Ahead Log (WAL) Operations ===

    def log_pending_operation(
        self,
        operation_type: str,
        operation_data: Dict[str, Any],
        grid_id: Optional[str] = None,
    ) -> int:
        """
        Log a pending operation BEFORE executing it.

        This implements write-ahead logging for crash recovery.
        Log the operation first, execute it, then mark as complete.

        Args:
            operation_type: Type of operation (SUBMIT_ORDER, CANCEL_ORDER, etc.)
            operation_data: Operation details as dictionary
            grid_id: Optional grid order ID for reference

        Returns:
            Operation ID for later completion
        """
        now = datetime.utcnow().isoformat()
        data_json = json.dumps(operation_data, cls=DecimalEncoder)

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO pending_operations
                (operation_type, operation_data, grid_id, created_at, status)
                VALUES (?, ?, ?, ?, 'pending')
                """,
                (operation_type, data_json, grid_id, now),
            )
            conn.commit()
            operation_id = cursor.lastrowid

        logger.debug(f"Logged pending operation {operation_id}: {operation_type}")
        return operation_id

    def complete_operation(
        self,
        operation_id: int,
        success: bool = True,
    ) -> None:
        """
        Mark an operation as completed AFTER successful execution.

        Args:
            operation_id: ID from log_pending_operation
            success: Whether operation succeeded
        """
        now = datetime.utcnow().isoformat()
        status = "completed" if success else "failed"

        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE pending_operations
                SET status = ?, completed_at = ?
                WHERE id = ?
                """,
                (status, now, operation_id),
            )
            conn.commit()

        logger.debug(f"Completed operation {operation_id}: {status}")

    def get_pending_operations(self) -> List[Dict[str, Any]]:
        """
        Get all incomplete operations for recovery.

        Returns:
            List of pending operation dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, operation_type, operation_data, grid_id, created_at
                FROM pending_operations
                WHERE status = 'pending'
                ORDER BY created_at ASC
                """
            )
            rows = cursor.fetchall()

        operations = []
        for row in rows:
            op = dict(row)
            # Parse JSON data
            try:
                op["operation_data"] = json.loads(op["operation_data"])
            except json.JSONDecodeError:
                op["operation_data"] = {}
            operations.append(op)

        if operations:
            logger.info(f"Found {len(operations)} pending operations for recovery")

        return operations

    def cleanup_completed_operations(self, hours: int = 24) -> int:
        """
        Clean up old completed operations.

        Args:
            hours: Delete completed operations older than this many hours

        Returns:
            Number of operations deleted
        """
        from datetime import timedelta
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM pending_operations
                WHERE status IN ('completed', 'failed')
                AND completed_at < ?
                """,
                (cutoff,),
            )
            deleted = cursor.rowcount
            conn.commit()

        if deleted:
            logger.debug(f"Cleaned up {deleted} completed operations")

        return deleted
