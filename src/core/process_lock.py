"""
Process Lock for Single Instance Enforcement.

Provides PID-based locking to prevent multiple bot instances from running
simultaneously, which could cause duplicate orders and state corruption.

Usage:
    from src.core.process_lock import ProcessLock

    lock = ProcessLock()
    if not lock.acquire():
        print("Another instance is running")
        sys.exit(1)

    try:
        # Run bot...
    finally:
        lock.release()

    # Or use as context manager:
    with ProcessLock() as lock:
        # Run bot...
"""

import json
import logging
import os
import signal
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class LockInfo:
    """Information stored in the lock file."""

    pid: int
    started_at: datetime
    hostname: str

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "pid": self.pid,
            "started_at": self.started_at.isoformat(),
            "hostname": self.hostname,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LockInfo":
        """Deserialize from dictionary."""
        return cls(
            pid=data["pid"],
            started_at=datetime.fromisoformat(data["started_at"]),
            hostname=data.get("hostname", "unknown"),
        )


class ProcessLockError(Exception):
    """Exception raised when lock cannot be acquired."""

    def __init__(self, message: str, pid: Optional[int] = None, started_at: Optional[datetime] = None):
        super().__init__(message)
        self.pid = pid
        self.started_at = started_at


class ProcessLock:
    """
    PID-based process lock with stale detection.

    Creates a lock file containing the PID and start time of the owning process.
    Detects stale locks (process no longer running) and allows override.

    Attributes:
        lock_path: Path to the lock file
    """

    def __init__(self, lock_path: str = "data/trading.pid"):
        """
        Initialize process lock.

        Args:
            lock_path: Path to the lock file
        """
        self._lock_path = Path(lock_path)
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._acquired = False

    @property
    def lock_path(self) -> Path:
        """Get the lock file path."""
        return self._lock_path

    def is_process_alive(self, pid: int) -> bool:
        """
        Check if a process with the given PID is still running.

        Args:
            pid: Process ID to check

        Returns:
            True if process is running, False otherwise
        """
        try:
            # Signal 0 doesn't kill the process, just checks if it exists
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def is_locked(self) -> Tuple[bool, Optional[LockInfo]]:
        """
        Check if the lock is currently held.

        Returns:
            Tuple of (is_locked, lock_info)
            - is_locked: True if lock is held by a running process
            - lock_info: LockInfo if lock file exists, None otherwise
        """
        if not self._lock_path.exists():
            return False, None

        try:
            with open(self._lock_path, "r") as f:
                data = json.load(f)
                lock_info = LockInfo.from_dict(data)

            # Check if the owning process is still alive
            if self.is_process_alive(lock_info.pid):
                return True, lock_info
            else:
                # Process is dead - lock is stale
                logger.info(f"Found stale lock from dead process {lock_info.pid}")
                return False, lock_info

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Corrupted lock file
            logger.warning(f"Corrupted lock file: {e}")
            return False, None

    def acquire(self, force: bool = False) -> bool:
        """
        Attempt to acquire the lock.

        Args:
            force: If True, override existing lock (use with caution)

        Returns:
            True if lock was acquired, False otherwise

        Raises:
            ProcessLockError: If lock is held by another running process and force=False
        """
        locked, lock_info = self.is_locked()

        if locked and not force:
            raise ProcessLockError(
                f"Another instance is running (PID {lock_info.pid}, started {lock_info.started_at})",
                pid=lock_info.pid,
                started_at=lock_info.started_at,
            )

        if locked and force:
            logger.warning(
                f"Force acquiring lock from PID {lock_info.pid} "
                f"(started {lock_info.started_at})"
            )

        # Create new lock
        current_lock = LockInfo(
            pid=os.getpid(),
            started_at=datetime.utcnow(),
            hostname=os.uname().nodename,
        )

        try:
            with open(self._lock_path, "w") as f:
                json.dump(current_lock.to_dict(), f, indent=2)
            self._acquired = True
            logger.info(f"Acquired process lock (PID {current_lock.pid})")
            return True
        except IOError as e:
            logger.error(f"Failed to create lock file: {e}")
            return False

    def release(self) -> bool:
        """
        Release the lock.

        Returns:
            True if lock was released, False if not held
        """
        if not self._acquired:
            return False

        try:
            # Verify we still own the lock before deleting
            if self._lock_path.exists():
                with open(self._lock_path, "r") as f:
                    data = json.load(f)
                    if data.get("pid") == os.getpid():
                        self._lock_path.unlink()
                        logger.info("Released process lock")
                    else:
                        logger.warning("Lock was taken by another process, not releasing")
            self._acquired = False
            return True
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error releasing lock: {e}")
            self._acquired = False
            return False

    def __enter__(self) -> "ProcessLock":
        """Context manager entry."""
        if not self.acquire():
            raise ProcessLockError("Failed to acquire lock")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.release()

    def get_lock_info(self) -> Optional[LockInfo]:
        """
        Get information about the current lock holder.

        Returns:
            LockInfo if lock file exists and is valid, None otherwise
        """
        _, lock_info = self.is_locked()
        return lock_info
