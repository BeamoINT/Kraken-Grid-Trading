"""
Grid Trading Module.

Provides adaptive grid trading with regime-based behavior:
- GridCalculator: Compute grid levels and volumes
- GridStrategy: Adapt grid to market regimes
- GridExecutor: Execute and manage grid orders
- Rebalancer: Detect drift and trigger rebalancing

Usage:
    from src.grid import (
        GridCalculator,
        GridStrategy,
        GridExecutor,
        Rebalancer,
    )

    # Setup
    calculator = GridCalculator(grid_config, risk_config)
    strategy = GridStrategy(grid_config, risk_config, calculator)
    executor = GridExecutor(order_manager, strategy, risk_config)
    rebalancer = Rebalancer()

    # Get regime prediction from ML model
    regime, confidence = MarketRegime.RANGING, 0.85

    # Compute adapted grid
    grid_state, adaptation = strategy.compute_adapted_grid(
        current_price=Decimal("50000"),
        atr=Decimal("500"),
        regime=regime,
        confidence=confidence,
    )

    # Deploy grid
    result = executor.deploy_grid(grid_state, adaptation, regime, confidence)

    # Monitor for rebalancing
    metrics = rebalancer.compute_drift_metrics(
        grid_state, current_price, position, order_manager
    )
    decision = rebalancer.should_rebalance(metrics, regime, previous_regime)
    if decision.should_rebalance:
        # Trigger grid recalculation
        new_center = rebalancer.get_suggested_new_center(...)
"""

from .grid_calculator import (
    GridLevel,
    GridParameters,
    GridState,
    GridCalculator,
)
from .grid_strategy import (
    RegimeAdaptation,
    GridStrategy,
)
from .grid_executor import (
    ExecutionResult,
    GridSnapshot,
    GridExecutor,
)
from .rebalancer import (
    DriftMetrics,
    RebalanceReason,
    RebalanceDecision,
    Rebalancer,
)

__all__ = [
    # Calculator
    "GridLevel",
    "GridParameters",
    "GridState",
    "GridCalculator",
    # Strategy
    "RegimeAdaptation",
    "GridStrategy",
    # Executor
    "ExecutionResult",
    "GridSnapshot",
    "GridExecutor",
    # Rebalancer
    "DriftMetrics",
    "RebalanceReason",
    "RebalanceDecision",
    "Rebalancer",
]
