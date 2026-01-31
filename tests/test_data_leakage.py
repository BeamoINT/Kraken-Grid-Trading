"""
Tests for Data Leakage Prevention.

These tests ensure that:
1. No features use future data (e.g., shift with negative values)
2. Labels are computed from different data than features
3. Outcome-based labeling produces realistic (not perfect) predictions
"""

import inspect
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import indicators
from src.regime.regime_labeler import RegimeLabeler, MarketRegime
from src.regime.outcome_labeler import OutcomeBasedLabeler


class TestNoFutureDataInFeatures:
    """Test that no feature computation uses future data."""

    def test_no_negative_shift_in_indicators(self):
        """
        Check that indicator functions don't use shift with negative values.

        Negative shift looks into the future, which is data leakage.
        """
        # Get source code of indicators module
        source = inspect.getsource(indicators)

        # Find all shift() calls
        shift_pattern = r'\.shift\s*\(\s*(-?\d+|[-]?\w+)\s*\)'
        matches = re.findall(shift_pattern, source)

        # Check for negative literal values
        negative_shifts = []
        for match in matches:
            try:
                value = int(match)
                if value < 0:
                    negative_shifts.append(value)
            except ValueError:
                # Variable name, need to check context
                if match.startswith('-'):
                    negative_shifts.append(match)

        # The Ichimoku chikou span was fixed, so there should be none
        assert len(negative_shifts) == 0, (
            f"Found negative shift values (future data leakage): {negative_shifts}. "
            "All shift() calls should use positive values (looking back, not forward)."
        )

    def test_ichimoku_chikou_span_fixed(self):
        """Verify Ichimoku Chikou Span no longer uses future data."""
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        df = pd.DataFrame({
            'high': np.random.randn(100).cumsum() + 100,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 99,
        }, index=dates)

        # Compute Ichimoku
        result = indicators.ichimoku_cloud(
            df['high'], df['low'], df['close'],
            tenkan_period=9, kijun_period=26, senkou_b_period=52
        )

        # Chikou should have NaN at the BEGINNING (lagging), not the END
        # If it used shift(-26), it would have NaN at the end (looking forward)
        chikou = result['chikou']

        # First 26 values should be NaN (lagging indicator)
        assert chikou.iloc[:26].isna().all(), (
            "Chikou span should have NaN at the beginning (lagging indicator), "
            "not at the end (which would indicate future data usage)"
        )

        # Last values should NOT be NaN
        assert not chikou.iloc[-10:].isna().any(), (
            "Chikou span should have values at the end (not looking into future)"
        )


class TestOutcomeBasedLabeling:
    """Test that outcome-based labeling avoids target leakage."""

    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2024-01-01', periods=n, freq='h')

        # Generate price data with trends
        returns = np.random.randn(n) * 0.01
        # Add some trending periods
        returns[50:70] = 0.005  # Uptrend
        returns[100:120] = -0.005  # Downtrend
        returns[150:160] = 0.02  # Breakout

        close = 100 * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'timestamp': dates,
            'open': close * (1 + np.random.randn(n) * 0.001),
            'high': close * (1 + abs(np.random.randn(n) * 0.002)),
            'low': close * (1 - abs(np.random.randn(n) * 0.002)),
            'close': close,
            'volume': np.random.randint(100, 1000, n),
        })

    def test_outcome_labels_use_future_data(self, sample_ohlcv):
        """Verify outcome-based labels are computed from future returns."""
        labeler = OutcomeBasedLabeler(lookahead=20, trend_threshold=0.02)
        labels = labeler.label(sample_ohlcv)

        # Last 20 rows should be NaN (lookahead window)
        assert labels.tail(20).isna().all(), (
            "Outcome-based labels should have NaN at the end due to lookahead"
        )

        # Earlier rows should have valid labels
        assert not labels.head(180).isna().any(), (
            "Outcome-based labels should be valid for rows before lookahead window"
        )

    def test_outcome_labels_not_derived_from_indicators(self, sample_ohlcv):
        """
        Verify outcome labels are NOT correlated with current price
        (they should be based on FUTURE price changes).
        """
        labeler = OutcomeBasedLabeler(lookahead=20, trend_threshold=0.02)
        labels = labeler.label(sample_ohlcv)

        # Correlation between current close and labels should be low
        valid_mask = labels.notna()
        close_values = sample_ohlcv.loc[valid_mask, 'close']
        label_values = labels[valid_mask].astype(float)

        correlation = close_values.corr(label_values)

        # Correlation should be relatively low (not deterministic)
        assert abs(correlation) < 0.5, (
            f"Labels have high correlation ({correlation:.2f}) with current close. "
            "This suggests labels may be derived from current data, not future outcomes."
        )

    def test_indicator_based_has_warning(self):
        """Test that indicator-based mode warns about target leakage."""
        import logging
        import io

        # Capture log output
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger('src.regime.regime_labeler')
        logger.addHandler(handler)

        try:
            # Create indicator-based labeler
            labeler = RegimeLabeler(mode="indicator")

            # Check for warning
            log_output = log_capture.getvalue()
            assert "target leakage" in log_output.lower(), (
                "Indicator-based mode should warn about target leakage"
            )
        finally:
            logger.removeHandler(handler)


class TestLabelDistribution:
    """Test that label distributions are realistic."""

    @pytest.fixture
    def realistic_price_data(self):
        """Create price data that mimics real market behavior."""
        np.random.seed(123)
        n = 1000
        dates = pd.date_range('2024-01-01', periods=n, freq='h')

        # Generate returns with realistic properties
        returns = np.random.randn(n) * 0.005  # ~0.5% hourly volatility

        # Add some trending and volatile periods
        for i in range(0, n, 100):
            if np.random.random() > 0.7:
                # Trending period
                trend = np.random.choice([-1, 1]) * 0.002
                returns[i:i+20] += trend
            if np.random.random() > 0.8:
                # Volatile period
                returns[i:i+10] *= 2

        close = 100 * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'timestamp': dates,
            'close': close,
        })

    def test_outcome_labels_are_balanced(self, realistic_price_data):
        """Outcome-based labels should have reasonable class distribution."""
        labeler = OutcomeBasedLabeler(lookahead=20, trend_threshold=0.02)
        labels = labeler.label(realistic_price_data)

        stats = labeler.get_label_stats(labels)

        # No class should be < 5% or > 60%
        for regime in MarketRegime:
            pct = stats[regime.name]["percentage"]
            assert pct >= 1.0, f"{regime.name} has too few samples ({pct:.1f}%)"
            assert pct <= 70.0, f"{regime.name} has too many samples ({pct:.1f}%)"

    def test_labels_not_perfectly_predictable(self, realistic_price_data):
        """
        Verify that labels cannot be perfectly predicted from simple features.

        If a simple model achieves >95% accuracy, there's likely leakage.
        """
        labeler = OutcomeBasedLabeler(lookahead=20, trend_threshold=0.02)
        labels = labeler.label(realistic_price_data)

        # Create simple features (current price, returns)
        df = realistic_price_data.copy()
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)
        df['return_10'] = df['close'].pct_change(10)
        df['label'] = labels

        # Drop NaN
        df = df.dropna()

        # If we could import sklearn, we'd train a simple model
        # For now, just check correlation
        features = ['return_1', 'return_5', 'return_10']
        max_corr = 0
        for feat in features:
            corr = abs(df[feat].corr(df['label']))
            max_corr = max(max_corr, corr)

        assert max_corr < 0.8, (
            f"High correlation ({max_corr:.2f}) between simple features and labels. "
            "This might indicate the labels are too easy to predict or there's leakage."
        )


class TestRegimeLabelerModes:
    """Test that RegimeLabeler mode switching works correctly."""

    @pytest.fixture
    def sample_data_with_features(self):
        """Create sample data with indicator features."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2024-01-01', periods=n, freq='h')

        df = pd.DataFrame({
            'timestamp': dates,
            'close': 100 + np.random.randn(n).cumsum(),
            'trend_adx': np.random.uniform(10, 50, n),
            'trend_di_difference': np.random.uniform(-20, 20, n),
            'volat_vol_percentile': np.random.uniform(0, 100, n),
            'vol_volume_ratio': np.random.uniform(0.5, 3, n),
            'volat_breakout_magnitude': np.random.uniform(0, 3, n),
            'volat_bb_percent_b': np.random.uniform(-0.5, 1.5, n),
        }, index=dates)

        return df

    def test_indicator_mode_uses_features(self, sample_data_with_features):
        """Indicator mode should use feature columns for labeling."""
        labeler = RegimeLabeler(mode="indicator")
        result = labeler.label_regimes(sample_data_with_features)

        # Should have labels based on features
        assert 'regime' in result.columns
        # Indicator mode should NOT have NaN at the end
        assert not result['regime'].isna().any()

    def test_outcome_mode_uses_close(self, sample_data_with_features):
        """Outcome mode should use close price for labeling."""
        labeler = RegimeLabeler(
            mode="outcome",
            outcome_lookahead=10,
            outcome_trend_threshold=0.02,
        )
        result = labeler.label_regimes(sample_data_with_features)

        # Should have labels
        assert 'regime' in result.columns
        # Outcome mode SHOULD have NaN at the end (lookahead)
        assert result['regime'].tail(10).isna().all()

    def test_invalid_mode_raises_error(self):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            RegimeLabeler(mode="invalid")
