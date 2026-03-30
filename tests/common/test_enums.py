"""
Unit tests for pamola_core.common.enum module enumerations.

Tests cover:
- MaskStrategyEnum (masking strategies)
- DistanceMetricType (distance metrics)
- FidelityMetricsType (fidelity metrics)
- Other common enums
- Enum member values and iteration

Run with: pytest -s tests/common/test_enums.py
"""

import pytest

from pamola_core.common.enum.mask_strategy_enum import MaskStrategyEnum
from pamola_core.common.enum.distance_metric_type import DistanceMetricType
from pamola_core.common.enum.fidelity_metrics_type import FidelityMetricsType


class TestMaskStrategyEnum:
    """Test MaskStrategyEnum enumeration."""

    def test_mask_strategy_has_members(self):
        """MaskStrategyEnum should have all required members."""
        required = {"FIXED", "PATTERN", "RANDOM", "WORDS"}
        members = {m.name for m in MaskStrategyEnum}
        assert required.issubset(members)

    def test_fixed_strategy_value(self):
        """FIXED strategy should have correct value."""
        assert MaskStrategyEnum.FIXED.value == "fixed"

    def test_pattern_strategy_value(self):
        """PATTERN strategy should have correct value."""
        assert MaskStrategyEnum.PATTERN.value == "pattern"

    def test_random_strategy_value(self):
        """RANDOM strategy should have correct value."""
        assert MaskStrategyEnum.RANDOM.value == "random"

    def test_words_strategy_value(self):
        """WORDS strategy should have correct value."""
        assert MaskStrategyEnum.WORDS.value == "words"

    def test_all_strategy_values_are_strings(self):
        """All strategy values should be strings."""
        for strategy in MaskStrategyEnum:
            assert isinstance(strategy.value, str)

    def test_can_create_from_value(self):
        """Should be able to create enum from value."""
        strategy = MaskStrategyEnum("fixed")
        assert strategy == MaskStrategyEnum.FIXED

    def test_can_access_by_name(self):
        """Should be able to access by name."""
        assert MaskStrategyEnum["FIXED"] == MaskStrategyEnum.FIXED

    def test_iterate_strategies(self):
        """Should be able to iterate over all strategies."""
        strategies = list(MaskStrategyEnum)
        assert len(strategies) == 4

    def test_strategy_comparison(self):
        """Should support equality comparison."""
        assert MaskStrategyEnum.FIXED == MaskStrategyEnum.FIXED
        assert MaskStrategyEnum.FIXED != MaskStrategyEnum.RANDOM

    def test_invalid_strategy_raises_error(self):
        """Should raise ValueError for invalid strategy."""
        with pytest.raises(ValueError):
            MaskStrategyEnum("invalid_strategy")

    def test_strategy_uniqueness(self):
        """All strategy values should be unique."""
        values = [s.value for s in MaskStrategyEnum]
        assert len(values) == len(set(values))


class TestDistanceMetricType:
    """Test DistanceMetricType enumeration."""

    def test_distance_metric_has_members(self):
        """DistanceMetricType should have required members."""
        required = {"EUCLIDEAN", "MANHATTAN", "COSINE", "MAHALANOBIS"}
        members = {m.name for m in DistanceMetricType}
        assert required.issubset(members)

    def test_euclidean_metric_value(self):
        """EUCLIDEAN metric should have correct value."""
        assert DistanceMetricType.EUCLIDEAN.value == "euclidean"

    def test_manhattan_metric_value(self):
        """MANHATTAN metric should have correct value."""
        assert DistanceMetricType.MANHATTAN.value == "manhattan"

    def test_cosine_metric_value(self):
        """COSINE metric should have correct value."""
        assert DistanceMetricType.COSINE.value == "cosine"

    def test_mahalanobis_metric_value(self):
        """MAHALANOBIS metric should have correct value."""
        assert DistanceMetricType.MAHALANOBIS.value == "mahalanobis"

    def test_all_metric_values_are_strings(self):
        """All metric values should be strings."""
        for metric in DistanceMetricType:
            assert isinstance(metric.value, str)

    def test_can_create_from_value(self):
        """Should be able to create enum from value."""
        metric = DistanceMetricType("euclidean")
        assert metric == DistanceMetricType.EUCLIDEAN

    def test_can_access_by_name(self):
        """Should be able to access by name."""
        assert DistanceMetricType["EUCLIDEAN"] == DistanceMetricType.EUCLIDEAN

    def test_iterate_metrics(self):
        """Should be able to iterate over all metrics."""
        metrics = list(DistanceMetricType)
        assert len(metrics) == 4

    def test_metric_comparison(self):
        """Should support equality comparison."""
        assert DistanceMetricType.EUCLIDEAN == DistanceMetricType.EUCLIDEAN
        assert DistanceMetricType.EUCLIDEAN != DistanceMetricType.COSINE

    def test_invalid_metric_raises_error(self):
        """Should raise ValueError for invalid metric."""
        with pytest.raises(ValueError):
            DistanceMetricType("invalid_metric")

    def test_metric_uniqueness(self):
        """All metric values should be unique."""
        values = [m.value for m in DistanceMetricType]
        assert len(values) == len(set(values))


class TestFidelityMetricsType:
    """Test FidelityMetricsType enumeration."""

    def test_fidelity_metrics_has_members(self):
        """FidelityMetricsType should have required members."""
        required = {"KS", "KL", "JS", "WASSERSTEIN"}
        members = {m.name for m in FidelityMetricsType}
        assert required.issubset(members)

    def test_ks_metric_value(self):
        """KS metric should have correct value."""
        assert FidelityMetricsType.KS.value == "ks"

    def test_kl_metric_value(self):
        """KL metric should have correct value."""
        assert FidelityMetricsType.KL.value == "kl"

    def test_js_metric_value(self):
        """JS metric should have correct value."""
        assert FidelityMetricsType.JS.value == "js"

    def test_wasserstein_metric_value(self):
        """WASSERSTEIN metric should have correct value."""
        assert FidelityMetricsType.WASSERSTEIN.value == "wasserstein"

    def test_all_fidelity_values_are_strings(self):
        """All fidelity metric values should be strings."""
        for metric in FidelityMetricsType:
            assert isinstance(metric.value, str)

    def test_can_create_from_value(self):
        """Should be able to create enum from value."""
        metric = FidelityMetricsType("ks")
        assert metric == FidelityMetricsType.KS

    def test_can_access_by_name(self):
        """Should be able to access by name."""
        assert FidelityMetricsType["KS"] == FidelityMetricsType.KS

    def test_iterate_metrics(self):
        """Should be able to iterate over all metrics."""
        metrics = list(FidelityMetricsType)
        assert len(metrics) == 4

    def test_metric_comparison(self):
        """Should support equality comparison."""
        assert FidelityMetricsType.KS == FidelityMetricsType.KS
        assert FidelityMetricsType.KS != FidelityMetricsType.KL

    def test_invalid_metric_raises_error(self):
        """Should raise ValueError for invalid metric."""
        with pytest.raises(ValueError):
            FidelityMetricsType("invalid_metric")

    def test_metric_uniqueness(self):
        """All metric values should be unique."""
        values = [m.value for m in FidelityMetricsType]
        assert len(values) == len(set(values))


class TestEnumIntegration:
    """Integration tests across different enums."""

    def test_enum_in_list(self):
        """Should work as list elements."""
        strategies = [
            MaskStrategyEnum.FIXED,
            MaskStrategyEnum.PATTERN,
            MaskStrategyEnum.RANDOM
        ]
        assert len(strategies) == 3
        assert MaskStrategyEnum.FIXED in strategies

    def test_enum_in_dict(self):
        """Should work as dictionary keys."""
        config = {
            DistanceMetricType.EUCLIDEAN: {"threshold": 0.5},
            DistanceMetricType.COSINE: {"threshold": 0.7}
        }
        assert config[DistanceMetricType.EUCLIDEAN]["threshold"] == 0.5

    def test_enum_in_set(self):
        """Should work in sets."""
        metrics = {
            FidelityMetricsType.KS,
            FidelityMetricsType.KL,
            FidelityMetricsType.KS  # Duplicate
        }
        assert len(metrics) == 2

    def test_enum_as_function_parameter(self):
        """Should work as function parameters."""
        def apply_strategy(strategy: MaskStrategyEnum) -> str:
            return f"Using {strategy.value} strategy"

        result = apply_strategy(MaskStrategyEnum.FIXED)
        assert "fixed" in result

    def test_enums_are_serializable_to_string(self):
        """Enums should be convertible to strings for config files."""
        strategy = MaskStrategyEnum.PATTERN
        serialized = strategy.value
        assert isinstance(serialized, str)

        # And back
        restored = MaskStrategyEnum(serialized)
        assert restored == strategy

    def test_multiple_enum_types_independent(self):
        """Different enum types should be independent."""
        strategy = MaskStrategyEnum.FIXED
        metric = DistanceMetricType.EUCLIDEAN

        assert strategy != metric
        assert type(strategy) != type(metric)


class TestEnumEdgeCases:
    """Edge case tests for enums."""

    def test_enum_hash(self):
        """Enums should be hashable."""
        metrics = {DistanceMetricType.EUCLIDEAN, DistanceMetricType.COSINE}
        assert len(metrics) == 2

    def test_enum_in_tuple(self):
        """Should work in tuples."""
        config = (MaskStrategyEnum.FIXED, "config.json")
        assert config[0] == MaskStrategyEnum.FIXED

    def test_enum_with_getattr(self):
        """Should support attribute access."""
        strategy = MaskStrategyEnum.FIXED
        assert hasattr(strategy, "value")
        assert hasattr(strategy, "name")

    def test_enum_value_case_sensitive(self):
        """Enum value lookup should be case-sensitive."""
        with pytest.raises(ValueError):
            MaskStrategyEnum("FIXED")  # All uppercase

    def test_can_convert_to_list(self):
        """Should be able to convert enum to list."""
        strategies_list = list(MaskStrategyEnum)
        assert len(strategies_list) > 0
        assert MaskStrategyEnum.FIXED in strategies_list
