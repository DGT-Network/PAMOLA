"""
PAMOLA Core Metrics Package: Unit Tests for SafeInstantiate
==========================================================
File:        tests/metrics/commons/test_safe_instantiate.py
Target:      pamola_core.metrics.commons.safe_instantiate
Coverage:    100% line coverage (see docs)
Top-matter:  Standardized (see process docs)

Description:
    Comprehensive unit tests for safe_instantiate, including:
    - Instantiation with valid and extra parameters
    - Edge cases and error handling
    - Compliance with ≥90% line coverage and process requirements

Process:
    - All tests must be self-contained and not depend on external state.
    - All branches and error paths must be exercised.
    - Top-matter must be present and up to date.
    - See process documentation for details.
    
**Version:** 4.0.0
**Coverage Status:** ✅ Full
**Last Updated:** 2025-07-23
"""

import pytest
from pamola_core.metrics.commons.safe_instantiate import safe_instantiate

class DummyMetric:
    def __init__(self, a, b=2):
        self.a = a
        self.b = b

class NoArgMetric:
    def __init__(self):
        self.x = 1

def test_safe_instantiate_valid_params():
    obj = safe_instantiate(DummyMetric, {"a": 5, "b": 10})
    assert obj.a == 5 and obj.b == 10

def test_safe_instantiate_extra_params():
    obj = safe_instantiate(DummyMetric, {"a": 1, "b": 2, "c": 99})
    assert obj.a == 1 and obj.b == 2
    # 'c' should be ignored
    assert not hasattr(obj, 'c')

def test_safe_instantiate_missing_optional():
    obj = safe_instantiate(DummyMetric, {"a": 7})
    assert obj.a == 7 and obj.b == 2

def test_safe_instantiate_no_args():
    obj = safe_instantiate(NoArgMetric, {"foo": 123})
    assert isinstance(obj, NoArgMetric)
    assert obj.x == 1

def test_safe_instantiate_missing_required():
    with pytest.raises(TypeError):
        safe_instantiate(DummyMetric, {"b": 2})
