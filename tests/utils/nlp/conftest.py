"""Isolation for the NLP tests.

``NLPModelManager`` is a process-wide singleton and ``get_nlp_model_manager()`` caches a second
handle. A test that swaps its model cache for a mock (or leaves a patched ``get_cache`` result
behind) would otherwise hand that state to every later test that calls ``NLPModelManager()`` -
the offline-by-default tests then saw a cached "model" instead of the typed error they assert
(ASB-10). Every test here starts and ends with a fresh manager.
"""

import pytest


def _reset_model_manager():
    from pamola_core.utils.nlp import model_manager

    model_manager.NLPModelManager._instance = None
    model_manager._nlp_model_manager = None


@pytest.fixture(autouse=True)
def _fresh_nlp_model_manager():
    _reset_model_manager()
    yield
    _reset_model_manager()
