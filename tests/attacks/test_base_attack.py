"""
PAMOLA - Privacy-Aware Machine Learning Analytics
Unit Tests for BaseAttack (AttackInitialization)

This module contains comprehensive unit tests for the AttackInitialization
(BaseAttack) abstract base class. Tests cover:
- Abstract class instantiation validation
- Abstract method enforcement
- Basic attack class inheritance patterns
"""

import unittest
from abc import ABC
import pandas as pd
import numpy as np
from pamola_core.attacks.base import AttackInitialization


class ConcreteAttackImplementation(AttackInitialization):
    """Concrete implementation for testing abstract base class"""

    def preprocess_data(self, data1, data2):
        """Simple preprocessing implementation"""
        return np.array(data1), np.array(data2)

    def process(self, data):
        """Simple process implementation"""
        return data


class TestBaseAttackInitialization(unittest.TestCase):
    """Test BaseAttack (AttackInitialization) abstract base class"""

    def test_base_attack_is_abstract(self):
        """Verify AttackInitialization is abstract and cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            AttackInitialization()

    def test_concrete_implementation_instantiation(self):
        """Verify concrete implementation of abstract class can be instantiated"""
        attack = ConcreteAttackImplementation()
        self.assertIsInstance(attack, AttackInitialization)
        self.assertIsInstance(attack, ABC)

    def test_abstract_method_preprocess_data_required(self):
        """Verify preprocess_data is abstract and must be implemented"""
        class IncompleteAttack(AttackInitialization):
            def process(self, data):
                pass

        with self.assertRaises(TypeError):
            IncompleteAttack()

    def test_abstract_method_process_required(self):
        """Verify process is abstract and must be implemented"""
        class IncompleteAttack(AttackInitialization):
            def preprocess_data(self, data1, data2):
                pass

        with self.assertRaises(TypeError):
            IncompleteAttack()

    def test_preprocess_data_method_exists(self):
        """Verify preprocess_data method is accessible"""
        attack = ConcreteAttackImplementation()
        self.assertTrue(hasattr(attack, 'preprocess_data'))
        self.assertTrue(callable(getattr(attack, 'preprocess_data')))

    def test_process_method_exists(self):
        """Verify process method is accessible"""
        attack = ConcreteAttackImplementation()
        self.assertTrue(hasattr(attack, 'process'))
        self.assertTrue(callable(getattr(attack, 'process')))

    def test_preprocess_data_with_simple_dataframes(self):
        """Test preprocess_data with simple DataFrames"""
        attack = ConcreteAttackImplementation()
        df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})

        result1, result2 = attack.preprocess_data(df1, df2)

        self.assertIsInstance(result1, np.ndarray)
        self.assertIsInstance(result2, np.ndarray)
        self.assertEqual(result1.shape[0], 2)
        self.assertEqual(result2.shape[0], 2)

    def test_process_with_data(self):
        """Test process method with sample data"""
        attack = ConcreteAttackImplementation()
        data = pd.DataFrame({'x': [1, 2, 3]})

        result = attack.process(data)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)

    def test_multiple_method_signatures(self):
        """Verify abstract methods have correct signatures"""
        attack = ConcreteAttackImplementation()

        # preprocess_data should accept 3 parameters (self, data1, data2)
        import inspect
        preprocess_sig = inspect.signature(attack.preprocess_data)
        self.assertEqual(len(preprocess_sig.parameters), 2)  # data1, data2 (self not counted)

        # process should accept 2 parameters (self, data)
        process_sig = inspect.signature(attack.process)
        self.assertEqual(len(process_sig.parameters), 1)  # data (self not counted)


if __name__ == '__main__':
    unittest.main()
