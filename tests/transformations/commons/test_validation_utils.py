"""
Tests for the validation_utils module in the PAMOLA.CORE package.

These tests verify the functionality of data validation utilities including
field existence, type checking, parameter validation, constraint enforcement, and group/aggregation validation.

Run with:
    pytest tests/transformations/commons/test_validation_utils.py
"""
import pytest
import pandas as pd
from pamola_core.transformations.commons import validation_utils
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

class TestValidationUtils:
    def setup_method(self):
        self.df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': ['x', 'y', 'z', 'x', 'y'],
            'd': [None, 2, 3, None, 5],
            'e': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'])
        })

    def teardown_method(self):
        pass

    # validate_fields_exist
    def test_validate_fields_exist_valid(self):
        result, missing = validation_utils.validate_fields_exist(self.df, ['a', 'b'])
        assert result is True
        assert missing is None

    def test_validate_fields_exist_missing(self):
        result, missing = validation_utils.validate_fields_exist(self.df, ['a', 'z'])
        assert result is False
        assert missing == ['z']

    def test_validate_fields_exist_invalid_df(self):
        with pytest.raises(TypeError):
            validation_utils.validate_fields_exist([1, 2, 3], ['a'])

    def test_validate_fields_exist_invalid_fields(self):
        with pytest.raises(TypeError):
            validation_utils.validate_fields_exist(self.df, 'a')

    # validate_field_types
    def test_validate_field_types_valid(self):
        result, errors = validation_utils.validate_field_types(self.df, {'a': 'int64', 'b': 'int64'})
        assert result is True
        assert errors is None

    def test_validate_field_types_numeric(self):
        result, errors = validation_utils.validate_field_types(self.df, {'a': 'numeric'})
        assert result is True
        assert errors is None

    def test_validate_field_types_datetime(self):
        result, errors = validation_utils.validate_field_types(self.df, {'e': 'datetime'})
        assert result is True
        assert errors is None

    def test_validate_field_types_type_mismatch(self):
        result, errors = validation_utils.validate_field_types(self.df, {'a': 'float64'})
        assert result is False
        assert 'a' in errors

    def test_validate_field_types_missing_field(self):
        with pytest.raises(ValueError):
            validation_utils.validate_field_types(self.df, {'z': 'int64'})

    def test_validate_field_types_invalid_df(self):
        with pytest.raises(TypeError):
            validation_utils.validate_field_types([1, 2, 3], {'a': 'int64'})

    def test_validate_field_types_invalid_types(self):
        with pytest.raises(TypeError):
            validation_utils.validate_field_types(self.df, [('a', 'int64')])

    # validate_parameters
    def test_validate_parameters_valid(self):
        params = {'x': 1, 'y': 'foo'}
        required = ['x', 'y']
        types = {'x': int, 'y': str}
        result, errors = validation_utils.validate_parameters(params, required, types)
        assert result is True
        assert errors is None

    def test_validate_parameters_missing_param(self):
        params = {'x': 1}
        required = ['x', 'y']
        types = {'x': int, 'y': str}
        result, errors = validation_utils.validate_parameters(params, required, types)
        assert result is False
        assert any('Missing required parameter' in e for e in errors)

    def test_validate_parameters_type_error(self):
        params = {'x': 1, 'y': 2}
        required = ['x', 'y']
        types = {'x': int, 'y': str}
        result, errors = validation_utils.validate_parameters(params, required, types)
        assert result is False
        assert any('expected type str' in e for e in errors)

    def test_validate_parameters_union_type(self):
        from typing import Union
        params = {'x': 1, 'y': 'foo'}
        required = ['x', 'y']
        types = {'x': Union[int, float], 'y': str}
        result, errors = validation_utils.validate_parameters(params, required, types)
        assert result is True
        assert errors is None

    def test_validate_parameters_invalid_dict(self):
        with pytest.raises(TypeError):
            validation_utils.validate_parameters('notadict', ['x'], {'x': int})

    def test_validate_parameters_invalid_required(self):
        with pytest.raises(TypeError):
            validation_utils.validate_parameters({'x': 1}, 'x', {'x': int})

    def test_validate_parameters_invalid_types(self):
        with pytest.raises(TypeError):
            validation_utils.validate_parameters({'x': 1}, ['x'], [('x', int)])

    # validate_constraints
    def test_validate_constraints_not_null(self):
        df = pd.DataFrame({'a': [1, None, 3]})
        constraints = {'a': {'not_null': True}}
        result = validation_utils.validate_constraints(df, constraints)
        assert 'a' in result
        assert 'not_null' in result['a']

    def test_validate_constraints_min_max(self):
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        constraints = {'a': {'min': 3, 'max': 4}}
        result = validation_utils.validate_constraints(df, constraints)
        assert 'a' in result
        assert 'min' in result['a']
        assert 'max' in result['a']

    def test_validate_constraints_allowed_values(self):
        df = pd.DataFrame({'a': [1, 2, 3, 4]})
        constraints = {'a': {'allowed_values': [1, 2]}}
        result = validation_utils.validate_constraints(df, constraints)
        assert 'a' in result
        assert 'allowed_values' in result['a']

    def test_validate_constraints_unique(self):
        df = pd.DataFrame({'a': [1, 1, 2, 3]})
        constraints = {'a': {'unique': True}}
        result = validation_utils.validate_constraints(df, constraints)
        assert 'a' in result
        assert 'unique' in result['a']

    def test_validate_constraints_regex(self):
        df = pd.DataFrame({'a': ['abc', 'def', '123']})
        constraints = {'a': {'regex': r'^[a-z]+$'}}
        result = validation_utils.validate_constraints(df, constraints)
        assert 'a' in result
        assert 'regex' in result['a']

    def test_validate_constraints_no_violations(self):
        df = pd.DataFrame({'a': [1, 2, 3]})
        constraints = {'a': {'min': 1, 'max': 3, 'not_null': True, 'allowed_values': [1, 2, 3], 'unique': True}}
        result = validation_utils.validate_constraints(df, constraints)
        assert result == {}

    def test_validate_constraints_missing_field(self):
        df = pd.DataFrame({'a': [1, 2, 3]})
        constraints = {'b': {'min': 1}}
        with pytest.raises(ValueError):
            validation_utils.validate_constraints(df, constraints)

    def test_validate_constraints_invalid_df(self):
        with pytest.raises(TypeError):
            validation_utils.validate_constraints('notadf', {'a': {'min': 1}})

    def test_validate_constraints_invalid_constraints(self):
        with pytest.raises(TypeError):
            validation_utils.validate_constraints(self.df, [('a', {'min': 1})])

    # validate_dataframe
    def test_validate_dataframe_valid(self):
        validation_utils.validate_dataframe(self.df, ['a', 'b'])

    def test_validate_dataframe_missing(self):
        with pytest.raises(ValueError):
            validation_utils.validate_dataframe(self.df, ['a', 'z'])

    # validate_group_and_aggregation_fields
    def test_validate_group_and_aggregation_fields_valid(self):
        validation_utils.validate_group_and_aggregation_fields(self.df, ['a'], {'b': ['sum']}, None)

    def test_validate_group_and_aggregation_fields_custom(self):
        validation_utils.validate_group_and_aggregation_fields(self.df, ['a'], None, {'b': sum})

    def test_validate_group_and_aggregation_fields_missing(self):
        with pytest.raises(ValueError):
            validation_utils.validate_group_and_aggregation_fields(self.df, ['z'], {'b': ['sum']}, None)

    # validate_join_type
    @pytest.mark.parametrize('join_type', ['left', 'right', 'inner', 'outer'])
    def test_validate_join_type_valid(self, join_type):
        validation_utils.validate_join_type(join_type)

    def test_validate_join_type_invalid(self):
        with pytest.raises(ValueError):
            validation_utils.validate_join_type('invalid')

if __name__ == "__main__":
    pytest.main()
