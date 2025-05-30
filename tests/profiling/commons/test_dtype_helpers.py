import pytest
import numpy as np
import pandas as pd
from pamola_core.profiling.commons import dtype_helpers

class Dummy:
    pass

def test_is_numeric_dtype_valid_cases():
    assert dtype_helpers.is_numeric_dtype(np.dtype('int64'))
    assert dtype_helpers.is_numeric_dtype(np.dtype('float32'))
    assert dtype_helpers.is_numeric_dtype(pd.Series([1, 2, 3]).dtype)
    assert dtype_helpers.is_numeric_dtype(pd.Series([1.1, 2.2, 3.3]).dtype)

def test_is_numeric_dtype_edge_cases():
    assert not dtype_helpers.is_numeric_dtype('object')
    assert not dtype_helpers.is_numeric_dtype(pd.Series(['a', 'b']))
    assert not dtype_helpers.is_numeric_dtype(None)
    assert not dtype_helpers.is_numeric_dtype('')
    assert not dtype_helpers.is_numeric_dtype(Dummy)

def test_is_numeric_dtype_invalid_input():
    assert not dtype_helpers.is_numeric_dtype([1, 2, 3])
    assert not dtype_helpers.is_numeric_dtype({'a': 1})
    assert not dtype_helpers.is_numeric_dtype(object)


def test_is_bool_dtype_valid_cases():
    assert dtype_helpers.is_bool_dtype(np.dtype('bool'))
    assert dtype_helpers.is_bool_dtype(pd.Series([True, False, True]).dtype)
    assert dtype_helpers.is_bool_dtype(np.dtype('bool'))

def test_is_bool_dtype_edge_cases():
    assert not dtype_helpers.is_bool_dtype('int64')
    assert not dtype_helpers.is_bool_dtype(pd.Series([1, 0, 1]))
    assert not dtype_helpers.is_bool_dtype(None)
    assert not dtype_helpers.is_bool_dtype('')
    assert not dtype_helpers.is_bool_dtype(Dummy)

def test_is_bool_dtype_invalid_input():
    assert not dtype_helpers.is_bool_dtype([True, False])
    assert not dtype_helpers.is_bool_dtype({'a': True})
    assert not dtype_helpers.is_bool_dtype(object)


def test_is_object_dtype_valid_cases():
    assert dtype_helpers.is_object_dtype(np.dtype('O'))
    assert dtype_helpers.is_object_dtype(pd.Series(['a', 'b']))
    assert dtype_helpers.is_object_dtype('object')
    class Custom:
        dtype = np.dtype('O')
    assert dtype_helpers.is_object_dtype(Custom())

def test_is_object_dtype_edge_cases():
    assert not dtype_helpers.is_object_dtype(np.dtype('int64'))
    assert not dtype_helpers.is_object_dtype(pd.Series([1, 2, 3]).dtype)
    assert not dtype_helpers.is_object_dtype(None)
    assert not dtype_helpers.is_object_dtype('')

def test_is_object_dtype_invalid_input():
    assert not dtype_helpers.is_object_dtype([1, 2, 3])
    assert not dtype_helpers.is_object_dtype({'a': 1})


def test_is_string_dtype_valid_cases():
    assert dtype_helpers.is_string_dtype('string')
    assert dtype_helpers.is_string_dtype('str')
    s = pd.Series(['a', 'b', 'c'])
    assert dtype_helpers.is_string_dtype(s)
    class Custom:
        dtype = np.dtype('O')
        def dropna(self):
            return ['a', 'b', 'c']
    assert dtype_helpers.is_string_dtype(Custom())

def test_is_string_dtype_edge_cases():
    s = pd.Series([1, 2, 3])
    assert not dtype_helpers.is_string_dtype(s)
    assert not dtype_helpers.is_string_dtype(np.dtype('int64'))
    assert not dtype_helpers.is_string_dtype(None)
    assert not dtype_helpers.is_string_dtype('')
    assert not dtype_helpers.is_string_dtype(Dummy)
    s_empty = pd.Series([], dtype=object)
    assert not dtype_helpers.is_string_dtype(s_empty)

def test_is_string_dtype_invalid_input():
    assert not dtype_helpers.is_string_dtype([1, 2, 3])
    assert not dtype_helpers.is_string_dtype({'a': 1})
    assert not dtype_helpers.is_string_dtype(object)


def test_is_datetime64_dtype_valid_cases():
    assert dtype_helpers.is_datetime64_dtype(np.dtype('datetime64[ns]'))
    assert dtype_helpers.is_datetime64_dtype(pd.Series(pd.date_range('2020-01-01', periods=3)).dtype)

def test_is_datetime64_dtype_edge_cases():
    assert not dtype_helpers.is_datetime64_dtype(np.dtype('int64'))
    assert not dtype_helpers.is_datetime64_dtype(pd.Series([1, 2, 3]))
    assert not dtype_helpers.is_datetime64_dtype(None)
    assert not dtype_helpers.is_datetime64_dtype('')
    assert not dtype_helpers.is_datetime64_dtype(Dummy)

def test_is_datetime64_dtype_invalid_input():
    assert not dtype_helpers.is_datetime64_dtype([1, 2, 3])
    assert not dtype_helpers.is_datetime64_dtype({'a': 1})
    assert not dtype_helpers.is_datetime64_dtype(object)


def test_is_categorical_dtype_valid_cases():
    s = pd.Series(['a', 'b', 'a'], dtype='category')
    assert dtype_helpers.is_categorical_dtype(s)
    class Cat:
        cat = True
    assert dtype_helpers.is_categorical_dtype(Cat())
    assert dtype_helpers.is_categorical_dtype('category')
    assert dtype_helpers.is_categorical_dtype('categorical')

def test_is_categorical_dtype_edge_cases():
    assert not dtype_helpers.is_categorical_dtype(np.dtype('int64'))
    assert not dtype_helpers.is_categorical_dtype(pd.Series([1, 2, 3]))
    assert not dtype_helpers.is_categorical_dtype(None)
    assert not dtype_helpers.is_categorical_dtype('')
    assert not dtype_helpers.is_categorical_dtype(Dummy)

def test_is_categorical_dtype_invalid_input():
    assert not dtype_helpers.is_categorical_dtype([1, 2, 3])
    assert not dtype_helpers.is_categorical_dtype({'a': 1})
    assert not dtype_helpers.is_categorical_dtype(object)


def test_is_integer_dtype_valid_cases():
    assert dtype_helpers.is_integer_dtype(np.dtype('int64'))
    assert dtype_helpers.is_integer_dtype(np.dtype('uint8'))
    assert dtype_helpers.is_integer_dtype(pd.Series([1, 2, 3]).dtype)

def test_is_integer_dtype_edge_cases():
    assert not dtype_helpers.is_integer_dtype(np.dtype('float64'))
    assert not dtype_helpers.is_integer_dtype(pd.Series([1.1, 2.2]))
    assert not dtype_helpers.is_integer_dtype(None)
    assert not dtype_helpers.is_integer_dtype('')
    assert not dtype_helpers.is_integer_dtype(Dummy)

def test_is_integer_dtype_invalid_input():
    assert not dtype_helpers.is_integer_dtype([1, 2, 3])
    assert not dtype_helpers.is_integer_dtype({'a': 1})
    assert not dtype_helpers.is_integer_dtype(object)


def test_is_float_dtype_valid_cases():
    assert dtype_helpers.is_float_dtype(np.dtype('float64'))
    assert dtype_helpers.is_float_dtype(np.dtype('float32'))
    assert dtype_helpers.is_float_dtype(pd.Series([1.1, 2.2, 3.3]).dtype)

def test_is_float_dtype_edge_cases():
    assert not dtype_helpers.is_float_dtype(np.dtype('int64'))
    assert not dtype_helpers.is_float_dtype(pd.Series([1, 2, 3]))
    assert not dtype_helpers.is_float_dtype(None)
    assert not dtype_helpers.is_float_dtype('')
    assert not dtype_helpers.is_float_dtype(Dummy)

def test_is_float_dtype_invalid_input():
    assert not dtype_helpers.is_float_dtype([1.1, 2.2])
    assert not dtype_helpers.is_float_dtype({'a': 1.1})
    assert not dtype_helpers.is_float_dtype(object)


def test_is_list_like_valid_cases():
    assert dtype_helpers.is_list_like([1, 2, 3])
    assert dtype_helpers.is_list_like((1, 2, 3))
    assert dtype_helpers.is_list_like(np.array([1, 2, 3]))
    assert dtype_helpers.is_list_like(pd.Series([1, 2, 3]))
    assert dtype_helpers.is_list_like({1, 2, 3})
    assert dtype_helpers.is_list_like(range(5))

def test_is_list_like_edge_cases():
    assert not dtype_helpers.is_list_like('abc')
    assert not dtype_helpers.is_list_like(b'abc')
    assert not dtype_helpers.is_list_like(123)
    assert not dtype_helpers.is_list_like(None)
    assert not dtype_helpers.is_list_like(Dummy())

def test_is_dict_like_valid_cases():
    assert dtype_helpers.is_dict_like({'a': 1, 'b': 2})
    class DictLike:
        def keys(self):
            return ['a']
        def __getitem__(self, key):
            return 1
    assert dtype_helpers.is_dict_like(DictLike())

def test_is_dict_like_edge_cases():
    assert not dtype_helpers.is_dict_like([('a', 1)])
    assert not dtype_helpers.is_dict_like('abc')
    assert not dtype_helpers.is_dict_like(123)
    assert not dtype_helpers.is_dict_like(None)
    assert not dtype_helpers.is_dict_like(Dummy())

if __name__ == "__main__":
    pytest.main()
