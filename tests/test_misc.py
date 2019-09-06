#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Miscellaneous utility tests'''

import pytest
import numpy as np

import jams
import pumpp
from pumpp.util import match_query

from pumpp import ParameterError

xfail = pytest.mark.xfail


@pytest.mark.parametrize('dtype',
                         [int, np.int64,
                          pytest.param('not a type',
                                       marks=xfail(raises=ParameterError))])
def test_scope_type(dtype):

    scope = pumpp.base.Scope(None)
    scope.register('foo', [None], dtype)


@pytest.mark.parametrize('shape',
                         [[None], [1], [1, None],
                          pytest.param(1, marks=xfail(raises=ParameterError)),
                          pytest.param(None, marks=xfail(raises=ParameterError)),
                          pytest.param(23.5, marks=xfail(raises=ParameterError)),
                          pytest.param('not a shape', marks=xfail(raises=ParameterError))])
def test_scope_badshape(shape):

    scope = pumpp.base.Scope(None)
    scope.register('foo', shape, int)


def test_bad_extractor():
    ext = pumpp.feature.FeatureExtractor(None, 22050, 512)

    with pytest.raises(NotImplementedError):
        ext.transform(np.zeros(1024), 22050)


@pytest.mark.parametrize('dtype, fill',
                         [(int, 0),
                          (bool, False),
                          (float, np.nan),
                          (complex, np.nan)])
def test_fill_value(dtype, fill):

    v = pumpp.task.base.fill_value(dtype)

    assert isinstance(v, dtype)
    assert v == fill or np.isnan(v) and np.isnan(fill)

@pytest.mark.parametrize(
    "expected,data,query",
    [
        (True,  5, None),
        (True,  5, 5),
        (True,  5, lambda x: x > 4),
        (True,  {'a': 'abc', 'b': 'def'}, {'b': 'def'}),
        (False, {'a': 'abc', 'b': 'def'}, {'b': 'abc'}),
        (True,  {'a': 'abc', 'b': 'def'}, {'b': lambda x: 'd' in x}),
        (False, {'a': 'abc', 'b': 'def'}, {'a': 'abc', 'b': 'deg'}),
        (True,  [1, 2, 3, 4, 5], lambda x: 5 in x),
        (True,  [1, 2, 3, 4, 5], [1, 2, 3, 4, lambda x: x in (5, 6, 7)]),
        (True,  5, {5, 9, 10}),
        (True,  'def', 'abc|def'),
        (False, {'b': 'def'}, {'b': 'def', 'c': 'ghi'}), # has key outside
        pytest.param(
            False, 5, {'b': 'def', 'c': 'ghi'}, marks=xfail(raises=ValueError)),
        pytest.param(
            False, 5, ['def', 'ghi'], marks=xfail(raises=ValueError)),
        pytest.param(
            False, [5], ['def', 'ghi'], marks=xfail(raises=ValueError)), ])
def test_match_query(expected, data, query):
    assert expected  == match_query(data, query)

@pytest.mark.parametrize(
    "expected,data,query",
    [(True, {'b': 'def', 'c': 'ghi'}, {'b': 'def'}),
    pytest.param(
        False, {'b': 'def'}, {'b': 'def', 'c': 'ghi'}, marks=xfail(raises=ValueError)), ])
def test_match_query_strict(expected, data, query):
    assert expected  == match_query(data, query, strict_keys=True)


def test_get_dtype():
    for namespace in jams.schema.__NAMESPACE__:
        schema = jams.schema.namespace(namespace)
        pumpp.task.lambd._get_dtype(schema)

        val = schema['properties']['value']
        try:
            value_has_defined_keys = (
                val['type'] == 'object' and 'properties' in val)
        except KeyError:
            value_has_defined_keys = False

        if value_has_defined_keys:
            for name, spec in val['properties'].items():
                pumpp.task.lambd._get_dtype(spec)

    # others that aren't tested:
    assert np.object_ == pumpp.task.lambd._get_dtype({'type': ['number', 'string']})
    assert np.object_ == pumpp.task.lambd._get_dtype({'enum': ['asdf', 5, 6]})
    assert np.float64 == pumpp.task.lambd._get_dtype({'oneOf': [{'type': 'number'}, {'type': 'null'}]})
    assert np.object_ == pumpp.task.lambd._get_dtype({'oneOf': [{'type': 'number'}, {'type': 'string'}]})
    assert np.object_ == pumpp.task.lambd._get_dtype({})
