#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Miscellaneous utility tests'''

import pytest
import numpy as np

import pumpp

from pumpp import ParameterError

@pytest.mark.parametrize('dtype',
                         [int, np.int64,
                          pytest.mark.xfail('not a type',
                                            raises=ParameterError)])
def test_scope_type(dtype):

    scope = pumpp.base.Scope(None)
    scope.register('foo', [None], dtype)


@pytest.mark.parametrize('shape',
                         [[None], [1], [1, None],
                          pytest.mark.xfail(1, raises=ParameterError),
                          pytest.mark.xfail(None, raises=ParameterError),
                          pytest.mark.xfail(23.5, raises=ParameterError),
                          pytest.mark.xfail('not a shape', raises=ParameterError)])
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
