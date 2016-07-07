#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Miscellaneous utility tests'''

import pytest
import numpy as np

import pumpp


@pytest.mark.parametrize('dtype',
                         [int, np.int64,
                          pytest.mark.xfail('not a type',
                                            raises=TypeError)])
def test_scope_type(dtype):

    scope = pumpp.core.Scope(None)
    scope.register('foo', [None], dtype)


@pytest.mark.parametrize('shape',
                         [[None], [1], [1, None],
                          pytest.mark.xfail(1, raises=ValueError),
                          pytest.mark.xfail(None, raises=ValueError),
                          pytest.mark.xfail(23.5, raises=ValueError),
                          pytest.mark.xfail('not a shape', raises=ValueError)])
def test_scope_badshape(shape):

    scope = pumpp.core.Scope(None)
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
