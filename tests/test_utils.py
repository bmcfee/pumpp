#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tests for feature utility helpers'''

import pytest
import numpy as np

import pumpp
import pumpp.feature._utils


@pytest.mark.parametrize('dtype', ['uint8', np.uint8])
def test_quantize(dtype):

    # The range -5 to 5 is broken into 256 equal pieces
    # -5/3 lands at 85 (1/3)
    # 5/3 lands at 2*85 = 270
    # 5 lands at the max
    x = np.asarray([-5, -5/3, 5/3, 5])
    y = pumpp.feature._utils.quantize(x, dtype=dtype)
    assert np.allclose(y, [0, 85, 170, 255])


def test_quantize_min():
    x = np.asarray([-5, -5/3, 5/3, 5])
    y = pumpp.feature._utils.quantize(x, ref_min=0)
    assert np.allclose(y, [0, 0, 85, 255])


def test_quantize_max():
    x = np.asarray([-5, -5/3, 5/3, 5])
    y = pumpp.feature._utils.quantize(x, ref_max=0)
    assert np.allclose(y, [0, 170, 255, 255])


@pytest.mark.xfail(raises=pumpp.ParameterError)
@pytest.mark.parametrize('dtype', ['int8', 'float32'])
def test_quantize_bad_dtype(dtype):
    x = np.asarray([-5, -5/3, 5/3, 5])
    pumpp.feature._utils.quantize(x, dtype=dtype)
