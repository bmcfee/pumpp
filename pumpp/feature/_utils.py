#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Utilities for feature extraction classes'''

import numpy as np

from ..exceptions import ParameterError


def phase_diff(phase, conv):
    '''Compute the phase differential along a given axis

    Parameters
    ----------
    phase : np.ndarray
        Input phase (in radians)

    conv: {None, 'tf', 'th', 'channels_last', 'channels_first'}
        Convolution mode

    Returns
    -------
    dphase : np.ndarray like `phase`
        The phase differential.
    '''

    if conv is None:
        axis = 0
    elif conv in ('channels_last', 'tf'):
        axis = 0
    elif conv in ('channels_first', 'th'):
        axis = 1

    # Compute the phase differential
    dphase = np.empty(phase.shape, dtype=phase.dtype)
    zero_idx = [slice(None)] * phase.ndim
    zero_idx[axis] = slice(1)
    else_idx = [slice(None)] * phase.ndim
    else_idx[axis] = slice(1, None)
    zero_idx = tuple(zero_idx)
    else_idx = tuple(else_idx)
    dphase[zero_idx] = phase[zero_idx]
    dphase[else_idx] = np.diff(np.unwrap(phase, axis=axis), axis=axis)
    return dphase


def quantize(x, ref_min=None, ref_max=None, dtype='uint8'):
    '''Quantize array entries to a fixed dtype.

    Parameters
    ----------
    x : np.ndarray
        The data to quantize

    ref_min : None or float

    ref_max : None or float
        The reference minimum (maximum) value for quantization.
        By default, `x.min()` (`x.max()`)

    dtype : np.dtype {'uint8', 'uint16'}
        The target data type.  Any unsigned int type is supported,
        but most cases will call for `uint8`.

    Returns
    -------
    y : np.ndarray, dtype=dtype
        The values of `x` quantized to integer values
    '''

    if ref_min is None:
        ref_min = np.min(x)

    if ref_max is None:
        ref_max = np.max(x)

    try:
        info = np.iinfo(dtype)
    except ValueError as exc:
        raise ParameterError('dtype={} must be an unsigned integer type'.format(dtype)) from exc
    if info.kind != 'u':
        raise ParameterError('dtype={} must be an unsigned integer type'.format(dtype))

    x_quant = np.empty_like(x, dtype=np.dtype(dtype))

    bins = np.linspace(ref_min, ref_max, num=info.max - info.min + 1)
    x_quant[:] = np.digitize(x, bins, right=True)
    x_quant[x > ref_max] = info.max
    x_quant[x < ref_min] = info.min
    return x_quant
