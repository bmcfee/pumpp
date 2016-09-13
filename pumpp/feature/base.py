#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Feature extraction base class'''

import numpy as np
import librosa

from ..base import Scope


class FeatureExtractor(Scope):
    '''The base feature extractor class.

    Attributes
    ----------
    name : str
        The name for this feature extractor

    sr : number > 0
        The sampling rate of audio for analysis

    hop_length : int > 0
        The hop length between analysis frames
    '''
    def __init__(self, name, sr, hop_length):

        super(FeatureExtractor, self).__init__(name)

        self.sr = sr
        self.hop_length = hop_length

    def transform(self, y, sr):
        '''Transform an audio signal

        Parameters
        ----------
        y : np.ndarray
            The audio signal

        sr : number > 0
            The native sampling rate of y

        Returns
        -------
        dict
            Data dictionary containing features extracted from y

        See Also
        --------
        transform_audio
        '''
        if sr != self.sr:
            y = librosa.resample(y, sr, self.sr)

        return self.merge([self.transform_audio(y)])

    def transform_audio(self, y):
        raise NotImplementedError


def phase_diff(phase, axis=0):
    '''Compute the phase differential along a given axis

    Parameters
    ----------
    phase : np.ndarray
        Input phase (in radians)

    axis : int
        The axis along which to differentiate

    Returns
    -------
    dphase : np.ndarray like `phase`
        The phase differential.
    '''

    # Compute the phase differential
    dphase = np.empty(phase.shape, dtype=phase.dtype)
    zero_idx = [slice(None)] * phase.ndim
    zero_idx[axis] = slice(1)
    else_idx = [slice(None)] * phase.ndim
    else_idx[axis] = slice(1, None)
    dphase[zero_idx] = phase[zero_idx]
    dphase[else_idx] = np.diff(np.unwrap(phase, axis=axis), axis=axis)
    return dphase
