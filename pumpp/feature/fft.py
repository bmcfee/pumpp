#!/usr/bin/env python
"""STFT feature extractors"""

import numpy as np
from librosa import stft, magphase
from librosa import amplitude_to_db, get_duration
from librosa.util import fix_length

from .base import FeatureExtractor
from ._utils import phase_diff, to_dtype

__all__ = ['STFT', 'STFTMag', 'STFTPhaseDiff']


class STFT(FeatureExtractor):
    '''Short-time Fourier Transform (STFT) with both magnitude
    and phase.

    Attributes
    ----------
    name : str
        The name of this transformer

    sr : number > 0
        The sampling rate of audio

    hop_length : int > 0
        The hop length of STFT frames

    n_fft : int > 0
        The number of FFT bins per frame

    log : bool
        If `True`, scale magnitude in decibels.

        Otherwise use linear magnitude.

    conv : str
        Convolution mode

    dtype : np.dtype
        The data type for the output features.  Default is `float32`.

        Setting to `uint8` will produce quantized features.

    See Also
    --------
    STFTMag
    STFTPhaseDiff
    '''
    def __init__(self, name='stft', sr=22050, hop_length=512, n_fft=2048, log=False,
                 conv=None, dtype='float32'):
        super(STFT, self).__init__(name, sr, hop_length, conv=conv, dtype=dtype)

        self.n_fft = n_fft
        self.log = log

        self.register('mag', 1 + n_fft // 2, self.dtype)
        self.register('phase', 1 + n_fft // 2, self.dtype)

    def transform_audio(self, y):
        '''Compute the STFT magnitude and phase.

        Parameters
        ----------
        y : np.ndarray
            The audio buffer

        Returns
        -------
        data : dict
            data['mag'] : np.ndarray, shape=(n_frames, 1 + n_fft//2)
                STFT magnitude

            data['phase'] : np.ndarray, shape=(n_frames, 1 + n_fft//2)
                STFT phase
        '''
        n_frames = self.n_frames(get_duration(y=y, sr=self.sr))

        D = stft(y, hop_length=self.hop_length,
                 n_fft=self.n_fft)

        D = fix_length(D, size=n_frames)

        mag, phase = magphase(D)
        if self.log:
            mag = amplitude_to_db(mag, ref=np.max)

        return {'mag': to_dtype(mag.T[self.idx], self.dtype),
                'phase': to_dtype(np.angle(phase.T)[self.idx], self.dtype)}


class STFTPhaseDiff(STFT):
    '''STFT with phase differentials

    See Also
    --------
    STFT
    '''
    def __init__(self, *args, **kwargs):
        super(STFTPhaseDiff, self).__init__(*args, **kwargs)
        phase_field = self.pop('phase')
        self.register('dphase', 1 + self.n_fft // 2, phase_field.dtype)

    def transform_audio(self, y):
        '''Compute the STFT magnitude and phase differential.

        Parameters
        ----------
        y : np.ndarray
            The audio buffer

        Returns
        -------
        data : dict
            data['mag'] : np.ndarray, shape=(n_frames, 1 + n_fft//2)
                STFT magnitude

            data['dphase'] : np.ndarray, shape=(n_frames, 1 + n_fft//2)
                STFT phase
        '''
        n_frames = self.n_frames(get_duration(y=y, sr=self.sr))

        D = stft(y, hop_length=self.hop_length,
                 n_fft=self.n_fft)

        D = fix_length(D, size=n_frames)

        mag, phase = magphase(D)
        if self.log:
            mag = amplitude_to_db(mag, ref=np.max)

        phase = phase_diff(np.angle(phase.T)[self.idx], self.conv)

        return {'mag': to_dtype(mag.T[self.idx], self.dtype),
                'dphase': to_dtype(phase, self.dtype)}


class STFTMag(STFT):
    '''STFT with only magnitude.

    See Also
    --------
    STFT
    '''
    def __init__(self, *args, **kwargs):
        super(STFTMag, self).__init__(*args, **kwargs)
        self.pop('phase')

    def transform_audio(self, y):
        '''Compute the STFT

        Parameters
        ----------
        y : np.ndarray
            The audio buffer

        Returns
        -------
        data : dict
            data['mag'] : np.ndarray, shape=(n_frames, 1 + n_fft//2)
                The STFT magnitude
        '''
        data = super(STFTMag, self).transform_audio(y)
        data.pop('phase')

        return data
