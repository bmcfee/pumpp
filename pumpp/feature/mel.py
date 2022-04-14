#!/usr/bin/env python
"""Mel spectrogram"""

import numpy as np
from librosa.feature import melspectrogram
from librosa import amplitude_to_db, get_duration
from librosa.util import fix_length

from .base import FeatureExtractor

from ._utils import to_dtype

__all__ = ['Mel']


class Mel(FeatureExtractor):
    '''Mel spectra feature extraction

    Attributes
    ----------
    name : str or None
        naming scope for this feature extractor

    sr : number > 0
        Sampling rate of the audio (in Hz)

    hop_length : int > 0
        Number of samples to advance between frames

    n_fft :  int > 0
        Number of samples per frame

    n_mels : int > 0
        Number of Mel frequency bins

    fmax : number > 0
        The maximum frequency bin.
        Defaults to `0.5 * sr`

    log : bool
        If `True`, scale magnitude in decibels.

        Otherwise, use a linear amplitude scale.

    dtype : np.dtype
        The data type for the output features.  Default is `float32`.

        Setting to `uint8` will produce quantized features.
    '''
    def __init__(self, name='mel', sr=22050, hop_length=512, n_fft=2048, n_mels=128,
                 fmax=None, log=False, conv=None, dtype='float32'):
        super(Mel, self).__init__(name, sr, hop_length, conv=conv, dtype=dtype)

        self.n_fft = n_fft
        self.n_mels = n_mels
        self.fmax = fmax
        self.log = log

        self.register('mag', n_mels, self.dtype)

    def transform_audio(self, y):
        '''Compute the Mel spectrogram

        Parameters
        ----------
        y : np.ndarray
            The audio buffer

        Returns
        -------
        data : dict
            data['mag'] : np.ndarray, shape=(n_frames, n_mels)
                The Mel spectrogram
        '''
        n_frames = self.n_frames(get_duration(y=y, sr=self.sr))

        mel = np.sqrt(melspectrogram(y=y, sr=self.sr,
                                     n_fft=self.n_fft,
                                     hop_length=self.hop_length,
                                     n_mels=self.n_mels,
                                     fmax=self.fmax))

        mel = fix_length(mel, size=n_frames)

        if self.log:
            mel = amplitude_to_db(mel, ref=np.max)

        # Type convert
        mel = to_dtype(mel, self.dtype)

        return {'mag': mel.T[self.idx]}
