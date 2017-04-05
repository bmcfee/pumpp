#!/usr/bin/env python
"""Mel spectrogram"""

import numpy as np
from librosa.feature import melspectrogram
from librosa import amplitude_to_db, get_duration
from librosa.util import fix_length

from .base import FeatureExtractor

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
    '''
    def __init__(self, name, sr, hop_length, n_fft, n_mels, fmax=None,
                 log=False, conv=None):
        super(Mel, self).__init__(name, sr, hop_length, conv=conv)

        self.n_fft = n_fft
        self.n_mels = n_mels
        self.fmax = fmax
        self.log = log

        self.register('mag', n_mels, np.float32)

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
                                     fmax=self.fmax)).astype(np.float32)

        mel = fix_length(mel, n_frames)

        if self.log:
            mel = amplitude_to_db(mel, ref=np.max)

        return {'mag': mel.T[self.idx]}
