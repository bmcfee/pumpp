#!/usr/bin/env python
'''CQT features'''

import numpy as np
from librosa import cqt, magphase, note_to_hz, amplitude_to_db

from .base import FeatureExtractor

__all__ = ['CQT', 'CQTMag', 'CQTPhaseDiff']


class CQT(FeatureExtractor):
    '''Constant-Q transform

    Attributes
    ----------
    name : str
        The name for this feature extractor

    sr : number > 0
        The sampling rate of audio

    hop_length : int > 0
        The number of samples between CQT frames

    n_octaves : int > 0
        The number of octaves in the CQT

    over_sample : int > 0
        The amount of frequency oversampling (bins per semitone)

    fmin : float > 0
        The minimum frequency of the CQT

    log : boolean
        If `True`, scale the magnitude to decibels

        Otherwise, use linear magnitude

    '''
    def __init__(self, name, sr, hop_length, n_octaves=8, over_sample=3,
                 fmin=None, log=False, conv=None):
        super(CQT, self).__init__(name, sr, hop_length, conv=conv)

        if fmin is None:
            fmin = note_to_hz('C1')

        self.n_octaves = n_octaves
        self.over_sample = over_sample
        self.fmin = fmin
        self.log = log

        n_bins = n_octaves * 12 * over_sample
        self.register('mag', n_bins, np.float32)
        self.register('phase', n_bins, np.float32)

    def transform_audio(self, y):
        '''Compute the CQT

        Parameters
        ----------
        y : np.ndarray
            The audio buffer

        Returns
        -------
        data : dict
            data['mag'] : np.ndarray, shape = (n_frames, n_bins)
                The CQT magnitude

            data['phase']: np.ndarray, shape = mag.shape
                The CQT phase
        '''
        cqtm, phase = magphase(cqt(y=y,
                                   sr=self.sr,
                                   hop_length=self.hop_length,
                                   fmin=self.fmin,
                                   n_bins=(self.n_octaves *
                                           self.over_sample * 12),
                                   bins_per_octave=(self.over_sample * 12)))
        if self.log:
            cqtm = amplitude_to_db(cqtm, ref=np.max)

        return {'mag': cqtm.T.astype(np.float32)[self.idx],
                'phase': np.angle(phase).T.astype(np.float32)[self.idx]}


class CQTMag(CQT):
    '''Magnitude CQT

    See Also
    --------
    CQT
    '''

    def __init__(self, *args, **kwargs):
        super(CQTMag, self).__init__(*args, **kwargs)
        self.pop('phase')

    def transform_audio(self, y):
        '''Compute CQT magnitude.

        Parameters
        ----------
        y : np.ndarray
            the audio buffer

        Returns
        -------
        data : dict
            data['mag'] : np.ndarray, shape=(n_frames, n_bins)
                The CQT magnitude
        '''
        data = super(CQTMag, self).transform_audio(y)
        data.pop('phase')
        return data


class CQTPhaseDiff(CQT):
    '''CQT with unwrapped phase differentials

    See Also
    --------
    CQT
    '''
    def __init__(self, *args, **kwargs):
        super(CQTPhaseDiff, self).__init__(*args, **kwargs)
        phase_field = self.pop('phase')

        self.register('dphase',
                      self.n_octaves * 12 * self.over_sample,
                      phase_field.dtype)

    def transform_audio(self, y):
        '''Compute the CQT with unwrapped phase

        Parameters
        ----------
        y : np.ndarray
            The audio buffer

        Returns
        -------
        data : dict
            data['mag'] : np.ndarray, shape=(n_frames, n_bins)
                CQT magnitude

            data['dphase'] : np.ndarray, shape=(n_frames, n_bins)
                Unwrapped phase differential
        '''
        data = super(CQTPhaseDiff, self).transform_audio(y)
        data['dphase'] = self.phase_diff(data.pop('phase'))
        return data
