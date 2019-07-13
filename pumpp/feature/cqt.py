#!/usr/bin/env python
'''CQT features'''

import numpy as np
from librosa import cqt, magphase, note_to_hz
from librosa import amplitude_to_db, get_duration
from librosa.util import fix_length

from ._utils import phase_diff, to_dtype
from .base import FeatureExtractor
from ..exceptions import ParameterError

__all__ = ['CQT', 'CQTMag', 'CQTPhaseDiff',
           'HCQT', 'HCQTMag', 'HCQTPhaseDiff']


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
                 fmin=None, log=False, conv=None, dtype='float32'):
        super(CQT, self).__init__(name, sr, hop_length, conv=conv, dtype=dtype)

        if fmin is None:
            fmin = note_to_hz('C1')

        self.n_octaves = n_octaves
        self.over_sample = over_sample
        self.fmin = fmin
        self.log = log

        n_bins = n_octaves * 12 * over_sample
        self.register('mag', n_bins, self.dtype)
        self.register('phase', n_bins, self.dtype)

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
        n_frames = self.n_frames(get_duration(y=y, sr=self.sr))

        C = cqt(y=y, sr=self.sr, hop_length=self.hop_length,
                fmin=self.fmin,
                n_bins=(self.n_octaves * self.over_sample * 12),
                bins_per_octave=(self.over_sample * 12))

        C = fix_length(C, n_frames)

        cqtm, phase = magphase(C)
        if self.log:
            cqtm = amplitude_to_db(cqtm, ref=np.max)

        return {'mag': to_dtype(cqtm.T[self.idx], self.dtype),
                'phase': to_dtype(np.angle(phase).T[self.idx], self.dtype)}


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
        n_frames = self.n_frames(get_duration(y=y, sr=self.sr))

        C = cqt(y=y, sr=self.sr, hop_length=self.hop_length,
                fmin=self.fmin,
                n_bins=(self.n_octaves * self.over_sample * 12),
                bins_per_octave=(self.over_sample * 12))

        C = fix_length(C, n_frames)

        cqtm, phase = magphase(C)
        if self.log:
            cqtm = amplitude_to_db(cqtm, ref=np.max)

        dphase = phase_diff(np.angle(phase).T[self.idx], self.conv)

        return {'mag': to_dtype(cqtm.T[self.idx], self.dtype),
                'dphase': to_dtype(dphase, self.dtype)}


class HCQT(FeatureExtractor):
    '''Harmonic Constant-Q transform

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

    harmonics : list of int >= 1
        The list of harmonics to compute

    log : boolean
        If `True`, scale the magnitude to decibels

        Otherwise, use linear magnitude

    conv : {'tf', 'th', 'channels_last', 'channels_first', None}
        convolution dimension ordering:

            - 'channels_last' for tensorflow-style 2D convolution
            - 'tf' equivalent to 'channels_last'
            - 'channels_first' for theano-style 2D convolution
            - 'th' equivalent to 'channels_first'

    dtype : np.dtype
        The data type for the output features.  Default is `float32`.

        Setting to `uint8` will produce quantized features.
    '''
    def __init__(self, name, sr, hop_length, n_octaves=8, over_sample=3,
                 fmin=None, harmonics=None, log=False, conv='channels_last',
                 dtype='float32'):

        if conv not in ('channels_last', 'tf', 'channels_first', 'th'):
            raise ParameterError('Invalid conv={}'.format(conv))

        super(HCQT, self).__init__(name, sr, hop_length, conv=conv, dtype=dtype)

        if fmin is None:
            fmin = note_to_hz('C1')

        if harmonics is None:
            harmonics = [1]
        else:
            harmonics = list(harmonics)
            if not all(isinstance(_, int) and _ > 0 for _ in harmonics):
                raise ParameterError('Invalid harmonics={}'.format(harmonics))

        self.n_octaves = n_octaves
        self.over_sample = over_sample
        self.fmin = fmin
        self.log = log
        self.harmonics = harmonics

        n_bins = n_octaves * 12 * over_sample
        self.register('mag', n_bins, self.dtype, channels=len(harmonics))
        self.register('phase', n_bins, self.dtype, channels=len(harmonics))

    def transform_audio(self, y):
        '''Compute the HCQT

        Parameters
        ----------
        y : np.ndarray
            The audio buffer

        Returns
        -------
        data : dict
            data['mag'] : np.ndarray, shape = (n_frames, n_bins, n_harmonics)
                The CQT magnitude

            data['phase']: np.ndarray, shape = mag.shape
                The CQT phase
        '''
        cqtm, phase = [], []

        n_frames = self.n_frames(get_duration(y=y, sr=self.sr))

        for h in self.harmonics:
            C = cqt(y=y, sr=self.sr, hop_length=self.hop_length,
                    fmin=self.fmin * h,
                    n_bins=(self.n_octaves * self.over_sample * 12),
                    bins_per_octave=(self.over_sample * 12))

            C = fix_length(C, n_frames)

            C, P = magphase(C)
            if self.log:
                C = amplitude_to_db(C, ref=np.max)
            cqtm.append(C)
            phase.append(P)

        cqtm = to_dtype(np.asarray(cqtm), self.dtype)
        phase = to_dtype(np.angle(np.asarray(phase)), self.dtype)

        return {'mag': self._index(cqtm),
                'phase': self._index(phase)}

    def _index(self, value):
        '''Rearrange a tensor according to the convolution mode

        Input is assumed to be in (channels, bins, time) format.
        '''

        if self.conv in ('channels_last', 'tf'):
            return np.transpose(value, (2, 1, 0))

        else:  # self.conv in ('channels_first', 'th')
            return np.transpose(value, (0, 2, 1))


class HCQTMag(HCQT):
    '''Magnitude HCQT

    See Also
    --------
    HCQT
    '''

    def __init__(self, *args, **kwargs):
        super(HCQTMag, self).__init__(*args, **kwargs)
        self.pop('phase')

    def transform_audio(self, y):
        '''Compute HCQT magnitude.

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
        data = super(HCQTMag, self).transform_audio(y)
        data.pop('phase')
        return data


class HCQTPhaseDiff(HCQT):
    '''HCQT with unwrapped phase differentials

    See Also
    --------
    HCQT
    '''
    def __init__(self, *args, **kwargs):
        super(HCQTPhaseDiff, self).__init__(*args, **kwargs)
        phase_field = self.pop('phase')

        self.register('dphase',
                      self.n_octaves * 12 * self.over_sample,
                      phase_field.dtype,
                      channels=len(self.harmonics))

    def transform_audio(self, y):
        '''Compute the HCQT with unwrapped phase

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
        data = super(HCQTPhaseDiff, self).transform_audio(y)
        data['dphase'] = phase_diff(data.pop('phase'), self.conv)
        return data

    def transform_audio(self, y):
        '''Compute the HCQT

        Parameters
        ----------
        y : np.ndarray
            The audio buffer

        Returns
        -------
        data : dict
            data['mag'] : np.ndarray, shape = (n_frames, n_bins, n_harmonics)
                The CQT magnitude

            data['phase']: np.ndarray, shape = mag.shape
                The CQT phase
        '''
        cqtm, phase = [], []

        n_frames = self.n_frames(get_duration(y=y, sr=self.sr))

        for h in self.harmonics:
            C = cqt(y=y, sr=self.sr, hop_length=self.hop_length,
                    fmin=self.fmin * h,
                    n_bins=(self.n_octaves * self.over_sample * 12),
                    bins_per_octave=(self.over_sample * 12))

            C = fix_length(C, n_frames)

            C, P = magphase(C)
            if self.log:
                C = amplitude_to_db(C, ref=np.max)
            cqtm.append(C)
            phase.append(P)

        cqtm = to_dtype(np.asarray(cqtm), self.dtype)
        phase = np.angle(np.asarray(phase))

        dphase = to_dtype(phase_diff(self._index(phase), self.conv),
                          self.dtype)

        return {'mag': self._index(cqtm),
                'dphase': dphase}
