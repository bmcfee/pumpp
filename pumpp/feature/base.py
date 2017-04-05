#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Feature extraction base class'''

import numpy as np
from librosa import resample, time_to_frames

from ..base import Scope
from ..exceptions import ParameterError


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

    conv : {'tf', 'th', 'channels_last', 'channels_first', None}
        convolution dimension ordering:

            - 'channels_last' for tensorflow-style 2D convolution
            - 'tf' equivalent to 'channels_last'
            - 'channels_first' for theano-style 2D convolution
            - 'th' equivalent to 'channels_first'
            - None for 1D or non-convolutional representations
    '''
    def __init__(self, name, sr, hop_length, conv=None):

        super(FeatureExtractor, self).__init__(name)

        if conv not in ('tf', 'th', 'channels_last', 'channels_first', None):
            raise ParameterError('conv="{}", must be one of '
                                 '("channels_last", "tf", '
                                 '"channels_first", "th", None)'.format(conv))

        self.sr = sr
        self.hop_length = hop_length
        self.conv = conv

    def register(self, key, dimension, dtype, channels=1):

        shape = [None, dimension]

        if self.conv in ('channels_last', 'tf'):
            shape.append(channels)

        elif self.conv in ('channels_first', 'th'):
            shape.insert(0, channels)

        super(FeatureExtractor, self).register(key, shape, dtype)

    @property
    def idx(self):
        if self.conv is None:
            return Ellipsis

        elif self.conv in ('channels_last', 'tf'):
            return (slice(None), slice(None), np.newaxis)

        elif self.conv in ('channels_first', 'th'):
            return (np.newaxis, slice(None), slice(None))

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
            y = resample(y, sr, self.sr)

        return self.merge([self.transform_audio(y)])

    def transform_audio(self, y):
        raise NotImplementedError

    def phase_diff(self, phase):
        '''Compute the phase differential along a given axis

        Parameters
        ----------
        phase : np.ndarray
            Input phase (in radians)

        Returns
        -------
        dphase : np.ndarray like `phase`
            The phase differential.
        '''

        if self.conv is None:
            axis = 0
        elif self.conv in ('channels_last', 'tf'):
            axis = 0
        elif self.conv in ('channels_first', 'th'):
            axis = 1

        # Compute the phase differential
        dphase = np.empty(phase.shape, dtype=phase.dtype)
        zero_idx = [slice(None)] * phase.ndim
        zero_idx[axis] = slice(1)
        else_idx = [slice(None)] * phase.ndim
        else_idx[axis] = slice(1, None)
        dphase[zero_idx] = phase[zero_idx]
        dphase[else_idx] = np.diff(np.unwrap(phase, axis=axis), axis=axis)
        return dphase

    def layers(self):
        '''Construct Keras input layers for the given transformer

        Returns
        -------
        layers : {field: keras.layers.Input}
            A dictionary of keras input layers, keyed by the corresponding
            field keys.
        '''
        from keras.layers import Input

        L = dict()
        for key in self.fields:
            L[key] = Input(name=key,
                           shape=self.fields[key].shape,
                           dtype=self.fields[key].dtype)

        return L

    def n_frames(self, duration):
        '''Get the number of frames for a given duration

        Parameters
        ----------
        duration : number >= 0
            The duration, in seconds

        Returns
        -------
        n_frames : int >= 0
            The number of frames at this extractor's sampling rate and
            hop length
        '''

        return int(time_to_frames(duration, sr=self.sr,
                                  hop_length=self.hop_length))
