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

    dtype : str or np.dtype
        The data type for features produced by this object.  Default is`float32`.

        Setting to `uint8` will produced quantized features.

    '''
    def __init__(self, name, sr, hop_length, conv=None, dtype='float32'):

        super(FeatureExtractor, self).__init__(name)

        if conv not in ('tf', 'th', 'channels_last', 'channels_first', None):
            raise ParameterError('conv="{}", must be one of '
                                 '("channels_last", "tf", '
                                 '"channels_first", "th", None)'.format(conv))

        self.sr = sr
        self.hop_length = hop_length
        self.conv = conv
        self.dtype = np.dtype(dtype)

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

    def layers(self, api='keras'):
        '''Construct input layers for the given transformer

        Parameters
        ----------

        Returns
        -------
        layers : {field: keras.layers.Input}
            A dictionary of keras input layers, keyed by the corresponding
            field keys.
        '''

        if api == 'keras':
            return self.layers_keras()
        elif api in ('tf', 'tensorflow'):
            return self.layers_tensorflow()
        else:
            raise ParameterError('Unsupported layer api={}'.format(api))

    def layers_tensorflow(self):
        from tensorflow import placeholder

        L = dict()
        for key in self.fields:
            shape = tuple([None] + list(self.fields[key].shape))
            L[key] = placeholder(self.fields[key].dtype,
                                 shape=shape, name=key)
        return L

    def layers_keras(self):
        from keras.layers import Input

        L = dict()
        for key in self.fields:
            L[key] = Input(name=key,
                           shape=self.fields[key].shape,
                           dtype=np.dtype(self.fields[key].dtype).name)

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
