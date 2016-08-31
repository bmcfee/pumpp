#!/usr/bin/env python

import numpy as np
import librosa

from .base import FeatureExtractor

__all__ = ['Tempogram', 'TempoScale']


class Tempogram(FeatureExtractor):

    def __init__(self, name, sr, hop_length, win_length):

        super(Tempogram, self).__init__(name, sr, hop_length)

        self.win_length = win_length

        self.register('tempogram', [None, win_length], np.float32)

    def transform_audio(self, y):
        tgram = librosa.feature.tempogram(y=y, sr=self.sr,
                                          hop_length=self.hop_length,
                                          win_length=self.win_length).astype(self.dtype)

        return {'tempogram': tgram.T}


class TempoScale(Tempogram):

    def __init__(self, name, sr, hop_length, win_length, n_fmt=128):

        super(TempoScale, self).__init__(name, sr, hop_length)

        self.n_fmt = n_fmt
        self.pop('tempogram')
        self.register('temposcale', [None, 1 + n_fmt // 2], np.float32)

    def transform_audio(self, y):

        data = super(TempoScale, self).transform_audio(y)
        data['temposcale'] = np.abs(librosa.fmt(data.pop('tempogram'),
                                                axis=0, n_fmt=n_fmt)).astype(self.dtype)
        return data
