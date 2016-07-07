#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Feature extraction base class'''

import librosa

from ..core import Scope


class FeatureExtractor(Scope):

    def __init__(self, name, sr, hop_length):

        super(FeatureExtractor, self).__init__(name)

        self.sr = sr
        self.hop_length = hop_length

    def transform(self, y, sr):

        if sr != self.sr:
            y = librosa.resample(y, sr, self.sr)

        return self.merge([self.transform_audio(y)])

    def transform_audio(self, y):
        raise NotImplementedError
