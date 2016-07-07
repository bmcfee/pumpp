
#!/usr/bin/env python

import numpy as np
import librosa

from .base import FeatureExtractor

__all__ = ['Mel']


class Mel(FeatureExtractor):

    def __init__(self, name, sr, hop_length, n_fft, n_mels, fmax=None):

        super(Mel, self).__init__(name, sr, hop_length)

        self.n_fft = n_fft
        self.n_mels = n_mels
        self.fmax = fmax

        self.register('mag', [None, n_mels], np.float32)

    def transform_audio(self, y):

        mel = np.sqrt(librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.n_fft,
                                                     hop_length=self.hop_length,
                                                     n_mels=self.n_mels,
                                                     fmax=self.fmax)).astype(np.float32)

        return {'mag': mel.T}
