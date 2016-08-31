#!/usr/bin/env python

import numpy as np
import librosa

from .base import FeatureExtractor, phase_diff

__all__ = ['STFT', 'STFTMag', 'STFTPhaseDiff']


class STFT(FeatureExtractor):

    def __init__(self, name, sr, hop_length, n_fft):

        super(STFT, self).__init__(name, sr, hop_length)

        self.n_fft = n_fft

        self.register('mag', [None, 1 + n_fft // 2], np.float32)
        self.register('phase', [None, 1 + n_fft // 2], np.float32)

    def transform_audio(self, y):

        mag, phase = librosa.magphase(librosa.stft(y,
                                                   hop_length=self.hop_length,
                                                   n_fft=self.n_fft,
                                                   dtype=np.float32))
        return {'mag': mag.T, 'phase': np.angle(phase.T)}


class STFTPhaseDiff(STFT):

    def __init__(self, *args, **kwargs):

        super(STFTPhaseDiff, self).__init__(*args, **kwargs)
        phase_field = self.pop('phase')
        self.register('dphase', phase_field.shape, phase_field.dtype)

    def transform_audio(self, y):

        data = super(STFTPhaseDiff, self).transform_audio(y)
        data['dphase'] = phase_diff(data.pop('phase'), axis=0)
        return data


class STFTMag(STFT):

    def __init__(self, *args, **kwargs):

        super(STFTMag, self).__init__(*args, **kwargs)
        self.pop('phase')

    def transform_audio(self, y):

        data = super(STFTMag, self).transform_audio(y)
        data.pop('phase')

        return data
