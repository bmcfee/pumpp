#!/usr/bin/env python

import numpy as np
import librosa

from .base import FeatureExtractor

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

    def __init__(self, name, sr, hop_length, n_fft):

        super(STFTPhaseDiff, self).__init__(name, sr, hop_length, n_fft)
        phase_field = self.fields.pop(self.scope('phase'))
        self.register('dphase', phase_field.shape, phase_field.dtype)

    def transform_audio(self, y):

        data = super(STFTPhaseDiff, self).transform_audio(y)

        phase = data.pop('phase')

        # Compute the phase differential
        dphase = np.empty(phase.shape, np.float32)
        dphase[0] = 0.0
        dphase[1:] = np.diff(np.unwrap(phase, axis=0), axis=0)

        data['dphase'] = dphase
        return data


class STFTMag(STFT):

    def __init__(self, name, sr, hop_length, n_fft):

        super(STFTMag, self).__init__(name, sr, hop_length, n_fft)
        self.fields.pop(self.scope('phase'))

    def transform_audio(self, y):

        data = super(STFTMag, self).transform_audio(y)
        data.pop('phase')

        return data
    
