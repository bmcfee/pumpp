#!/usr/bin/env python
'''CQT features'''

import numpy as np
import librosa

from .base import FeatureExtractor, phase_diff

__all__ = ['CQT', 'CQTMag', 'CQTPhaseDiff']


class CQT(FeatureExtractor):

    def __init__(self, name, sr, hop_length, n_octaves=8, over_sample=3, fmin=None):

        super(CQT, self).__init__(name, sr, hop_length)

        if fmin is None:
            fmin = librosa.note_to_hz('C1')

        self.n_octaves = n_octaves
        self.over_sample = over_sample
        self.fmin = fmin

        self.register('mag', [None, n_octaves * 12 * over_sample], np.float32)
        self.register('phase', [None, n_octaves * 12 * over_sample], np.float32)

    def transform_audio(self, y):

        cqt, phase = librosa.magphase(librosa.cqt(y=y,
                                                  sr=self.sr,
                                                  hop_length=self.hop_length,
                                                  fmin=self.fmin,
                                                  n_bins=self.n_octaves *
                                                         self.over_sample * 12,
                                                  bins_per_octave=self.over_sample * 12,
                                                  real=False))

        return {'mag': cqt.T.astype(np.float32),
                'phase': np.angle(phase).T.astype(np.float32)}


class CQTMag(CQT):

    def __init__(self, *args, **kwargs):

        super(CQTMag, self).__init__(*args, **kwargs)
        self.fields.pop(self.scope('phase'))

    def transform_audio(self, y):

        data = super(CQTMag, self).transform_audio(y)
        data.pop('phase')
        return data


class CQTPhaseDiff(CQT):

    def __init__(self, *args, **kwargs):

        super(CQTPhaseDiff, self).__init__(*args, **kwargs)
        phase_field = self.fields.pop(self.scope('phase'))
        self.register('dphase', phase_field.shape, phase_field.dtype)

    def transform_audio(self, y):
        data = super(CQTPhaseDiff, self).transform_audio(y)
        data['dphase'] = phase_diff(data.pop('phase'), axis=0)
        return data
