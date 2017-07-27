#!/usr/bin/env python
'''Time features'''

import numpy as np

from librosa import get_duration

from .base import FeatureExtractor

__all__ = ['TimePosition']


class TimePosition(FeatureExtractor):
    '''TimePosition: encode frame position as features.

    Attributes
    ----------
    name : str
        The name of this feature extractor

    sr : number > 0
        The sampling rate of audio

    hop_length : int > 0
        The hop length of analysis windows
    '''

    def __init__(self, name, sr, hop_length, conv=None):
        super(TimePosition, self).__init__(name, sr, hop_length, conv=conv)

        self.register('relative', 2, np.float32)
        self.register('absolute', 2, np.float32)

    def transform_audio(self, y):
        '''Compute the time position encoding

        Parameters
        ----------
        y : np.ndarray
            Audio buffer

        Returns
        -------
        data : dict
            data['relative'] = np.ndarray, shape=(n_frames, 2)
            data['absolute'] = np.ndarray, shape=(n_frames, 2)

                Relative and absolute time positional encodings.
        '''

        duration = get_duration(y=y, sr=self.sr)
        n_frames = self.n_frames(duration)

        relative = np.zeros((n_frames, 2), dtype=np.float32)
        relative[:, 0] = np.cos(np.pi * np.linspace(0, 1, num=n_frames))
        relative[:, 1] = np.sin(np.pi * np.linspace(0, 1, num=n_frames))

        absolute = relative * np.sqrt(duration)

        return {'relative': relative[self.idx],
                'absolute': absolute[self.idx]}
