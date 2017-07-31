#!/usr/bin/env python
'''Beat-synchronizer'''

import numpy as np

from librosa import get_duration
from librosa.beat import beat_track

from .base import TimeSync


class BeatSync(TimeSync):

    def __init__(self, name, sr, hop_length, start_bpm=120):

        super(BeatSync, self).__init__(name)

        self.sr = sr
        self.hop_length = hop_length
        self.start_bpm = start_bpm

    def transform_audio(self, y):

        _, times = beat_track(y=y,
                              sr=self.sr,
                              hop_length=self.hop_length,
                              start_bpm=self.start_bpm,
                              trim=False, units='time')

        duration = get_duration(y=y, sr=self.sr)

        # Pad [0, duration]
        times = np.unique(np.concatenate([times, [0, duration]]))
        return {'_intervals': np.asarray([ival for ival in zip(times,
                                                               times[1:])])}
