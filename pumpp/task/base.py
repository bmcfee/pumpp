#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''The base class for task transformer objects'''

import numpy as np
import librosa


__all__ = ['BaseTaskTransformer']

class BaseTaskTransformer(object):
    '''Base class for task transformer objects'''

    def __init__(self, namespace, fill_na, sr, hop_length):
        self.namespace = namespace

        if fill_na is None:
            fill_na = np.nan

        self.fill_na = fill_na
        self.sr = sr
        self.hop_length = hop_length

    def find_annotation(self, jam):
        '''Retrieve a random annotation matching the target namespace
        
        Parameters
        ----------
        jam : jams.JAMS
            The JAMS object to query

        Returns
        -------
        ann : jams.Annotation or None
            If any annotations matching this task's namespace can be found,
            a random one is selected.

            Otherwise, `None` is returned.
        '''
        anns = jam.search(namespace=self.namespace)

        if anns:
            i = np.random.choice(len(anns))
            return anns[i]

        # FIXME: raise an exception instead of returning None
        return None

    def encode_events(self, duration, events, values):
        '''Encode labeled events as a time-series matrix.

        Parameters
        ----------
        duration : number
            The duration of the track

        events : ndarray, shape=(n,)
            Time index of the events

        values : ndarray, shape=(n, m)
            Values array.  Must have the same first index as `events`.

        Returns
        -------
        target : ndarray, shape=(n_frames, n_values)
        '''

        # FIXME: support sparse encoding
        # FIXME: support non-bool dtypes
        # FIXME: it's not clear what type `values` is here
        frames = librosa.time_to_frames(events,
                                        sr=self.sr,
                                        hop_length=self.hop_length)

        n_total = librosa.time_to_frames(duration,
                                         sr=self.sr,
                                         hop_length=self.hop_length)

        target = np.empty((n_total, values.shape[1]),
                          dtype=values.dtype)

        target.fill(self.fill_na)

        for column, event in zip(values, frames):
            target[event] = column

        return target.astype(np.bool)

    def encode_intervals(self, duration, intervals, values):

        frames = librosa.time_to_frames(intervals,
                                        sr=self.sr,
                                        hop_length=self.hop_length)

        n_total = librosa.time_to_frames(duration,
                                         sr=self.sr,
                                         hop_length=self.hop_length)

        target = np.empty((n_total, values.shape[-1]),
                          dtype=values.dtype)

        target.fill(self.fill_na)

        for column, interval in zip(values, frames):
            target[interval[0]:interval[1]] += column

        return target.astype(np.bool)
